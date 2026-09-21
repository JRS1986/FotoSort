"""Local review records and report access, independent of models and the CLI."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

STORE_NAME = ".fotosort_review.json"
REVIEW_FIELDS = ("photo_id", "photo_sha256", "auto_selected", "auto_decision", "manual_decision",
                 "manual_reason", "review_status", "review_revision")


class ReviewError(ValueError):
    """Invalid input or unavailable review data."""


class Conflict(ReviewError):
    """A source, report, or review revision changed since it was read."""


def relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReviewError("Expected a path relative to the collection")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or re.fullmatch(r"[A-Za-z]:", path.parts[0])):
        raise ReviewError("Path must stay inside the collection")
    return path.as_posix()


def inside(root: Path, value: str) -> Path:
    path = root / relative_name(value)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ReviewError("Path resolves outside the collection")
    return path


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise Conflict("Photo changed while reading; reload the collection")
    return digest


def stamp(path: Path) -> tuple[int, int]:
    """Size and modification time: a cheap signal that verified contents are unchanged."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def photo_id(relative: str, digest: str) -> str:
    return hashlib.sha256(f"{relative_name(relative)}\0{digest}".encode()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}-", delete=False) as stream:
            name = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # NamedTemporaryFile creates 0600 files; reports keep their mode or follow the umask.
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def empty_state() -> dict:
    return dict(version=1, revision=0, decisions={}, preferences=[], alternatives={}, history=[])


def validate_feedback(state: dict) -> None:
    def valid_id(value):
        return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None

    if (not isinstance(state["decisions"], dict) or not isinstance(state["preferences"], list)
            or not isinstance(state["alternatives"], dict)):
        raise ValueError("invalid feedback")
    for key, decision in state["decisions"].items():
        if (not valid_id(decision["sha256"]) or not isinstance(decision.get("note", ""), str)
                or decision["choice"] not in {"keep", "reject"}
                or key != photo_id(decision["relative_file"], decision["sha256"])):
            raise ValueError("invalid decision")
    for preference in state["preferences"]:
        if (not valid_id(preference["winner"]) or not valid_id(preference["loser"])
                or preference["winner"] == preference["loser"]):
            raise ValueError("invalid preference")
    for name, ids in state["alternatives"].items():
        if not name or not isinstance(ids, list) or not all(valid_id(i) for i in ids) or len(set(ids)) < 2:
            raise ValueError("invalid alternatives")


FEEDBACK_KEYS = ("decisions", "preferences", "alternatives")


def inverse_delta(before: dict, after: dict) -> dict:
    """The previous value of every changed entry, so undo never stores whole copies of the state."""
    delta = {}
    for key in ("decisions", "alternatives"):
        changed = {name: copy.deepcopy(before[key].get(name)) for name in before[key].keys() | after[key].keys()
                   if before[key].get(name) != after[key].get(name)}
        if changed:
            delta[key] = changed
    if before["preferences"] != after["preferences"]:
        delta["preferences"] = copy.deepcopy(before["preferences"])
    return delta


def apply_delta(state: dict, delta: dict) -> None:
    for key in ("decisions", "alternatives"):
        for name, previous in delta.get(key, {}).items():
            if previous is None:
                state[key].pop(name, None)
            else:
                state[key][name] = previous
    if "preferences" in delta:
        state["preferences"] = delta["preferences"]


def validate_history(entry: dict) -> None:
    if set(entry) == set(FEEDBACK_KEYS):  # a whole-state snapshot written by an earlier preview
        return validate_feedback(entry)
    if set(entry) != {"delta"} or not isinstance(entry["delta"], dict) or set(entry["delta"]) - set(FEEDBACK_KEYS):
        raise ValueError("invalid history")
    delta = entry["delta"]
    present = dict(decisions={k: v for k, v in delta.get("decisions", {}).items() if v is not None},
                   preferences=delta.get("preferences", []),
                   alternatives={k: v for k, v in delta.get("alternatives", {}).items() if v is not None})
    validate_feedback(present)


class ReviewStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = inside(self.root, STORE_NAME)

    def load(self) -> dict:
        inside(self.root, STORE_NAME)
        if not self.path.exists():
            return empty_state()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if (state["version"] != 1 or type(state["revision"]) is not int or state["revision"] < 0
                    or not isinstance(state["decisions"], dict) or not isinstance(state["preferences"], list)
                    or not isinstance(state["alternatives"], dict) or not isinstance(state["history"], list)):
                raise ValueError("invalid schema")
            validate_feedback(state)
            if len(state["history"]) > 50:
                raise ValueError("invalid history")
            for previous in state["history"]:
                validate_history(previous)
            return state
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ReviewError("Invalid or unsupported review file; restore a valid copy before continuing") from exc

    @contextmanager
    def locked(self):
        # POSIX advisory locks are released by the OS even if the process dies.
        import fcntl

        with inside(self.root, ".fotosort_review.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def update(self, revision: int, mutate, *, undo: bool = False) -> dict:
        with self.locked():
            state = self.load()
            if type(revision) is not int or revision != state["revision"]:
                raise Conflict("Review changed in another session; reload before saving")
            if undo:
                if not state["history"]:
                    raise ReviewError("Nothing to undo")
                previous = state["history"].pop()
                if "delta" in previous:
                    apply_delta(state, previous["delta"])
                else:
                    state.update(previous)
            else:
                before = copy.deepcopy({k: state[k] for k in FEEDBACK_KEYS})
                mutate(state)
                state["history"] = (state["history"] + [dict(delta=inverse_delta(before, state))])[-50:]
            state["revision"] += 1
            inside(self.root, STORE_NAME)  # recheck a replaced symlink before writing
            atomic_write(self.path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
            return state


def read_report(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if "file" not in fields or "selected" not in fields or len(set(fields)) != len(fields):
            raise ReviewError("Report needs unique column names including file and selected")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) or row["selected"] not in {"0", "1"}
           or ("auto_selected" in row and row["auto_selected"] not in {"0", "1"})
           for row in rows):
        raise ReviewError("Malformed report row or selection; selected must be 0 or 1")
    return fields, rows


ORIGINS = ("explicit review", "csv import")


def manual_text(choice: str, reason: str) -> str:
    return f"Manual {choice}: {reason}" if reason else f"Manual {choice}"


def moved_decisions(state: dict, current: dict[str, str]) -> dict[str, str]:
    """Report decisions whose path left the collection while one unannotated photo has the same contents.

    Such a decision is never transferred silently; the status names where it was recorded."""
    annotated = {d["relative_file"] for d in state["decisions"].values()}
    orphans: dict[str, list[str]] = {}
    for decision in state["decisions"].values():
        if decision["relative_file"] not in current:
            orphans.setdefault(decision["sha256"], []).append(decision["relative_file"])
    matches: dict[str, list[str]] = {}
    for relative, digest in current.items():
        if digest in orphans and relative not in annotated:
            matches.setdefault(digest, []).append(relative)
    return {relatives[0]: f"decision recorded for {orphans[digest][0]}; not transferred"
            for digest, relatives in matches.items() if len(relatives) == 1 and len(orphans[digest]) == 1}


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue())


class Collection:
    """A frozen CSV snapshot with verified identities and confined image paths."""

    def __init__(self, root: Path, report: str = "fotosort_report.csv", *, verify_sources: bool = True):
        """verify_sources=False trusts the report's recorded hashes until a photo is actually used:
        `verify` still hashes every photo before a decision or export relies on it."""
        self.root = root.expanduser().resolve()
        self.report = inside(self.root, report)
        self.report_stamp = stamp(self.report)
        self.report_digest = fingerprint(self.report)
        self.fields, self.rows = read_report(self.report)
        self.store = ReviewStore(self.root)
        self.entries: dict[str, dict] = {}
        absolute = [Path(r["file"]) for r in self.rows if Path(r["file"]).is_absolute()]
        old_root = Path(os.path.commonpath([p.parent for p in absolute])) if absolute else None
        roots = set()
        for row in self.rows:
            if row.get("relative_file") and Path(row["file"]).is_absolute():
                rel = PurePosixPath(relative_name(row["relative_file"]))
                original = Path(row["file"])
                if original.parts[-len(rel.parts):] == rel.parts:
                    roots.add(original.parents[len(rel.parts) - 1])
        if len(roots) == 1:
            old_root = roots.pop()
        seen = set()
        for row in self.rows:
            relative = row.get("relative_file")
            if not relative:
                original = Path(row["file"])
                if original.is_absolute():
                    base = self.root if original.is_relative_to(self.root) else old_root
                    if base is None or not original.is_relative_to(base):
                        raise ReviewError(f"Report path is outside the collection: {original.name}")
                    relative = original.relative_to(base).as_posix()
                else:
                    relative = original.as_posix()
            relative = relative_name(relative)
            if relative in seen:
                raise ReviewError(f"Ambiguous report: repeated relative path {relative}")
            seen.add(relative)
            path = inside(self.root, relative)
            expected = row.get("photo_sha256", "")
            digest, status, verified = expected, "available", None
            try:
                verified = stamp(path)
                if verify_sources or not expected:
                    digest = fingerprint(path)
                    if expected and digest != expected:
                        status = "changed"
                else:
                    verified = None  # not hashed yet; verify() does so on first use
            except OSError:
                status = "missing"
            identifier = photo_id(relative, digest or "missing")
            source = path
            source_note = ""
            raw_source = row.get("source")
            if raw_source and raw_source != row["file"]:
                candidate = Path(raw_source)
                if candidate.is_absolute() and candidate.is_relative_to(self.root):
                    pass
                elif candidate.is_absolute() and old_root and candidate.is_relative_to(old_root):
                    candidate = self.root / candidate.relative_to(old_root)
                elif not candidate.is_absolute():
                    candidate = self.root / candidate
                if candidate.resolve().is_relative_to(self.root) and candidate.is_file():
                    source = candidate
                else:
                    source_note = "Processed source unavailable inside this collection; showing original"
            self.entries[identifier] = dict(id=identifier, relative_file=relative, sha256=digest, status=status,
                                            path=path, source=source, source_note=source_note, row=row,
                                            stamp=verified)
        self.ensure_current()

    def ensure_current(self):
        """Contents are re-hashed only when size or modification time moved since the last check."""
        try:
            current = stamp(self.report)
            if current != self.report_stamp:
                if fingerprint(self.report) != self.report_digest:
                    raise Conflict("Report changed; reopen the reviewer to use the new analysis")
                self.report_stamp = current
        except OSError as exc:
            raise Conflict("Report is no longer available; reopen the reviewer") from exc

    def verify(self, identifier: str) -> dict:
        entry = self.entries.get(identifier)
        if entry is None:
            raise ReviewError("Unknown photo")
        path = inside(self.root, entry["relative_file"])
        try:
            current = stamp(path) if entry["status"] == "available" and path.is_file() else None
            if current is not None and current != entry["stamp"]:
                if fingerprint(path) != entry["sha256"]:
                    current = None
                else:
                    entry["stamp"] = current
        except OSError:
            current = None
        if current is None:
            raise Conflict(f"Photo is missing or changed: {entry['relative_file']}; rerun analysis before reviewing it")
        return entry

    def view(self, state: dict | None = None) -> list[dict]:
        state = state if state is not None else self.store.load()
        result = []
        annotated_paths = {d["relative_file"] for d in state["decisions"].values()}
        moved = moved_decisions(state, {e["relative_file"]: e["sha256"] for e in self.entries.values()})
        for entry in self.entries.values():
            row = entry["row"]
            decision = state["decisions"].get(entry["id"]) if entry["status"] == "available" else None
            auto = row.get("auto_selected", row["selected"]) == "1"
            # A legacy CSV's selected column remains its starting recommendation.
            choice = decision["choice"] if decision else ""
            stale = entry["relative_file"] in annotated_paths and decision is None
            result.append(dict(id=entry["id"], relative_file=entry["relative_file"], sha256=entry["sha256"],
                               status=entry["status"], source_note=entry["source_note"], auto_selected=auto,
                               selected=(choice == "keep") if choice else auto, manual_decision=choice,
                               manual_reason=decision.get("note", "") if decision else "",
                               review_status="stale decision" if stale and not decision
                               else moved.get(entry["relative_file"], entry["status"]),
                               clearable=entry["relative_file"] in annotated_paths,
                               **{k: row.get(k, "") for k in ("day", "datetime", "label", "judge_label", "scene",
                                   "moment", "burst", "score", "shortlisted", "judge_reason", "preselection_reason")},
                               auto_decision=row.get("auto_decision", row.get("decision", ""))))
        return result

    def change(self, revision: int, changes: dict[str, str], note: str = "", *, preference=None,
               alternatives=None, origin: str = "explicit review") -> dict:
        if not isinstance(changes, dict) or not isinstance(note, str) or len(note) > 4000:
            raise ReviewError("Invalid decisions or note")
        if origin not in ORIGINS:
            raise ReviewError("Unknown decision origin")

        def mutate(state):
            self.ensure_current()
            now = datetime.now(UTC).isoformat()
            for identifier, choice in changes.items():
                if choice not in {"keep", "reject", "clear"}:
                    raise ReviewError("Decision must be keep, reject, or clear")
                if choice == "clear":
                    entry = self.entries.get(identifier)
                    if entry is None:
                        raise ReviewError("Unknown photo")
                    state["decisions"] = {key: d for key, d in state["decisions"].items()
                                          if d["relative_file"] != entry["relative_file"]}
                else:
                    entry = self.verify(identifier)
                    state["decisions"][identifier] = dict(relative_file=entry["relative_file"],
                        sha256=entry["sha256"], choice=choice, note=note, updated_at=now, origin=origin)
            if preference is not None:
                if not isinstance(preference, list) or len(preference) != 2 or preference[0] == preference[1]:
                    raise ReviewError("Preference needs two different photos: winner, loser")
                for identifier in preference:
                    self.verify(identifier)
                # A changed mind replaces the earlier verdict on the same pair instead of contradicting it.
                state["preferences"] = [p for p in state["preferences"]
                                        if {p["winner"], p["loser"]} != set(preference)]
                state["preferences"].append(dict(winner=preference[0], loser=preference[1], note=note, updated_at=now))
            if alternatives is not None:
                name, identifiers = alternatives
                if not isinstance(name, str) or not name.strip() or len(name) > 100:
                    raise ReviewError("An alternative group needs a name of at most 100 characters")
                if not isinstance(identifiers, list) or len(set(identifiers)) < 2:
                    raise ReviewError("An alternative group needs at least two different photos")
                for identifier in identifiers:
                    self.verify(identifier)
                state["alternatives"][name] = list(dict.fromkeys(identifiers))
        return self.store.update(revision, mutate)

    def undo(self, revision: int) -> dict:
        self.ensure_current()
        return self.store.update(revision, lambda state: None, undo=True)

    def export(self, output: str, revision: int) -> Path:
        destination = inside(self.root, output)
        if destination.suffix.lower() != ".csv":
            raise ReviewError("Export must be a CSV inside the collection")
        if destination.exists() and destination.samefile(self.report):
            raise ReviewError("Export would overwrite the report being reviewed; choose another output name")
        with self.store.locked():
            state = self.store.load()
            if revision != state["revision"]:
                raise Conflict("Review changed; reload before exporting")
            self.ensure_current()
            rows = []
            for view in self.view(state):
                entry = self.entries[view["id"]]
                # Missing or changed picks stay visible through review_status (and the recorded hash)
                # instead of blocking the export; apply-report counts or refuses them explicitly.
                if view["selected"] and entry["status"] == "available":
                    self.verify(view["id"])
                row = dict(entry["row"])
                row.update(file=str(entry["path"]), relative_file=entry["relative_file"],
                           photo_id=view["id"], photo_sha256=entry["row"].get("photo_sha256", entry["sha256"])
                           if entry["status"] == "changed" else entry["sha256"],
                           auto_selected=str(int(view["auto_selected"])), auto_decision=view["auto_decision"],
                           selected=str(int(view["selected"])), manual_decision=view["manual_decision"],
                           manual_reason=view["manual_reason"], review_status=view["review_status"],
                           review_revision=str(revision))
                if view["manual_decision"]:
                    row["decision"] = manual_text(view["manual_decision"], view["manual_reason"])
                elif "auto_decision" in entry["row"]:
                    # A cleared override returns to the automatic explanation, not the earlier manual text.
                    row["decision"] = view["auto_decision"]
                if row["photo_sha256"]:
                    row["photo_id"] = photo_id(row["relative_file"], row["photo_sha256"])
                rows.append(row)
            fields = list(dict.fromkeys([*self.fields, "relative_file", "decision", *REVIEW_FIELDS]))
            write_rows(destination, fields, rows)
        return destination


def apply_manual_decisions(photos, root: Path, state: dict | None = None) -> list[str]:
    """Overlay explicit choices after selection, preserving automated provenance.

    Returns warnings about decisions that no longer match a photo; nothing is transferred silently."""
    state = state if state is not None else ReviewStore(root).load()
    paths_with_choices = {d["relative_file"] for d in state["decisions"].values()}
    current = {}
    for photo in photos:
        if photo.auto_selected is None:  # --include already recorded the automatic result
            photo.auto_selected, photo.auto_decision = photo.selected, photo.decision
        relative = photo.path.relative_to(root).as_posix()
        source = getattr(photo, "source", None)
        current[relative] = getattr(photo, "digest", "") if source is None or source == photo.path else ""
        photo.review_revision = state["revision"]
        if relative not in paths_with_choices:
            continue
        try:
            digest = fingerprint(inside(root, relative))
        except OSError:
            photo.review_status = "missing"
            continue
        except ReviewError:  # changed while reading, or a path the store cannot address
            photo.review_status = "stale decision"
            continue
        decision = state["decisions"].get(photo_id(relative, digest))
        if decision:
            photo.manual_decision, photo.manual_reason = decision["choice"], decision.get("note", "")
            photo.selected = decision["choice"] == "keep"
            photo.decision = manual_text(decision["choice"], photo.manual_reason)
            photo.review_status = "applied"
        else:
            photo.review_status = "stale decision"
    moved = moved_decisions(state, current)
    for photo in photos:
        status = moved.get(photo.path.relative_to(root).as_posix())
        if status:
            photo.review_status = status
    orphaned = sorted(paths_with_choices - current.keys())
    warnings = [f"{relative}: {status}" for relative, status in sorted(moved.items())]
    if orphaned:
        warnings.append(f"{len(orphaned)} manual decision(s) refer to photos that are no longer in this collection "
                        f"(for example {orphaned[0]}); they were not applied. `fotosort decisions show` lists them.")
    return warnings
