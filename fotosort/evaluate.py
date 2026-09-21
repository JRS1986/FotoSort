"""Evaluate saved selections against explicit feedback, without model or network calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from fotosort import __version__
from fotosort.review_data import (
    STORE_NAME,
    Collection,
    ReviewError,
    atomic_write,
    fingerprint,
    inside,
    photo_id,
    read_report,
    relative_name,
)

SPLITS = {"development", "heldout", "diagnostic"}
BUDGET_FIELDS = {"candidate_limit", "request_limit", "max_output_tokens", "cost_limit_usd"}
USAGE_FIELDS = {"requests", "input_tokens", "output_tokens", "cost_usd", "runtime_seconds"}
STATUSES = {"complete", "incomplete", "failed", "unevaluated"}
METRICS = ("shortlist_exact", "selection_exact", "shortlist_moments", "selection_moments")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def anonymous(kind, value):
    return f"{kind}-{digest(value)[:16]}"


def numeric(value, name):
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ReviewError(f"{name} must be a finite nonnegative number")
    return value


def numbers(values, allowed, name):
    if not isinstance(values, dict) or set(values) - allowed:
        raise ReviewError(f"Invalid {name} fields; see the evaluation format")
    result = {key: numeric(value, key) for key, value in values.items()}
    for key in set(result) - {"cost_usd", "cost_limit_usd", "runtime_seconds"}:
        if type(result[key]) is not int:
            raise ReviewError(f"{key} must be an integer")
    return result


def local_path(base, value):
    if not isinstance(value, str) or not value:
        raise ReviewError("Expected an input path")
    return (base / Path(value).expanduser()).resolve()


def load_manifest(path: Path):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (type(manifest["version"]) is not int or manifest["version"] != 1
                or not isinstance(manifest["shoots"], list) or not manifest["shoots"]):
            raise ValueError("version or shoots")
        names = set()
        for shoot in manifest["shoots"]:
            if not isinstance(shoot["id"], str) or not shoot["id"] or shoot["id"] in names:
                raise ValueError("duplicate or empty shoot id")
            names.add(shoot["id"])
            if shoot["split"] not in SPLITS:
                raise ValueError("split must be development, heldout, or diagnostic")
            photos = shoot["photos"]
            if not isinstance(photos, list) or not photos:
                raise ValueError("photos must declare the eligible input set")
            ids, paths = set(), set()
            for photo in photos:
                identifier, name = photo["id"], relative_name(photo["file"])
                if not isinstance(identifier, str) or not identifier or identifier in ids or name in paths:
                    raise ValueError("ambiguous photo identity")
                if photo.get("sha256") and (len(photo["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in photo["sha256"])):
                    raise ValueError("invalid photo digest")
                ids.add(identifier)
                paths.add(name)
            for key in ("favourites", "rejected"):
                if not isinstance(shoot.get(key, []), list) or set(shoot.get(key, [])) - ids:
                    raise ValueError("unknown feedback photo")
            if set(shoot.get("favourites", [])) & set(shoot.get("rejected", [])):
                raise ValueError("a favourite cannot also be rejected")
            for group in shoot.get("acceptable", []):
                if not isinstance(group, list) or not group or len(set(group)) != len(group) or set(group) - ids:
                    raise ValueError("invalid acceptable moment")
            for pair in shoot.get("preferences", []):
                if pair["winner"] not in ids or pair["loser"] not in ids or pair["winner"] == pair["loser"]:
                    raise ValueError("invalid pairwise preference")
            if not isinstance(shoot["runs"], dict) or not shoot["runs"]:
                raise ValueError("runs must contain saved reports")
            for run in shoot["runs"].values():
                if not isinstance(run["report"], str) or not run["report"]:
                    raise ValueError("invalid report path")
                numbers(run.get("budget", {}), BUDGET_FIELDS, "budget")
                numbers(run.get("usage", {}), USAGE_FIELDS, "usage")
                for key in ("review_seconds", "manual_replacements"):
                    if run.get(key) is not None:
                        numeric(run[key], key)
                if run.get("manual_replacements") is not None and type(run["manual_replacements"]) is not int:
                    raise ValueError("manual_replacements must be an integer")
                if not isinstance(run.get("judgments", {}), dict):
                    raise ValueError("invalid judgments")
                if set(run.get("judgments", {})) - ids or any(
                        status not in STATUSES for status in run.get("judgments", {}).values()):
                    raise ValueError("invalid judgment status or photo id")
        return manifest
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError(f"Invalid evaluation manifest: {exc}") from exc


def recall_counts(hits, total, unknown):
    """Recall is only a number when every group is known.

    A group with an unknown member can become a hit but never a miss, so hits / total over the known
    groups alone is biased upward. With unknown groups the result is null and the bounds say what is
    established: every unknown group missed (lower) or every unknown group hit (upper)."""
    groups = total + unknown
    return dict(hits=hits, total=total, unknown=unknown,
                recall=hits / total if total and not unknown else None,
                recall_lower_bound=hits / groups if groups else None,
                recall_upper_bound=(hits + unknown) / groups if groups else None)


def metric(groups, values):
    """Unknown members make a missed group unknown; any known hit establishes a hit."""
    hits = total = unknown = 0
    for group in groups:
        verdicts = [values.get(identifier) for identifier in group]
        if True in verdicts:
            hits += 1
            total += 1
        elif None in verdicts:
            unknown += 1
        else:
            total += 1
    return recall_counts(hits, total, unknown)


def match_absolute(filename: str, by_file: dict) -> str | None:
    """A legacy absolute path identifies a declared photo only when exactly one declared file ends it."""
    parts = Path(filename).parts
    found = [identifier for name, identifier in by_file.items()
             if parts[-len(Path(name).parts):] == Path(name).parts]
    if len(found) > 1:
        raise ReviewError("Legacy absolute report path matches several declared photos; add relative_file")
    return found[0] if found else None


def source_state(shoot, base, cache: dict) -> dict:
    """Missing/changed state of every declared source, read once per shoot however many runs use it."""
    if shoot["id"] not in cache:
        states = {}
        root = local_path(base, shoot["root"]) if shoot.get("root") else None
        for photo in shoot["photos"]:
            if root is None:
                states[photo["id"]] = "unverified"
            else:
                source = inside(root, photo["file"])
                if not source.is_file():
                    states[photo["id"]] = "missing"
                elif photo.get("sha256") and fingerprint(source) != photo["sha256"]:
                    states[photo["id"]] = "changed"
                else:
                    states[photo["id"]] = "available"
        cache[shoot["id"]] = states
    return cache[shoot["id"]]


def forced_by_hand(row) -> bool:
    """--include is a human choice. Reports that recorded it as automatic cannot say what automation chose."""
    if row.get("judge_reason", "").lower() != "forced by --include":
        return False
    return "auto_selected" not in row or row.get("auto_decision") == "Forced by --include"


def evaluate_shoot(shoot, run_name, base, sources: dict | None = None):
    if run_name not in shoot["runs"]:
        raise ReviewError("Requested run is absent from a shoot")
    run = shoot["runs"][run_name]
    report = local_path(base, run["report"])
    report_hash = fingerprint(report)
    if run.get("report_sha256") and run["report_sha256"] != report_hash:
        raise ReviewError("Frozen report changed; restore the declared report before evaluating")
    _, rows = read_report(report)
    if fingerprint(report) != report_hash:
        raise ReviewError("Report changed during evaluation")
    photos = {p["id"]: p for p in shoot["photos"]}
    by_file = {relative_name(p["file"]): p["id"] for p in shoot["photos"]}
    matched = {}
    for row in rows:
        identifier = row.get("photo_id")
        if identifier not in photos:
            filename = row.get("relative_file") or row["file"]
            identifier = (match_absolute(filename, by_file) if Path(filename).is_absolute()
                          else by_file.get(relative_name(filename)))
        if identifier is None or identifier not in photos:
            raise ReviewError("Report contains a photo outside the declared eligible input set")
        if identifier in matched:
            raise ReviewError("Report contains duplicate photo identities")
        matched[identifier] = row
    coverage = dict(eligible=len(photos), report_rows=len(matched), missing_files=0, changed_files=0,
                    files_unverified=0, unevaluated=0, incomplete_judgments=0, failed_judgments=0,
                    unknown_judgments=0, shortlist_unknown=0, not_shortlisted=0, human_forced=0)
    shortlisted, selected = {}, {}
    states = source_state(shoot, base, sources if sources is not None else {})
    # Analysis without a judge writes shortlisted=0 on every row. That records no exposure at all,
    # not a judge that saw nothing, so exposure is unknown unless the report shortlisted something.
    exposure_recorded = any(row.get("shortlisted") == "1" for row in rows)
    for identifier, photo in photos.items():
        row = matched.get(identifier)
        state = states[identifier]
        unavailable = state in {"missing", "changed"}
        counter = dict(missing="missing_files", changed="changed_files", unverified="files_unverified").get(state)
        if counter:
            coverage[counter] += 1
        if row and row.get("manual_decision") and "auto_selected" not in row:
            raise ReviewError("Manual report needs auto_selected to avoid evaluating human corrections as automation")
        if row and photo.get("sha256") and row.get("photo_sha256") and row["photo_sha256"] != photo["sha256"]:
            raise ReviewError("Report photo content differs from the declared eligible input")
        status = run.get("judgments", {}).get(identifier)
        if row is None or (row and row.get("reject") == "unreadable"):
            status = "unevaluated"
        elif status is None and "judge failed" in row.get("judge_reason", "").lower():
            status = "failed"
        counter = {None: "unknown_judgments", "unevaluated": "unevaluated", "incomplete": "incomplete_judgments",
                   "failed": "failed_judgments"}.get(status)
        if counter:
            coverage[counter] += 1
        raw_shortlist = row.get("shortlisted", "") if row else ""
        if raw_shortlist not in {"", "0", "1"}:
            raise ReviewError("shortlisted must be 0, 1, or empty")
        shortlisted[identifier] = ((raw_shortlist == "1")
                                   if raw_shortlist and exposure_recorded and not unavailable else None)
        selected[identifier] = (row.get("auto_selected", row["selected"]) == "1") if row else None
        if row and forced_by_hand(row):
            coverage["human_forced"] += 1
            selected[identifier] = None
        if unavailable or status == "unevaluated":
            shortlisted[identifier] = selected[identifier] = None
        elif status in {"incomplete", "failed"}:
            selected[identifier] = None
        if shortlisted[identifier] is None:
            coverage["shortlist_unknown"] += 1
        elif not shortlisted[identifier]:
            coverage["not_shortlisted"] += 1
    favourites = [[p] for p in sorted(set(shoot.get("favourites", [])))]
    moments = [list(group) for group in dict.fromkeys(tuple(sorted(g)) for g in shoot.get("acceptable", []))]
    metrics = {f"{stage}_{target}": metric(groups, values)
               for stage, values in (("shortlist", shortlisted), ("selection", selected))
               for target, groups in (("exact", favourites), ("moments", moments))}
    pairs = dict(preferred_only=0, other_only=0, both=0, neither=0, unknown=0)
    verdicts = {}  # the latest preference for a pair replaces an earlier, possibly reversed one
    for preference in shoot.get("preferences", []):
        verdicts[frozenset((preference["winner"], preference["loser"]))] = (preference["winner"], preference["loser"])
    for winner, loser in sorted(verdicts.values()):
        a, b = selected[winner], selected[loser]
        outcome = "unknown" if a is None or b is None else "both" if a and b else (
            "preferred_only" if a else "other_only" if b else "neither")
        pairs[outcome] += 1
    excess = sum(max(0, sum(selected[i] is True for i in group) - 1) for group in moments)
    required = dict(add=[anonymous("photo", i) for i in sorted(set(shoot.get("favourites", [])))
                         if selected[i] is False],
                    remove=[anonymous("photo", i) for i in sorted(set(shoot.get("rejected", [])))
                            if selected[i] is True])
    budget, usage = run.get("budget", {}), run.get("usage", {})
    candidate_count = sum(v is True for v in shortlisted.values())
    # candidate_count is a lower bound while exposure is unknown: it can already prove a breach,
    # but it can only prove compliance once every exposure is known.
    within = None
    if "candidate_limit" in budget:
        if candidate_count > budget["candidate_limit"]:
            within = False
        elif not coverage["shortlist_unknown"]:
            within = True
    limits = dict(candidates=within,
                  requests=(usage["requests"] <= budget["request_limit"])
                  if "requests" in usage and "request_limit" in budget else None)
    return dict(shoot=anonymous("shoot", shoot["id"]), split=shoot["split"],
                metrics=metrics, coverage=coverage, pairwise=pairs,
                redundant_picks=dict(excess_in_annotated_moments=excess,
                    fully_evaluated_moments=sum(all(selected[i] is not None for i in g) for g in moments)),
                corrections_required=required, manual_replacements=run.get("manual_replacements"),
                review_seconds=run.get("review_seconds"),
                candidate_count=candidate_count, budget_compliance=limits,
                selected_count=sum(v is True for v in selected.values()), usage=run.get("usage", {}),
                provenance=dict(report_sha256=report_hash, config_sha256=digest(run.get("config", {})),
                    models_sha256=digest(run.get("models", {})), budget=run.get("budget", {}),
                    input_set_sha256=digest(shoot["photos"]),
                    automatic_selection=bool(rows) and all("auto_selected" in r for r in rows)),
                _cohort=set(matched), _identities={i: r.get("photo_sha256", "") for i, r in matched.items()})


def aggregate(results):
    splits = {}
    for split in sorted({r["split"] for r in results}):
        members = [r for r in results if r["split"] == split]
        metrics = {}
        for name in METRICS:
            metrics[name] = recall_counts(*(sum(r["metrics"][name][k] for r in members)
                                            for k in ("hits", "total", "unknown")))
        splits[split] = dict(shoot_count=len(members), metrics=metrics,
            coverage={k: sum(r["coverage"][k] for r in members) for k in members[0]["coverage"]},
            candidate_count=sum(r["candidate_count"] for r in members),
            selected_count=sum(r["selected_count"] for r in members),
            pairwise={k: sum(r["pairwise"][k] for r in members) for k in members[0]["pairwise"]},
            usage={k: dict(total=sum(r["usage"][k] for r in members if k in r["usage"]),
                           measured_shoots=sum(k in r["usage"] for r in members))
                   for k in sorted(USAGE_FIELDS) if any(k in r["usage"] for r in members)},
            manual_replacements=sum(r["manual_replacements"] for r in members if r["manual_replacements"] is not None)
                if any(r["manual_replacements"] is not None for r in members) else None,
            replacement_measured_shoots=sum(r["manual_replacements"] is not None for r in members),
            excess_in_annotated_moments=sum(r["redundant_picks"]["excess_in_annotated_moments"] for r in members),
            review_seconds=sum(r["review_seconds"] for r in members if r["review_seconds"] is not None)
                if any(r["review_seconds"] is not None for r in members) else None,
            timed_shoots=sum(r["review_seconds"] is not None for r in members))
    return splits


def evaluate(path: Path, run_name: str, baseline: str | None = None):
    path = path.resolve()
    manifest_hash = fingerprint(path)
    manifest = load_manifest(path)
    results, baseline_results, sources, verified = [], [], {}, []
    for shoot in manifest["shoots"]:
        result = evaluate_shoot(shoot, run_name, path.parent, sources)
        if baseline:
            baseline_run, candidate_run = shoot["runs"].get(baseline, {}), shoot["runs"][run_name]
            if not baseline_run.get("report_sha256"):
                raise ReviewError("Comparison requires a frozen baseline report_sha256")
            budget = baseline_run.get("budget", {})
            if not {"candidate_limit", "request_limit"} <= set(budget) or budget != candidate_run.get("budget", {}):
                raise ReviewError("Comparison requires identical declared candidate_limit and request_limit budgets")
            previous = evaluate_shoot(shoot, baseline, path.parent, sources)
            if any(v is False for r in (previous, result) for v in r["budget_compliance"].values()):
                raise ReviewError("Comparison run exceeds its declared candidate or request budget")
            expected = {p["id"] for p in shoot["photos"]}
            if previous.pop("_cohort") != expected or result["_cohort"] != expected:
                raise ReviewError("Comparison requires the same complete eligible input set in both reports")
            # A legacy report records no hashes: that is an unverified identity, not a different photo.
            before = previous.pop("_identities")
            if any(before[i] and after and before[i] != after for i, after in result["_identities"].items()):
                raise ReviewError("Comparison reports disagree on photo content identities")
            verified.append(all(before[i] and after for i, after in result["_identities"].items()))
            baseline_results.append(previous)
        result.pop("_cohort")
        result.pop("_identities")
        results.append(result)
    if fingerprint(path) != manifest_hash:
        raise ReviewError("Manifest changed during evaluation")
    output = dict(version=1, evaluator_version=__version__, evaluator_sha256=fingerprint(Path(__file__)),
                  manifest_sha256=manifest_hash,
                  mode="saved_reports", run=anonymous("run", run_name), shoots=results, splits=aggregate(results))
    if baseline:
        output["baseline"] = dict(run=anonymous("run", baseline), shoots=baseline_results,
                                  splits=aggregate(baseline_results))
        output["comparison"] = []
        for before, after, identities in zip(baseline_results, results, verified, strict=True):
            deltas = {}
            for name in METRICS:
                a, b = before["metrics"][name], after["metrics"][name]
                # Coverage changes must not masquerade as a recall improvement.
                comparable = a["unknown"] == b["unknown"] == 0 and a["total"] == b["total"] and a["total"] > 0
                deltas[name] = b["recall"] - a["recall"] if comparable else None
            output["comparison"].append(dict(shoot=after["shoot"], split=after["split"], recall_delta=deltas,
                                             content_identities_verified=identities))
    return output


def export_feedback(folder, report, output, split, review_seconds=None, manual_replacements=None,
                    include_imported=False):
    refuse_review_record(output)
    collection = Collection(folder, report)
    with collection.store.locked():
        state = collection.store.load()
        # `decisions import` copies a report's own selected column. Those rows are not favourites or
        # rejects the photographer chose one by one, so they are feedback only on request.
        decisions = {i: d for i, d in state["decisions"].items()
                     if include_imported or d.get("origin") != "csv import"}
        referenced = set(decisions) | {i for p in state["preferences"] for i in (p["winner"], p["loser"])}
        for identifier in referenced:
            collection.verify(identifier)
        for group in state["alternatives"].values():
            for identifier in group:
                collection.verify(identifier)
        # All photos remain eligible, including frames never exposed to the judge.
        photos = []
        for entry in collection.entries.values():
            expected = entry["row"].get("photo_sha256") or entry["sha256"]
            identifier = photo_id(entry["relative_file"], expected or "missing")
            photos.append(dict(id=identifier, file=entry["relative_file"], sha256=expected))
        result = dict(version=1, shoots=[dict(id=anonymous("shoot", [p["id"] for p in photos]), split=split,
            root=os.path.relpath(collection.root, output.parent.resolve()), photos=photos,
            favourites=[i for i, d in decisions.items() if d["choice"] == "keep"],
            rejected=[i for i, d in decisions.items() if d["choice"] == "reject"],
            acceptable=list(state["alternatives"].values()),
            preferences=[dict(winner=p["winner"], loser=p["loser"]) for p in state["preferences"]],
            runs=dict(baseline=dict(review_seconds=review_seconds, manual_replacements=manual_replacements,
                report=os.path.relpath(collection.report, output.parent.resolve()),
                report_sha256=collection.report_digest, config={}, models={}, budget={}, usage={})))])
        collection.ensure_current()
        atomic_write(output, json.dumps(result, indent=2) + "\n")
    return result, len(state["decisions"]) - len(decisions)


def refuse_review_record(output: Path) -> None:
    # Reserve the name even before the store exists, including aliases and case-insensitive volumes.
    if output.resolve().name.casefold() == STORE_NAME.casefold():
        raise ReviewError("Evaluation output must not overwrite or create a review record")


def refuse_overwrite(output: Path, manifest_path: Path, manifest: dict) -> None:
    """Checked before evaluating. samefile also catches another spelling of the same file, such as
    different letter case on a case-insensitive volume, a symlink, or a hard link."""
    if output.suffix.lower() != ".json":
        raise ReviewError("Summary output must be a JSON file")
    refuse_review_record(output)
    base = manifest_path.resolve().parent
    inputs = {manifest_path.resolve()}
    inputs.update(local_path(base, r["report"]) for s in manifest["shoots"] for r in s["runs"].values())
    for shoot in manifest["shoots"]:
        if shoot.get("root"):
            root = local_path(base, shoot["root"])
            inputs.update(inside(root, p["file"]).resolve() for p in shoot["photos"])
            inputs.update({root / STORE_NAME, root / ".fotosort_review.lock"})
    if (output.resolve() in inputs
            or (output.exists() and any(path.exists() and output.samefile(path) for path in inputs))):
        raise ReviewError("Summary output must not overwrite its manifest, reports, photos, or review record")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fotosort evaluate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    feedback = sub.add_parser("feedback", help="Create a private manifest from explicit saved feedback")
    feedback.add_argument("folder", type=Path)
    feedback.add_argument("--report", default="fotosort_report.csv")
    feedback.add_argument("--output", type=Path, required=True)
    feedback.add_argument("--split", choices=sorted(SPLITS), default="diagnostic")
    feedback.add_argument("--review-seconds", type=float)
    feedback.add_argument("--manual-replacements", type=int)
    feedback.add_argument("--include-imported", action="store_true",
                          help="Also export decisions recorded by `fotosort decisions import` (default: skip them)")
    for command in ("run", "compare"):
        help_text = "Evaluate saved reports" if command == "run" else "Compare with a frozen baseline"
        child = sub.add_parser(command, help=help_text)
        child.add_argument("manifest", type=Path)
        child.add_argument("--run", default="baseline" if command == "run" else "candidate")
        child.add_argument("--output", type=Path, help="Write a shareable JSON summary (default: stdout)")
        if command == "compare":
            child.add_argument("--baseline", default="baseline")
    args = parser.parse_args(argv)
    try:
        if args.command == "feedback":
            for key in ("review_seconds", "manual_replacements"):
                if getattr(args, key) is not None:
                    numeric(getattr(args, key), key)
            if args.output.suffix.lower() != ".json":
                raise ReviewError("Feedback output must be a JSON file")
            if args.output.exists():
                raise ReviewError("Feedback output already exists; choose a new manifest filename")
            _, skipped = export_feedback(args.folder, args.report, args.output, args.split,
                                         args.review_seconds, args.manual_replacements, args.include_imported)
            print("Created private evaluation manifest. Fill in run configuration, models, "
                  "and budgets before comparison.")
            if skipped:
                print(f"Skipped {skipped} decisions recorded by `decisions import`; "
                      "--include-imported exports them as favourites and rejects.")
        else:
            if args.output:
                refuse_overwrite(args.output, args.manifest, load_manifest(args.manifest))
            result = evaluate(args.manifest, args.run, getattr(args, "baseline", None))
            text = json.dumps(result, indent=2, allow_nan=False) + "\n"
            if args.output:
                atomic_write(args.output, text)
            else:
                print(text, end="")
        return 0
    except (OSError, ValueError, csv.Error) as exc:  # ValueError covers ReviewError and undecodable CSVs
        parser.exit(2, f"Evaluation error: {exc}\n")
