"""Evaluate saved selections against explicit feedback, without model or network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from fotosort import __version__
from fotosort.review_data import (
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
    return dict(hits=hits, total=total, unknown=unknown, recall=hits / total if total else None)


def evaluate_shoot(shoot, run_name, base):
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
            if Path(filename).is_absolute():
                raise ReviewError("Legacy absolute reports need relative_file or matching photo_id columns")
            identifier = by_file.get(relative_name(filename))
        if identifier is None or identifier not in photos:
            raise ReviewError("Report contains a photo outside the declared eligible input set")
        if identifier in matched:
            raise ReviewError("Report contains duplicate photo identities")
        matched[identifier] = row
    coverage = dict(eligible=len(photos), report_rows=len(matched), missing_files=0, changed_files=0,
                    files_unverified=0, unevaluated=0, incomplete_judgments=0, failed_judgments=0,
                    unknown_judgments=0, shortlist_unknown=0, not_shortlisted=0)
    shortlisted, selected = {}, {}
    root = local_path(base, shoot["root"]) if shoot.get("root") else None
    for identifier, photo in photos.items():
        row = matched.get(identifier)
        unavailable = False
        if root:
            source = inside(root, photo["file"])
            if not source.is_file():
                coverage["missing_files"] += 1
                unavailable = True
            elif photo.get("sha256") and fingerprint(source) != photo["sha256"]:
                coverage["changed_files"] += 1
                unavailable = True
        else:
            coverage["files_unverified"] += 1
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
        shortlisted[identifier] = (raw_shortlist == "1") if raw_shortlist and not unavailable else None
        selected[identifier] = (row.get("auto_selected", row["selected"]) == "1") if row else None
        if unavailable or status == "unevaluated":
            shortlisted[identifier] = selected[identifier] = None
        elif status in {"incomplete", "failed"}:
            selected[identifier] = None
        if shortlisted[identifier] is None:
            coverage["shortlist_unknown"] += 1
        elif not shortlisted[identifier]:
            coverage["not_shortlisted"] += 1
    favourites = [[p] for p in sorted(set(shoot.get("favourites", [])))]
    moments = shoot.get("acceptable", [])
    metrics = {f"{stage}_{target}": metric(groups, values)
               for stage, values in (("shortlist", shortlisted), ("selection", selected))
               for target, groups in (("exact", favourites), ("moments", moments))}
    pairs = dict(preferred_only=0, other_only=0, both=0, neither=0, unknown=0)
    for winner, loser in sorted({(p["winner"], p["loser"]) for p in shoot.get("preferences", [])}):
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
    limits = dict(candidates=(candidate_count <= budget["candidate_limit"])
                  if "candidate_limit" in budget and not coverage["shortlist_unknown"] else None,
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
                    automatic_selection=all("auto_selected" in r for r in rows)),
                _cohort=set(matched), _identities={i: r.get("photo_sha256", "") for i, r in matched.items()})


def aggregate(results):
    splits = {}
    for split in sorted({r["split"] for r in results}):
        members = [r for r in results if r["split"] == split]
        metrics = {}
        for name in METRICS:
            counts = {k: sum(r["metrics"][name][k] for r in members) for k in ("hits", "total", "unknown")}
            counts["recall"] = counts["hits"] / counts["total"] if counts["total"] else None
            metrics[name] = counts
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
    results, baseline_results = [], []
    for shoot in manifest["shoots"]:
        result = evaluate_shoot(shoot, run_name, path.parent)
        if baseline:
            baseline_run, candidate_run = shoot["runs"].get(baseline, {}), shoot["runs"][run_name]
            if not baseline_run.get("report_sha256"):
                raise ReviewError("Comparison requires a frozen baseline report_sha256")
            budget = baseline_run.get("budget", {})
            if not {"candidate_limit", "request_limit"} <= set(budget) or budget != candidate_run.get("budget", {}):
                raise ReviewError("Comparison requires identical declared candidate_limit and request_limit budgets")
            previous = evaluate_shoot(shoot, baseline, path.parent)
            if any(v is False for r in (previous, result) for v in r["budget_compliance"].values()):
                raise ReviewError("Comparison run exceeds its declared candidate or request budget")
            expected = {p["id"] for p in shoot["photos"]}
            if previous.pop("_cohort") != expected or result["_cohort"] != expected:
                raise ReviewError("Comparison requires the same complete eligible input set in both reports")
            if previous.pop("_identities") != result["_identities"]:
                raise ReviewError("Comparison reports disagree on photo content identities")
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
        for before, after in zip(baseline_results, results, strict=True):
            deltas = {}
            for name in METRICS:
                a, b = before["metrics"][name], after["metrics"][name]
                # Coverage changes must not masquerade as a recall improvement.
                comparable = a["unknown"] == b["unknown"] == 0 and a["total"] == b["total"] and a["total"] > 0
                deltas[name] = b["recall"] - a["recall"] if comparable else None
            output["comparison"].append(dict(shoot=after["shoot"], split=after["split"], recall_delta=deltas))
    return output


def export_feedback(folder, report, output, split, review_seconds=None, manual_replacements=None):
    collection = Collection(folder, report)
    with collection.store.locked():
        state = collection.store.load()
        referenced = set(state["decisions"]) | {i for p in state["preferences"] for i in (p["winner"], p["loser"])}
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
            favourites=[i for i, d in state["decisions"].items() if d["choice"] == "keep"],
            rejected=[i for i, d in state["decisions"].items() if d["choice"] == "reject"],
            acceptable=list(state["alternatives"].values()),
            preferences=[dict(winner=p["winner"], loser=p["loser"]) for p in state["preferences"]],
            runs=dict(baseline=dict(review_seconds=review_seconds, manual_replacements=manual_replacements,
                report=os.path.relpath(collection.report, output.parent.resolve()),
                report_sha256=collection.report_digest, config={}, models={}, budget={}, usage={})))])
        collection.ensure_current()
        atomic_write(output, json.dumps(result, indent=2) + "\n")
    return result


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
            export_feedback(args.folder, args.report, args.output, args.split,
                            args.review_seconds, args.manual_replacements)
            print("Created private evaluation manifest. Fill in run configuration, models, "
                  "and budgets before comparison.")
        else:
            result = evaluate(args.manifest, args.run, getattr(args, "baseline", None))
            text = json.dumps(result, indent=2, allow_nan=False) + "\n"
            if args.output:
                manifest = load_manifest(args.manifest)
                inputs = {args.manifest.resolve()}
                if args.output.suffix.lower() != ".json":
                    raise ReviewError("Summary output must be a JSON file")
                inputs.update(local_path(args.manifest.resolve().parent, r["report"])
                              for s in manifest["shoots"] for r in s["runs"].values())
                for shoot in manifest["shoots"]:
                    if shoot.get("root"):
                        root = local_path(args.manifest.resolve().parent, shoot["root"])
                        inputs.update(inside(root, p["file"]).resolve() for p in shoot["photos"])
                if args.output.resolve() in inputs:
                    raise ReviewError("Summary output must not overwrite its manifest or source reports")
                atomic_write(args.output, text)
            else:
                print(text, end="")
        return 0
    except (OSError, ReviewError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
