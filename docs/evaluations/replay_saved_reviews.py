"""Replay archived shortlist memberships, without images or model calls.

Run from the checkout with PYTHONPATH=. python docs/evaluations/replay_saved_reviews.py.
"""
import argparse
import csv
import json
import tempfile
from pathlib import Path

from fotosort.evaluate import anonymous, evaluate
from fotosort.review_data import atomic_write, fingerprint, write_rows


def replay():
    source = Path(__file__).with_name("burst-alternatives-membership.csv")
    with source.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    shoots = []
    with tempfile.TemporaryDirectory(prefix="fotosort-evaluation-") as scratch:
        root = Path(scratch)
        for index, dataset in enumerate(("Botlierskop", "Buffelsdrift")):
            rows = [r for r in saved if r["dataset"] == dataset]
            ids = {r["file"]: anonymous("frame", (dataset, r["experience"], r["file"])) for r in rows}
            assert len(ids) == len(rows), "This archive needs experience-qualified feedback identities"
            favourites = [ids[name] for name in ("P9051472.JPG", "P9051008.JPG", "P9050886.JPG")] if index == 0 else []
            acceptable = [[ids["P9071549.JPG"]], [ids["P9071694.JPG"], ids["P9071709.JPG"]]] if index == 1 else []
            runs = {}
            for run, column in (("baseline", "old_shortlist"), ("candidate", "new_shortlist")):
                report = root / f"shoot-{index}-{run}.csv"
                normalized = [dict(file=ids[r["file"]], selected="0", shortlisted=str(int(r[column] == "True")))
                              for r in rows]
                write_rows(report, list(normalized[0]), normalized)
                runs[run] = dict(report=report.name, report_sha256=fingerprint(report),
                    budget=dict(candidate_limit=148 if index == 0 else 455, request_limit=0),
                    config=dict(archived_shortlist=column), models=dict(mode="archived_membership_only"),
                    usage=dict(requests=0), judgments={i: "incomplete" for i in ids.values()})
            shoots.append(dict(id=f"diagnostic-{index + 1}", split="diagnostic",
                photos=[dict(id=i, file=i) for i in ids.values()], favourites=favourites,
                acceptable=acceptable, preferences=[], runs=runs))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(version=1, shoots=shoots), indent=2) + "\n")
        return evaluate(manifest, "candidate", "baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(replay(), indent=2) + "\n"
    if args.output:
        atomic_write(args.output, result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
