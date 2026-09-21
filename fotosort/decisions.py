"""Inspect and edit persistent human decisions without running image models."""
import argparse
import csv
import json
from pathlib import Path

from fotosort.review_data import Collection, ReviewError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fotosort decisions", description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--report", default="fotosort_report.csv")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="Show decisions, revision, and feedback")
    set_parser = commands.add_parser("set", help="Keep/reject photos, or clear overrides")
    set_parser.add_argument("files", nargs="+")
    set_parser.add_argument("--choice", required=True, choices=["keep", "reject", "clear"])
    set_parser.add_argument("--note", default="")
    prefer = commands.add_parser("prefer", help="Record an explicit pairwise preference")
    prefer.add_argument("winner")
    prefer.add_argument("loser")
    prefer.add_argument("--note", default="")
    alternatives = commands.add_parser("alternatives", help="Record acceptable alternatives for one moment")
    alternatives.add_argument("name")
    alternatives.add_argument("files", nargs="+")
    commands.add_parser("undo", help="Undo the most recent review edit (up to 50 edits)")
    commands.add_parser("import", help="Explicitly keep/reject every row according to this CSV's selected column")
    export = commands.add_parser("export", help="Write effective choices to an apply-report compatible CSV")
    export.add_argument("--output", default="fotosort_reviewed.csv")
    args = parser.parse_args(argv)
    try:
        # Photos are hashed when a decision or export relies on them, not on every command.
        collection = Collection(args.folder, args.report, verify_sources=False)
        state = collection.store.load()
        by_name = {e["relative_file"]: e["id"] for e in collection.entries.values()}

        def identifier(name):
            if name not in by_name:
                raise ReviewError(f"Not in report: {name}; use its complete relative path")
            return by_name[name]

        if args.command == "show":
            print(json.dumps({k: v for k, v in state.items() if k != "history"}, indent=2))
            return 0
        if args.command == "export":
            print(collection.export(args.output, state["revision"]))
            return 0
        if args.command == "undo":
            state = collection.undo(state["revision"])
        elif args.command == "set":
            state = collection.change(state["revision"], {identifier(f): args.choice for f in args.files}, args.note)
        elif args.command == "prefer":
            state = collection.change(state["revision"], {}, args.note,
                                      preference=[identifier(args.winner), identifier(args.loser)])
        elif args.command == "alternatives":
            state = collection.change(state["revision"], {},
                                      alternatives=(args.name, [identifier(f) for f in args.files]))
        elif args.command == "import":
            state = collection.change(state["revision"], {
                e["id"]: "keep" if e["row"]["selected"] == "1" else "reject" for e in collection.entries.values()
            }, note="Explicit CSV import", origin="csv import")
        print(f"Saved review revision {state['revision']}")
        return 0
    except (OSError, ReviewError, UnicodeDecodeError, csv.Error) as exc:
        parser.exit(2, f"Review error: {exc}\n")
