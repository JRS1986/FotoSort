# Persistent review decisions

FotoSort stores explicit human choices separately from its automated report in
`.fotosort_review.json` in the photo folder. The format is versioned; no image
models or judge requests are needed to edit it.

```bash
fotosort decisions /path/to/photos set IMG_0042.JPG --choice keep --note "Best expression"
fotosort decisions /path/to/photos set IMG_0043.JPG --choice reject
fotosort decisions /path/to/photos set IMG_0042.JPG --choice clear
fotosort decisions /path/to/photos prefer IMG_0042.JPG IMG_0043.JPG --note "Eye contact"
fotosort decisions /path/to/photos alternatives "lion greeting" IMG_0042.JPG IMG_0044.JPG
fotosort decisions /path/to/photos undo
fotosort decisions /path/to/photos show
fotosort decisions /path/to/photos export
fotosort /path/to/photos --apply-report --report fotosort_reviewed.csv --copy
```

Use complete paths relative to the photo folder, including subfolders. Add
`--report award_roll_report.csv` after the folder to review a portfolio. Exports
default to `fotosort_reviewed.csv`; `export --output other.csv` changes this.

A normal analysis rerun reloads saved choices after automatic selection, including
edits made while features and judge results were computed. If review changes
again while the run is finishing, it stops before replacing the report or
exporting photos; rerun to apply the latest choices. Reports
retain `auto_selected` and `auto_decision` alongside effective `selected`,
`manual_decision`, `manual_reason`, `review_status`, and `review_revision`.
Explicit choices take precedence over automated budgets and forced includes.
`--include` changes eligibility only after automatic selection, preserving the
automatic rejection reason even for a manually included blurry frame.
For award-roll, a human addition has no automatic award rank or exceptional-moment
claim. An award plan remains a plan and does not apply human choices.

Clearing an override returns to the latest automatic recommendation. Pairwise
preferences and acceptable-alternative groups record feedback only; they do not
train a model or implicitly change picks. Undo restores the previous review edit,
including feedback, for the last 50 edits. It does not undo photo exports.

To preserve edits made directly in a CSV, explicitly import that report:

```bash
fotosort decisions /path/to/photos --report edited.csv import
```

Import records **every row** as keep or reject according to `selected`, including
zero rows, and marks those decisions with the origin `csv import` so they stay
distinguishable from choices made one photo at a time. Opening a legacy report
alone never infers negative feedback. Existing
`--apply-report` continues to honor the supplied CSV; export after a review edit
before applying it. Use the same `--recursive` setting for subfolder exports.

## Identity and concurrent edits

Identity combines the original photo's relative path and SHA-256 content digest.
Moving the whole collection preserves decisions. Same-named files in different
subfolders remain distinct, even if byte-identical. Individual file renames or
moves inside the collection do not transfer decisions: the next analysis warns
about decisions whose photo is gone, and when exactly one unannotated photo has
the same contents its `review_status` names the path the decision was recorded
for. Changed files are flagged and cannot inherit old choices; rerun analysis to
review their new content. Missing or changed picks remain visible in exports
through `review_status` rather than blocking them; `--apply-report` counts
missing picks and refuses changed ones. A stale decision can be cleared without
restoring the old file. A pairwise preference for a pair replaces an earlier
preference for the same pair.

`fotosort decisions` trusts the content hashes recorded in the report until a
photo is used: setting a decision hashes that photo, and an export hashes the
effective picks. Legacy reports without hashes are read in full on opening.
After a photo has been verified, its size and modification time decide whether
it must be read again. `--apply-report` verifies the picks it is about to copy
or move. Images are never changed by this process.
Processed sources outside the collection are not exposed by review tools; the
original is used instead, with an explicit notice.

Writes use a POSIX file lock, an expected revision, and atomic file replacement.
An outdated session must reload instead of overwriting another edit. Replacing
the report during review also requires reopening it. Back up the JSON alongside
your report if you want to retain the human annotations. Malformed or unknown
review formats are rejected rather than reset.

## Version 1 record

The document has `version`, monotonic `revision`, `decisions` keyed by photo ID,
`preferences` (winner/loser IDs with notes), named `alternatives` (acceptable photo
IDs for one moment), and a bounded `history` for undo. Each history entry stores
only the previous values of what an edit changed. Decisions retain relative
path, content hash, choice, note, timestamp, and their origin (`explicit review`
or `csv import`).
Treat this as local data: notes and relative filenames can be personal. The
evaluation harness can use anonymous IDs for shareable summaries.

Applying a report checks the content hashes of selected photos. With `--raw-cull`,
unselected report photos are checked too, before writing cull lists or moving RAWs.
A stale rejection must not relocate a different photo that now occupies that path.
