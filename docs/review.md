# Persistent review decisions

## Visual reviewer

```bash
fotosort review /path/to/photos
fotosort review /path/to/photos --report award_roll_report.csv --no-browser
```

The command opens a local browser session for an existing report. It performs
no analysis or judging. Filter by day, subject, scene, moment, or judge exposure;
all report frames are available, including excluded alternatives. Pin a reference
and browse nearby frames in capture order. Fit and 100% views synchronize zoom
and pan; drag either native-resolution image, or focus it and use arrow keys.
Full RAW decoding is deferred until you request native detail; at 100% a JPEG
source is shown as the file itself, without re-encoding.

Keep (`K`), reject (`X`), clear (`C`), pin (`P`), and undo (`Z`) have keyboard
shortcuts. Arrow keys browse frames outside a zoomed image. **Use candidate
instead** atomically rejects the reference, keeps the candidate, and records a
pairwise preference. **Prefer candidate** and **Both acceptable** record feedback
without changing picks. Add a review note before taking an action. The note
field always shows the saved note of the candidate frame, including when a
filter or a saved choice moves the selection to another frame.

**Export reviewed CSV** writes `fotosort_reviewed.csv` in the collection. It can
then be applied using the command below. An export never replaces the report
being reviewed: reviewing `fotosort_reviewed.csv` itself exports to
`fotosort_reviewed_2.csv`. A stale session displays a conflict; use **Reload**
to refresh before editing again. A failed image verification updates the source status, so **Reload** and the
**Needs attention** filter expose changed files. Changed/missing sources remain visible, and
selected unavailable sources are exported with their `review_status` instead of
being silently dropped. Stop the server with Ctrl+C.

The server binds only to `127.0.0.1` on an automatically selected port. `--port`
chooses a port; `--no-browser` prints the URL without launching a browser. The
printed URL carries a random session token: keep it local. Requests check the
host, origin, and token; image paths are restricted to report entries inside
the collection. The UI has no remote scripts, fonts, or image uploads.
Thumbnails are loaded lazily, with 48 frames per page and a 32 MiB image cache.
Opening the reviewer reads no photo when the report records content hashes: a
photo is hashed when it is first shown or decided, and read again only after its
size or modification time changed. Connections are accepted concurrently, so an
idle or slow connection cannot stall the page, while requests are still handled
one at a time: peak image memory depends on one source's resolution, not the
number of simultaneous browser requests. Error messages sent to the browser
never contain absolute paths.

## Command-line decisions

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
