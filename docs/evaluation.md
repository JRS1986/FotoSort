# Evaluate saved photo selections

The evaluator measures saved results against explicit photographer feedback. It
loads CSV/JSON only: no models, API calls, photo exports, or selection changes.
To acquire new results, run the normal FotoSort pipeline separately; cloud
judging still requires its explicit `--judge` option.

```bash
# After reviewing a collection, capture its explicit choices and a frozen baseline.
fotosort evaluate feedback /path/to/photos --output /path/to/private-evaluation.json \
  --split diagnostic --review-seconds 180 --manual-replacements 4
fotosort evaluate run /path/to/private-evaluation.json --output summary.json

# Add a candidate run and equal budgets to that manifest before comparing.
fotosort evaluate compare /path/to/private-evaluation.json --output comparison.json
```

`feedback` exports manual keeps as exact favourites, manual rejects as explicit
rejected picks, acceptable alternatives, and pairwise preferences. An acceptable alternative may still be rejected for the
current selection; moment coverage and exact corrections measure different things.
Untouched
frames are **not** negative labels. Stale feedback must be cleared or reviewed
again before export. The command refuses to overwrite an existing manifest.
Review duration and replacement counts are optional measurements for the
baseline run, never inferred from model runtime or pairwise feedback. Supply
candidate measurements separately when you have actually reviewed that run.

Manifests are private: they contain relative filenames and may locate local
roots or model/configuration paths. Keep them and original photos out of Git.
The output summary contains hashed shoot/run/photo identifiers, numbers, and
provenance hashes, without filenames, notes, configuration text, or model paths.
These identifiers support comparisons, but do not constitute strong anonymization
against someone who already knows the inputs. No source image is included.

## Version 1 manifest

A hand-authored manifest works without the reviewer. Input paths are resolved
relative to the manifest. Each shoot declares its entire eligible input set;
photo IDs can be stable review IDs or unique aliases within the shoot.

```json
{
  "version": 1,
  "shoots": [{
    "id": "shoot-a",
    "split": "heldout",
    "photos": [
      {"id": "frame-a", "file": "a.jpg"},
      {"id": "frame-b", "file": "b.jpg"}
    ],
    "favourites": ["frame-a"],
    "rejected": [],
    "acceptable": [["frame-a", "frame-b"]],
    "preferences": [{"winner": "frame-a", "loser": "frame-b"}],
    "runs": {
      "baseline": {
        "report": "baseline.csv",
        "config": {"preselect_share": 0.5},
        "models": {"judge": "record-the-actual-model-and-revision"},
        "budget": {"candidate_limit": 2, "request_limit": 1},
        "usage": {"requests": 1},
        "review_seconds": 45,
        "manual_replacements": 1
      },
      "candidate": {
        "report": "candidate.csv",
        "config": {"preselect_share": 0.5, "burst_alternatives": true},
        "models": {"judge": "record-the-actual-model-and-revision"},
        "budget": {"candidate_limit": 2, "request_limit": 1}
      }
    }
  }]
}
```

Each CSV requires `file` and binary `selected`. `photo_id` matches first;
otherwise `relative_file` (or a relative `file`) maps to the manifest's `file`.
Legacy absolute paths require `relative_file` or matching `photo_id`; the
evaluator does not guess their old root. Duplicate IDs/paths, extra report
photos, malformed values, and contradictory keep/reject labels are rejected.

When present, `auto_selected` is evaluated instead of effective `selected`, so
manual corrections cannot inflate automatic recall. A report with manual
choices but without automatic provenance is rejected. Without automatic
provenance, legacy `selected` is treated as the supplied baseline; the summary
flags this distinction.

Optional photo `sha256` records the original's content. Optional shoot `root`
enables missing-file/content checks using those relative paths. Without it,
`files_unverified` is reported; the harness can replay archived CSVs without
original photos. This verifies presence/bytes, not image decoding. Include
content hashes when comparing real runs; legacy reports without hashes can
only establish equality of the declared identities, not of historical bytes.

A run may include `report_sha256` to freeze its CSV. Comparisons require a
frozen baseline (automatically supplied by `feedback`). For a hand-authored
baseline, calculate it with `shasum -a 256 baseline.csv` and copy the hex digest
into `report_sha256`. A changed frozen report stops evaluation. Candidate
hashes are recorded in every result and can also be frozen explicitly.

Comparisons require all declared inputs in both reports, matching reported
content hashes, and identical budgets including `candidate_limit` and
`request_limit`. Budgets represent totals for that shoot, not per-round caps.
Optional `max_output_tokens` and `cost_limit_usd` also participate in equality.
Known candidate/request counts above the declared limits block comparison.
Unknown request usage stays unknown; equal declared budgets alone cannot prove
that an uninstrumented run respected them. Configuration/model objects remain
in the private manifest; their hashes, report hashes, input-set hash, manifest
hash, evaluator source hash, and package version are retained in the summary.

## Metrics and missing data

Every recall metric contains `hits`, `total` (known denominator), `unknown`,
and `recall`. An empty known denominator produces `null`, not zero.

| Result | Definition |
|---|---|
| `shortlist_exact` | Exact favourites with `shortlisted=1` / favourites with known exposure |
| `selection_exact` | Automatically selected exact favourites / favourites with known selection |
| `shortlist_moments` | Acceptable groups with at least one shortlisted member / groups with known coverage |
| `selection_moments` | Acceptable groups with at least one automatic pick / groups with known coverage |
| `redundant_picks` | Sum of picks beyond the first in each annotated acceptable group; a lower bound if members are unknown |
| `pairwise` | Unique pairs split into preferred-only, other-only, both, neither, and unknown outcomes |
| `corrections_required` | Explicit favourites missing from known picks and explicit rejects still selected; hashed IDs |
| `candidate_count` | Unique photos known to be shortlisted, not requests or repeated tournament presentations |
| `manual_replacements` | Explicitly supplied observed replacements for this run, or `null` |
| `review_seconds` | Explicitly measured photographer review time for this run, or `null` |

For a group, any known hit proves coverage. If no member hits and at least one
member is unknown, coverage is unknown. Missing/changed originals, missing CSV
rows, and unreadable rows do not become negative photographic judgments.
Absent `shortlisted` is unknown exposure. `shortlisted=0` means not exposed;
it does not by itself make a selected/omitted result unknown.

Optional run `judgments` maps photo IDs to `complete`, `incomplete`, `failed`,
or `unevaluated`. Incomplete/failed judgments make final selection unknown but
retain observed shortlist exposure. Unevaluated inputs make both unknown.
Legacy reports often lack reliable per-frame completion data: their selection
remains measurable, but `unknown_judgments` explicitly counts that uncertainty.
The existing `judge failed` fallback marker is recognized; missing failure
markers cannot be reconstructed. Coverage counters may overlap (for example,
a file can be both missing now and absent from the old report).

Optional measured `usage` supports `requests`, `input_tokens`, `output_tokens`,
`cost_usd`, and `runtime_seconds`. Omitted values are unknown, not estimated.
Totals identify how many shoots provided measurements. Review time and model
runtime are separate. Do not interpret acceptable alternatives as a complete
duplicate inventory; overlapping groups count excess picks per group.

Results include every shoot and separate `development`, `heldout`, and
`diagnostic` aggregates. Recall is micro-averaged by adding hits/denominators,
not averaging percentages. Keep whole shoots held out from tuning and personal
preference training. Comparisons emit recall deltas only with complete, equal
known denominators; an unknown result cannot masquerade as an improvement.
No all-splits accuracy claim is produced.

## Reproducible diagnostic replay

The repository's saved burst membership table can exercise the harness without
access to private originals:

```bash
PYTHONPATH=. python docs/evaluations/replay_saved_reviews.py --output /tmp/review-replay.json
```

This replays archived memberships, not selection models. It uses the three
recorded Botlierskop exact favourites and the two explicitly documented
Buffelsdrift reference moments (one singleton and one pair of alternatives).
The other six historical reference moments lack individual labels in these
artifacts and are not invented. Final judgments are marked incomplete because
this trial did not run a judge. The 152 historically unavailable Buffelsdrift
sources are outside both saved cohorts; no current filesystem availability is
claimed. These cases are diagnostic, not held-out photographic validation.

The archived replay produces these results:

| Diagnostic cohort | Eligible rows | Baseline → candidate shortlist | Annotated shortlist recall |
|---|---:|---:|---|
| Botlierskop | 294 | 148 → 148 | Exact favourites: 0/3 → 0/3 |
| Buffelsdrift | 823 | 455 → 455 | The two documented moments: 1/2 → 2/2 |

All 1,117 final judgments are marked incomplete in this replay; final recall
is `null`. No review duration or token usage is invented. These numbers verify
the harness against the archived memberships, not an improvement in general
photographic accuracy.
