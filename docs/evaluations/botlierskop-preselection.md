# Botlierskop preselection review — 2026-09-10

The user identified three favourites from September 5 that the judge never
saw. Restoring a 50% coverage floor increases the candidate pool but, with the
current nomination algorithm, does **not** recover these three frames.

## Method

- Dataset: `20260904-06_Botlierskop_GameDrives-HorseBackSafari`.
- Baseline: `fotosort_report_codex_run.csv` (1,531 images).
- Replayed only September 5's lion and hippopotamus groups: 294 images.
- Retained report scores, labels, scenes, moments and independent nomination
  metrics. These report values are rounded.
- Regenerated visual features locally into scratch storage because the photo
  folder's current v5 cache lacks subject crops. The source images and photo
  folder cache were not modified.
- The share-0 replay matched all 48 originally shortlisted filenames exactly.
- No cloud judge calls, forced includes or filename-based selection rules.

## Results

| Share | Hippo candidates / 197 | Lion candidates / 97 | Named favourites reaching judge input |
|---|---:|---:|---|
| 0 | 27 | 21 | None |
| 0.25 | 50 | 25 | None |
| 0.5 | 99 | 49 | None |
| 0.75 | 148 | 73 | P9051008, P9050886 |
| 1 | 197 | 97 | All three |

These counts measure initial candidate exposure, not final judge choices or
billable token totals. Share-based counts round up. The examples were supplied
as known failures, so they are diagnostic cases, not an unbiased quality sample.

| Favourite | User subject | Combined-score rank within group | Share 0.5 |
|---|---|---:|---|
| P9051472.JPG | Hippopotamus | 167 / 197 | Excluded |
| P9051008.JPG | Lion | 26 / 97 | Excluded |
| P9050886.JPG | Lion | 27 / 97 | Excluded |

The original judge selected nearby variants P9051009 and P9050882, but never
compared them with the user's preferred alternatives. Their full-image DINO
similarities to the favourites were approximately 0.966 and 0.953. The current
coverage utility discounts candidates similar to previously nominated frames;
this can favour another view over a better expression in an existing view.

## Decision and next proposal

Restore `--preselect-share 0.5` as the default. Keep share 0 as an explicit
cost-saving choice. Enforce the percentage floor with ceiling, not rounding to
nearest, so odd-sized groups also meet the promised minimum.

The next selection change should protect alternatives within short bursts and
nominate action/expression independently of overall photographic polish and
background diversity. Multiple promising poses from a burst should compete
for slots within the existing shortlist cap. Include adult/young interactions
from the separate meerkat feedback in that evaluation. Increasing the share
alone is not a demonstrated fix for these cases.

This change restores the coverage setting only; it does not implement the
proposed action/expression nominations or claim improved final judge recall.

Machine-readable results: [botlierskop-preselection.json](botlierskop-preselection.json).

Follow-up: the proposal was implemented as an opt-in experiment and tested
across two shoots. See [burst alternatives](burst-alternatives.md) for results
and why it remains off by default.
