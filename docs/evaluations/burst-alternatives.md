# General burst alternatives — 2026-09-10

Implemented and tested as `--burst-alternatives`, **off by default**. The first
trial has mixed results and does not establish a general improvement. The
default remains the established selection policy with `--preselect-share 0.5`.

## Implementation

- Capture sequences use existing scene boundaries, a 3-second gap, a maximum
  12-second span and whole-view DINO similarity below 0.75. Subtle changes in
  the subject do not split the comparison sequence. Missing timestamps do not
  receive invented burst assignments.
- Inside the existing shortlist cap, target 40% quality/scene representatives,
  40% burst alternatives, and the remainder coverage/exploration. Scene
  representatives take priority; unused burst capacity returns to coverage.
  The quality allocation actually fills its reserved slots.
- Share opportunities across bursts according to the fraction of their frames
  already nominated. Alternate expression/interaction/gesture challenges with
  visual pose and quality alternatives. Similarity is considered locally for
  a pose alternative, allowing a challenge to an existing burst winner.
- Use general paired CLIP prompts and the existing frozen model. A semantic
  nomination requires at least 0.005 spread within its burst and improvement
  over an already nominated frame on that aspect. This is an uncalibrated
  heuristic, not an expression detector. No rules mention mouths, teeth,
  particular species, camera filenames or these feedback cases.
- Retain up to six subject boxes. Compare the whole view, primary subject and
  an expanded interaction crop. At most two additional CLIP views and one
  interaction DINO view are extracted per image and cached. Cache v8 reads
  v6/v7; upgrading secondary boxes can rerun detection, retaining full-image
  features. Additional analysis runs only for the opt-in preselection trial.
- Pack small burst groups together where possible. The number of initial
  requests remains `ceil(candidate_count / judge_chunk)`; no frame is repeated
  during this packing. Oversized or non-fitting bursts can still split.
- No model weights, global photo scores, keeper budgets or API crop settings
  are changed. Reports expose burst IDs, box counts, margins and nomination
  routes. With the option disabled, cached trial features cannot influence
  ordinary preselection.

## Evaluation method

Replayed **1,117 available JPGs** using saved report scores, labels, scenes and
moment metrics, with regenerated local crop features. Compared a frozen copy
of the pre-change selector with the trial, using the same source set and
50% coverage caps. Also asserted that the final implementation with the option
disabled reproduces the frozen selector's candidate membership in every group.

The Botlierskop subset contains the September 5 lion and hippo groups (294
frames). Buffelsdrift contains 823 currently available frames across the four
experiences and in-between photos. The 152 files absent from the original
975-frame report are excluded from both sides of this comparison. Consequently,
these Buffelsdrift counts are not a replay of the original 250-candidate run.

Only scratch caches/reports were written. Source sizes and modification times
were verified unchanged. No cloud judge calls, copies, enhancements or forced
includes were made. The known favourites are diagnostic cases; they are not an
unbiased validation set, and previous judge picks are only a preservation proxy.

## Results

| Dataset | Frames | Initial candidates, both policies | Previous judge picks retained, baseline → trial |
|---|---:|---:|---:|
| Botlierskop lion/hippo | 294 | 148 | 16 → 16 |
| Buffelsdrift | 823 | 455 | 87 → 82 |

**Exact Botlierskop favourites:** P9051472, P9051008 and P9050886 remain outside
the trial shortlist. Preserving variants does not establish which subtle
expression the photographer prefers, and generic local signals did not recover
these exact choices.

**Meerkat feedback:** the baseline already includes P9071549 under this 50%
replay. The trial additionally includes P9071694 and P9071709, the two accepted
alternatives for the same hug moment. Thus reference-moment coverage improves
from **1 of 8 to 2 of 8**, not from 1 to 3. The other six named moments still miss.

**Displaced previous judge picks:** `_9072338`, `_9072379`, `_9072444` from the
elephant experience; `_9072841` from the elephant walk; and `P9072544` from the
weaver-bird group. The trial replaces them with other candidates; without a
new judge/human review this does not prove the replacements are worse or better.
There is insufficient evidence to promote the trial to the default.

Groups whose complete candidate sets already fit the cap remain unchanged,
including the available night-sky group. Synthetic tests additionally exercise
portrait, bird, landscape and architecture labels, but this is a test of
category-independent logic, not proof of accuracy on those genres.

## Decision

Keep the experiment available for explicit trials and retain the 50% baseline
as the default. Do not tune constants or add species-specific prompts until
broader reviewed data can distinguish genuine gains from fitting these examples.
The implementation and its limitation are both retained for future evaluation.

Data: [group results](burst-alternatives.json) and
[candidate membership and reasons](burst-alternatives-membership.csv).
