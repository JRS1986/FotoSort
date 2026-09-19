# How FotoSort works

[Back to the README](../README.md) · [Command-line reference](cli-reference.md)

These are starting heuristics for an experimental preview. See the
[known limitations](../README.md#known-limitations) before relying on selections.

Each eligible JPEG or RAW input receives local measurements, which are cached in the photo folder:

| Signal | How |
|---|---|
| **Sharpness** | Detail inside the detected subject, with light noise suppression. A 4x4 tile measure supplies a fallback when no reliable subject region is available. Both measurements appear in the report. |
| **Exposure** | Fraction of clipped black and white pixels. |
| **Aesthetics** | CLIP ViT-L/14 + the [LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), trained to predict human image ratings, including AVA photographs. Runs locally on Apple GPU (MPS), CUDA or CPU. |
| **Composition** | Gentle bonuses for detected subject placement near thirds, golden-ratio lines or the centre. These are alternative compositions, not requirements. No extra image decoding is needed. |
| **Editorial preferences** | Local CLIP contrasts for expressive light, deliberate composition, subject clarity and a compelling moment. Reuses cached image embeddings; no additional images are uploaded. |
| **Subject label** | Zero-shot CLIP label from a list of ~70 subjects (lion, elephant, bird, landscape, sunset, ...). Bring your own list with `--labels`. |
| **Subject box** | YOLOv8 detector for people and animals: how big the main subject is and whether it is cut off by the frame. A detected person covering at least 2% of the frame can put the photo in the "people" group when CLIP agrees (`--person-agree 0.2`). A person covering at least half the frame is trusted without CLIP agreement; both models can misidentify animals. |
| **Subject appearance** | A second, local DINO view of the detected subject helps reveal pose changes hidden by the background. Tiny or missing boxes remain unknown. Subject features are batched and cached. |
| **Sameness** | A DINOv2 embedding and a tiny normalised thumbnail check visual similarity and framing. Within the same scene and subject bucket, strong whole-image and subject-crop agreement can also recognize reframed versions when thumbnail alignment changes. Only SHA-256-verified identical files are removed before the judge. |

Then the folder is organised:

1. **Scenes.** Within one day, consecutive shots closer than `--gap-seconds` and
   visually similar form one scene (a burst). By default each scene gets one
   label from its mean embedding, so a burst stays together. On a boat or
   anywhere the background never changes, `--label-mode image` labels every
   frame on its own instead, so bursts do not merge into one giant group.
   Within each scene and subject group, finer **moments** split on capture gaps
   over 8 seconds or whole-frame/subject DINO similarity below 0.92 against a
   running moment centroid. These are configurable starting heuristics, not
   calibrated measures of animal behaviour. Moment IDs never expand the
   shortlist or keeper budget.
2. **Groups.** Photos are grouped by *(day, subject)*, e.g. "2026-09-01 /
   elephant". All people labels share one "people" group. Each group gets a
   pick budget: `--max-per-group` (default 5) plus one per `--extra-per`
   photos, capped at three times the base.
3. **Selection by score.** Candidates are ranked within each *(day, subject)*
   group, so adding another day or species does not change an existing group's
   scores. Subject-region sharpness replaces background sharpness when a reliable
   detection exists. Aesthetics, focus, subject size and exposure determine the
   ranking, with small bounded bonuses for composition and editorial preferences.
   Close alternatives within a moment receive additional local subject-focus
   checks, as described below. Candidates are considered best-first with day-wide duplicate checks
   and per-group budgets. There is no default relative score cutoff; `--min-score`
   is an optional stricter filter. Local blur rejection compares known scenes,
   with the absolute blur floor as a fallback.
4. **Judge (optional, `--judge`).** By default, local scoring and scene/visual
   diversity build a shortlist before any images are sent (`--judge-coverage
   preselect`). Its size is at least three times the keeper budget, 12 files,
   and half of the distinct eligible files in each day/subject group
   (`--preselect-share 0.5`, rounded up), capped by the available eligible files. Scene coverage can expand it further.
   This broader default reduces how aggressively long bursts are filtered,
   at the cost of more judge input. Where the quality
   allocation has room, aesthetics, focus,
   editorial preferences, geometric composition and distinctive moments each
   nominate alternatives, even when their combined scores are weaker. Moment
   distinctiveness is weighted by quality to reduce nominations of accidental pans.
   Remaining slots reward both quality and visual novelty, with a soft preference
   for an unrepresented moment. Whole-frame and subject views both contribute:
   either can reveal a meaningful difference. Low scores reduce preference but
   never remove a candidate from this pool. Up to two existing slots explore
   unusual frames or disagreement between signals without a quality penalty.
   These nominations and exploration slots use the existing cap. Blur and exposure heuristics
   do not automatically reject the judge's candidates.
   The judge looks for subject focus, composition, expression, behaviour,
   interaction, unusual light and storytelling; an exceptional moment can
   outweigh a small technical flaw. By default each shortlisted photo is sent
   as a full composition plus one true pixel-for-pixel subject detail, when a
   subject was detected. Each initial batch can advance the group's entire
   keeper budget. Winner pools are combined into as few comparisons as fit the
   request size, so one strong batch can supply every keeper without unnecessary
   repeat rounds. Similar winners filed under different labels get a further
   visual comparison across the day.

`--judge-coverage full` explicitly sends every distinct eligible file. It uses
more API tokens and time and can catch keepers missed by local preselection.
`--preselect-share 0.5` is the default: at least half of each distinct eligible
day/subject group reaches the judge, rounded up. It improves coverage but does
not guarantee that every favourite reaches the judge or becomes a final pick.
Use `--preselect-share 0` explicitly for a smaller, cheaper shortlist sized by
keeper budget and scene coverage, accepting more risk of missed moments.
The percentage applies to initial candidate coverage; final comparisons can
send some images again, so 50% coverage does not imply 50% of full-mode cost.
In a [replay of three reviewed Botlierskop favourites](evaluations/botlierskop-preselection.md),
share 0.5 still missed all three: wider coverage is a safeguard, not a substitute
for improving which similar-burst alternatives are nominated.

The default detail crop shows the subject centre. `--judge-crops multi` adds a
subject overview and native windows at possible head positions, at extra token
cost; there is no eye detector. `--judge-crops none` omits subject details.
Cloud judging remains opt-in; a normal run uses local models only.

## Wildlife portfolio (`--mode award-roll --judge`)

 Curate one wildlife
portfolio across the entire input folder, including all dates. The target and
maximum is 12 photos; choose any target from 10 to 15 with `--roll-size`.
Normally one standout represents each species. A second must add a clearly
different exceptional moment, with a specific explanation; at most two photos
of a species can be selected. Compelling interactions between species qualify
under their main subject species. People-led photos, pets and empty scenery
are excluded by the visual judge; incidental people in a wildlife-led scene
are allowed. Local labels do not veto candidates, since animals can be
mislabelled as people.

```bash
fotosort /path/to/trip --recursive --mode award-roll --judge
fotosort /path/to/trip --recursive --mode award-roll --judge --roll-size 15 --copy
```

For large collections, this mode defaults to a **strict global shortlist of up
to 480 distinct photos**, controlled by `--award-candidates`. All input photos
receive local analysis, reusing cached features on subsequent runs. Verified
identical copies collapse across dates. Local subject groups share the candidate
budget in proportion to the square root of their sizes, so shooting a much longer
burst has diminishing influence. Within each subject group, day-relative scores
become tied percentile ranks before dates compete. The original scores are
unchanged. Selection mixes scene representatives, independent aesthetics/focus/
moment nominations, visual diversity and exploration. Scene representatives
use roughly 35% of a group's slots, reserving room for alternatives. These are
starting heuristics, not calibrated guarantees of keeper recall.

The global limit overrides standard `--preselect`, `--preselect-min` and
`--preselect-share` settings. Extra days, species or scenes cannot increase it;
under a strict budget some may be unrepresented. `--preselect-explore` still
controls exploration per local subject group. Use `--award-candidates 120` or
`240` for smaller searches. Use `--judge-coverage full` explicitly to bypass
the limit and send every distinct eligible photo. The usual `--skip-labels`
filter applies in either case, defaulting to `document or screenshot`;
`--skip-labels ""` removes that filter. The award shortlist is built from the
input collection, not from standard mode's selected highlights.

Candidates from small day/subject groups share initial batches, reducing request
overhead. Initial batches can advance up to twice the roll target. Winner pools
are compared across the collection until at most that many finalists remain;
one final visual comparison chooses the roll. Merge requests contain at most
the larger of `--judge-chunk` and four times `--roll-size`. For 480 candidates,
a target of 12 and chunks of 24, there are 20 initial requests and at most 40
successful requests total, containing up to 1,416 photo presentations because
finalists reappear in comparisons. Detail crops and retries add work. The
candidate limit is not a token or money limit; provider usage is reported after
judging. Local work still grows with collection size on the first scan.

Preview the candidate list and comparison bounds with **no judge calls**:

```bash
fotosort /path/to/trip --recursive --mode award-roll --award-plan
```

This writes `award_roll_plan.csv`, with nomination reasons and candidates marked
`Planned for award judge (not sent)`. They are not marked as judged or selected.
The plan needs no API key and does not export photos. The normal run reuses its
local feature cache; judge verdicts are not cached, so another judge run makes
fresh requests. Use `--apply-report` to export an existing result without
rejudging.

Bounded preselection can miss exceptional frames: earlier reviewed favourites
were missed even with broader local coverage. Full coverage also cannot guarantee
the globally best portfolio; batch comparisons and visual judgments remain
fallible. The `award-roll` mode asks for an ambitious editorial selection; it
does not predict competition results. Automated tests check logic and resource bounds; real-photo curation
quality needs evaluation.

The judge may return fewer than the target, including zero; ordinary frames
are not added to fill space. Failed judging stops the run instead of substituting
relative local scores. Output defaults to `AwardRoll/` and
`award_roll_report.csv`. The CSV includes `award_rank`, `judge_label`,
`award_moment`, `award_special_moment`, `award_exceptional_reason`, and the
selection reason. Species labels come from the curation verdict, so the
standard post-selection naming pass is skipped. `--include` cannot bypass the
roll's cap or species policy; manual changes can be made in the CSV and applied
without rejudging:

```bash
fotosort /path/to/trip --recursive --mode award-roll --apply-report --copy
```

## Experimental alternatives within bursts (`--burst-alternatives`)

 This is
off by default: the first [real-photo replay](evaluations/burst-alternatives.md)
found mixed results, so it has not replaced the established selection policy.
It groups continuous capture sequences using a 3-second gap, a 12-second span
and major whole-view changes. Small subject changes remain competing poses;
missing timestamps do not get invented burst assignments.

The trial reserves roughly 40% of the existing shortlist for quality/scene
representatives, 40% for burst alternatives, and the remainder for coverage
and the existing exploration allowance. Scene coverage has priority and unused
burst slots return to coverage. Burst opportunities account for how many frames
from each sequence are already shortlisted. General CLIP contrasts nominate
expressions, interactions and gestures locally; visual pose alternatives and
quality alternatives remain separate routes. There is no mouth-opening rule,
species preference, fine-tuning or change to the overall photo score.

For this trial, up to six detector boxes are retained, with a primary subject
crop and an expanded interaction crop. At most two extra CLIP views and one
extra DINO view per image are computed locally and cached. These are weak cues,
not reliable face, expression, age or animal-count detectors. The API still
receives the configured overview/detail payload. Burst packing keeps alternatives
together where they fit, without adding initial requests or repeating files.
The trial preserves the same scene-adjusted shortlist cap; final judge costs
can still vary with the returned winners. Enable it by adding
`--burst-alternatives` to a judge run. Reports include the burst ID, number of
subject boxes, semantic margins and nomination reason.

Local duplicate matching normally requires DINO and thumbnail agreement, with
subject DINO also agreeing when available. Within the same known scene and
subject bucket, `--dup-reframe-sim` (default 0.98) also allows a strong whole-image
DINO match plus subject DINO above `--dup-sim` to establish similarity despite
changed pixel alignment. Set it to 0 to disable that extra route. A reviewed
ostrich pair had whole/subject similarities of 0.991/0.983 but thumbnail
correlation of only 0.846. A more aggressive 0.94 threshold also merged
giraffe variants, but the scorer then dropped two user-preferred frames in
favour of a weaker pose. The conservative default preserves those alternatives. It applies to local selection and judge-failure fallback, never to
removing non-identical input from the visual judge. The score still chooses the
local winner; recognizing a similar sequence does not establish its best pose.

## Local focus refinement

 By default, up to
`--focus-refine-per-group 12` frames per day/subject get full-resolution focus
checks. A moment may contribute up to three plausible alternatives: overall
scores must be within one point of its leader, and scores excluding focus within
half a point. Clear winners and moments without alternatives skip this work.
Each selected subject is decoded from its actual JPEG/RAW/DxO source, oriented
and resized to a consistent 512 px long side. Tiny subjects are not enlarged.
Only successful comparisons of all alternatives change their focus preference;
missing detail never gets compared with a preview measurement. The adjustment
is bounded by twice the sharpness weight and never introduces a rejection.
Measurements and unusably small subjects are cached; transient decode errors
are retried on a later run. This remains a texture-sensitive focus heuristic,
with noise suppression, rather than face/eye detection.

Subject views add one DINO inference per usable detection, using the already
loaded model. Cache v8 preserves v6/v7 features and backfills missing subject
views, without repeating CLIP, detection or full-image DINO. Refinement decodes
one full image at a time. No new model downloads or cloud comparisons are added.
Disable subject views with `--no-subject-embeddings`, refinement with
`--focus-refine-per-group 0`, or exploration with `--preselect-explore 0`.
Enabling the separate burst trial on an older cache can rerun detection once
to collect secondary boxes; existing full-image embeddings are retained.
The new grouping and selection weights need evaluation on reviewed shoots;
deterministic tests establish their logic and bounds, not photographic accuracy.

## Composition and editorial preferences

 The geometric scores measure
how close the detected subject's box centre lies to thirds or golden-ratio
lines/intersections (approximately 0.382 and 0.618), plus an equally valid
central composition. The best matching pattern contributes at most
`--composition-weight` (default `0.2`) to the score; overlapping thirds and
golden-ratio matches never stack. Large boxes reduce the reliability of point
placement. Missing detections leave these measures unknown. This geometry does
not locate eyes, estimate horizons or fit a golden spiral.

The editorial score compares the existing CLIP image embedding with positive
and negative descriptions of light, composition, subject clarity and moment.
These cosine-similarity margins provide an uncalibrated preference; they are
not probabilities or award predictions. Above-average editorial scores within
a day/subject group receive at most `--editorial-weight` (default `0.15`).
Both bonuses are optional (`0` disables their effect and their shortlist
nominations). They never reject a frame for breaking a compositional rule.

The existing aesthetic head supplies a separate learned human-rating signal.
Its training includes AVA, whose photographs come from a photography community
and contests, as described in this [composition-aware aesthetics paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Liu_Composition-Aware_Image_Aesthetics_Assessment_WACV_2020_paper.pdf).
FotoSort has not trained a new model on verified award winners. The editorial
criteria are inspired by published judging priorities such as
[originality and narrative](https://www.nhm.ac.uk/wpy/wpy/competition/enter-the-competition)
and [composition and creativity](https://www.worldphoto.org/sony-world-photography-awards/single-image).
Composition research also treats [thirds, golden ratio, central and symmetric layouts as alternative patterns](https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB).

For a personal reference collection, `--taste-dir` can use locally available
photos you admire, including a curated collection of award-winning work. That
existing feature rewards image similarity and can also favour matching subject
matter; it does not retrain the aesthetic model.

## Reading the report

The CSV report lists every score, subject-focus measurement, label, group,
decode source, whether the judge saw the frame, and the selection decision.
The `rule_of_thirds`, `golden_ratio`, `center_composition`, `placement_reliability`
and `composition_score` columns show geometric affinities (0..1).
`editorial_light`, `editorial_composition`, `editorial_subject`, `editorial_moment`
and `editorial_score` show raw similarity margins; `composition_bonus` and
`editorial_bonus` show their actual contribution to the final score.
`moment`, `moment_novelty`, `subject_embedding` and `preselection_reason` explain
local moment coverage and nominations. `focus_rank`, `detail_focus`,
`refined_focus_adjustment` and `focus_refinement` expose extra focus checks,
including cases where detail could not be compared. The novelty and uncertainty
preferences used by preselection are heuristics, not confidence probabilities.
Relative file paths preserve your choices when a folder is moved, including
when different subfolders contain the same camera filename.
Edit the `selected` column and run `--apply-report` to override any decision.

## Output layout

 `--layout species` puts the picks into one subfolder per
subject (`Highlights/zebra/`, `Highlights/lion/`, `Highlights/people/`),
`--layout day` one per day, `--layout day-species` both. The default is one
flat folder.

## DxO PureRAW

With `--dxo`, FotoSort looks for the processed twin of each RAW in the `DxO`
folder next to it. It recognizes the original stem, date-prefixed stems, and
`-DxO`/`_DxO` suffixes, preferring DNG over TIFF and JPEG. On macOS it hands
missing files to an installed PureRAW app with
`open -a`, tells you to press Process, and waits (up to `--dxo-wait` hours)
for the outputs to appear. From then on the twin is what gets scored,
enhanced and copied, while the original RAW keeps its name in the report and
sidecars. `--dxo all` processes every RAW before analysis (the denoised files
are what gets judged); `--dxo picks` scores the originals and processes only
the selected frames, which is many times faster. `--dxo-wait 0` uses existing
outputs only and skips the macOS app hand-off. RAW+JPEG pairs also use their processed RAW twin when available,
and the judge uses the same processed source as feature extraction.

## RAW input

RAW files (including ORF, NEF, CR2, CR3, ARW, RAF, RW2, DNG, PEF and SRW) are analysed
through their embedded camera preview, so a RAW-only card works directly. A
RAW next to a JPEG with the same stem, or in its `RAW/` subfolder, is
represented by that JPEG. Generated `Highlights`, `AwardRoll`, `Enhanced`, `DxO` and
`_DELETE_ME_raw` directories are excluded from recursive input scans. Enhanced output of a RAW pick is a full-resolution JPEG demosaiced
with the camera white balance, carrying the camera model and capture time.
`--no-raw` ignores RAW files.
