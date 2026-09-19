# FotoSort

[![CI](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml/badge.svg)](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Picks the best, most varied photos out of a folder with thousands of JPEGs or
RAW files and puts them into a subfolder, or tells Lightroom about them via
XMP sidecars. Built for safari-style shooting: burst mode, many
near-identical frames, several animals per day, and no wish to click through
every frame by hand.

```bash
fotosort /Volumes/SDCARD/DCIM/100OMSYS                # dry run: prints what it would pick
fotosort /Volumes/SDCARD/DCIM/100OMSYS --copy --enhance   # enhanced copies of the picks in Highlights/
```

Photo contents are never modified or deleted. A normal run writes a CSV report
and a feature cache. `--move` relocates picks; `--raw-cull-move` relocates culled
RAWs. `--xmp` writes sidecars, replacing existing ones only with `--xmp-overwrite`.

## How it works

Every JPEG gets a handful of signals, computed once and cached in the folder:

| Signal | How |
|---|---|
| **Sharpness** | Detail inside the detected subject, with light noise suppression. A 4x4 tile measure supplies a fallback when no reliable subject region is available. Both measurements appear in the report. |
| **Exposure** | Fraction of clipped black and white pixels. |
| **Aesthetics** | CLIP ViT-L/14 + the [LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), trained to predict human image ratings, including AVA photographs. Runs locally on Apple GPU (MPS), CUDA or CPU. |
| **Composition** | Gentle bonuses for detected subject placement near thirds, golden-ratio lines or the centre. These are alternative compositions, not requirements. No extra image decoding is needed. |
| **Editorial preferences** | Local CLIP contrasts for expressive light, deliberate composition, subject clarity and a compelling moment. Reuses cached image embeddings; no additional images are uploaded. |
| **Subject label** | Zero-shot CLIP label from a list of ~70 subjects (lion, elephant, bird, landscape, sunset, ...). Bring your own list with `--labels`. |
| **Subject box** | YOLOv8 detector for people and animals: how big the main subject is and whether it is cut off by the frame. A person covering at least 2 % of the frame puts the photo in the "people" group, because CLIP alone files a child in a field under "kudu antelope". |
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
   (`--preselect-share 0.5`, rounded up). Scene coverage can expand it further.
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
In a [replay of three reviewed Botlierskop favourites](docs/evaluations/botlierskop-preselection.md),
share 0.5 still missed all three: wider coverage is a safeguard, not a substitute
for improving which similar-burst alternatives are nominated.

The default detail crop shows the subject centre. `--judge-crops multi` adds a
subject overview and native windows at possible head positions, at extra token
cost; there is no eye detector. `--judge-crops none` omits subject details.
Cloud judging remains opt-in; a normal run uses local models only.

**Award winning camera roll (`--mode award-roll --judge`).** Curate one wildlife
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
fallible. "Award winning" describes the editorial brief, not a prediction of an
award. Automated tests check logic and resource bounds; real-photo curation
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

**Experimental alternatives within bursts (`--burst-alternatives`).** This is
off by default: the first [real-photo replay](docs/evaluations/burst-alternatives.md)
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

**Extra local work for difficult choices.** By default, up to
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

**Composition and an award-oriented aesthetic.** The geometric scores measure
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

**Output layout.** `--layout species` puts the picks into one subfolder per
subject (`Highlights/zebra/`, `Highlights/lion/`, `Highlights/people/`),
`--layout day` one per day, `--layout day-species` both. The default is one
flat folder.

**DxO PureRAW.** With `--dxo`, RAW files go through PureRAW first. PureRAW
has no command line, so FotoSort does what can be done: it looks for the
processed twin of each RAW in the `DxO` folder next to it (any of PureRAW's
naming templates, DNG preferred), hands the missing ones to the app with
`open -a`, tells you to press Process, and waits (up to `--dxo-wait` hours)
for the outputs to appear. From then on the twin is what gets scored,
enhanced and copied, while the original RAW keeps its name in the report and
sidecars. `--dxo all` processes every RAW before analysis (the denoised files
are what gets judged); `--dxo picks` scores the originals and processes only
the selected frames, which is many times faster. `--dxo-wait 0` uses existing
outputs only. RAW+JPEG pairs also use their processed RAW twin when available,
and the judge uses the same processed source as feature extraction.

**RAW files** (ORF, NEF, CR2, CR3, ARW, RAF, RW2, DNG, PEF, SRW) are analysed
through their embedded camera preview, so a RAW-only card works directly. A
RAW next to a JPEG with the same stem, or in its `RAW/` subfolder, is
represented by that JPEG. Generated `Highlights`, `AwardRoll`, `Enhanced`, `DxO` and
`_DELETE_ME_raw` directories are excluded from recursive input scans. Enhanced output of a RAW pick is a full-resolution JPEG demosaiced
with the camera white balance, carrying the camera model and capture time.
`--no-raw` ignores RAW files.

## Install

Requires Python 3.11+ and about 1.1 GB of disk for the model weights.

```bash
git clone https://github.com/JRS1986/FotoSort.git
cd FotoSort
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[judge]"     # drop [judge] if you will not use --judge
fotosort /path/to/photos
```

`./fotosort.sh /path` does the same from a clean checkout, creating the
virtualenv on first use. The first run downloads the CLIP weights (~900 MB,
Hugging Face cache), the aesthetic head (~4 MB) and the YOLOv8 detector
(~50 MB) into `~/.cache/fotosort`, and DINOv2-small (~90 MB) into the
Hugging Face cache. Per-image features are cached in
`.fotosort_cache.npz` inside the photo folder, so re-running with different
settings reuses the features. Cache loading uses memory proportional to the
number of photos; saves replace the cache atomically. Caches older than v6 need
one re-analysis; v6 caches retain their existing features and acquire the new
subject views as needed. Earlier throughput on an Apple M4 was roughly 7 images/s
including detection.

## Usage

```bash
fotosort /path                                   # dry run: picks + fotosort_report.csv
fotosort /path --copy --enhance                  # enhanced copies of the picks in /path/Highlights
fotosort /path --move                            # move originals into /path/Highlights
fotosort /path --recursive --with-sidecars       # include subfolders, bring RAW/XMP files along
fotosort /path --max-per-group 3                 # tighter selection
# Use the label file from the checkout:
fotosort /path --labels labels/whale_watching.txt --label-mode image --judge \
    --judge-hint "Whale-watching trip: keepers show the animal itself, not just a blow or splash"
fotosort /path --copy --enhance --layout species # one subfolder per animal: Highlights/zebra/, Highlights/lion/, ...
fotosort /path --dxo picks --copy --enhance      # PureRAW on the picks, then enhance from the DxO DNGs
fotosort /path --include P9051472,P9050886       # frames you know you want, picked regardless
fotosort /path --apply-report --copy --enhance   # re-apply (possibly hand-edited) report picks, no re-analysis
fotosort enhance /any/folder                     # enhance any folder into <folder>/Enhanced
```

### The judge

```bash
export OPENAI_API_KEY=sk-...                     # or --key-file path/to/.env (a KEY=value or key: value line)
fotosort /path --judge                           # OpenAI gpt-5.6-sol, shortlisted frames at high detail
fotosort /path --judge --judge-model gpt-5.6-terra
fotosort /path --judge --judge-provider anthropic # claude-opus-5 via ANTHROPIC_API_KEY
fotosort /path --judge --judge-coverage full     # optional exhaustive pass; preselection is the default
fotosort /path --judge --judge-detail low        # smaller full-image overviews
fotosort /path --judge --judge-crops multi       # optional extra subject details
```

**Local judge, nothing leaves your machine.** Any OpenAI-compatible server
works: Ollama, LM Studio, vLLM, an MLX server. With Ollama:

```bash
ollama pull qwen3-vl:8b
printf 'FROM qwen3-vl:8b\nPARAMETER num_ctx 16384\n' > Modelfile && ollama create qwen3-vl-judge -f Modelfile
fotosort /path --judge --judge-base-url http://localhost:11434/v1 --judge-model qwen3-vl-judge \
    --judge-detail low --judge-chunk 6
```

No key is needed. The `ollama create` step matters: Ollama loads the model
with its full 262k context by default, which spills most of it to the CPU and
makes each request take minutes; a 16k context fits the GPU. Local models are
slower and less discerning than the frontier ones, so keep chunks small
(`--judge-chunk 6`), send frames at low detail, and consider
a small `--preselect`. Expect several seconds per frame on an
Apple-silicon Mac.

With a cloud provider, the judge sends downsized overviews and subject details
to that provider's API. Do not use it on photos you must not upload. Full
coverage and multiple detail windows cost more than the default shortlist and
single crop; cost depends
on the model, number of distinct files, detections, keeper budgets and rounds.
The final token counts show actual usage. `--judge-detail low` reduces the full
image size, and `--judge-crops none` removes the extra detail image. Locally
cached image encodings save decoding work, not the tokens for repeat submissions.
The report's `shortlisted` column shows what the judge saw and `judge_reason`
records its reasons or the stage at which a frame was excluded. Byte-identical
copies are marked as such; similar thumbnails alone never cause this removal.

### Which RAWs to keep: edit benefit, presets, RAW cull

Every frame gets an **edit benefit** score (0..100, `edit_score`) from what a
RAW file could still recover: blown highlights weigh most, then crushed
shadows, a dark or flat exposure, and high ISO noise (ISO is read from EXIF).
Judged picks additionally get the judge's own verdict (`edit_benefit`
low/medium/high with a few words in `edit_why`, e.g. "sky to recover") and a
**preset** suggestion from a fixed menu: Natural, Color Pop, Wildlife Crisp,
Golden Hour Warmth, Moody Colors, Low Key Dramatic, High Key, Black & White,
Vignette, Portrait Soft. Non-judged picks get a preset from the detected
style. Both land in the report and, with `--xmp`, as keywords
(`FotoSort|Preset|Color Pop`, `FotoSort|RAW edit high`).

```bash
fotosort /path --raw-cull                        # raw_keep.txt / raw_cull.txt: RAWs of picks vs the rest
fotosort /path --raw-cull --raw-keep-benefit medium   # keep a pick's RAW only if editing it pays
fotosort /path --apply-report --raw-cull --raw-cull-move   # move the culled RAWs into _DELETE_ME_raw
```

`--raw-cull` finds each frame's RAW (same stem next to it or in a `RAW/`
subfolder) and keeps the RAWs of picks whose benefit reaches
`--raw-keep-benefit`. `--raw-cull-move` moves the others into a
`_DELETE_ME_raw` folder for you to empty; nothing is ever deleted. Decisions
are combined per RAW file, and any qualifying keep decision protects it from
culling even if another representation was rejected.

### Lightroom, Capture One, Bridge, digiKam: XMP sidecars

```bash
fotosort /path --xmp picks                       # <stem>.xmp next to every pick: rating 5, keywords, judge reason
fotosort /path --judge --xmp all                 # picks 5 stars, judged-but-not-picked 3, rejected 1, others 0
```

Sidecars carry `xmp:Rating`, a colour label (green for picks, red for
rejects), keywords `FotoSort`, `FotoSort|<subject>` and `FotoSort|Pick`, and the
judge's reason as the description. Nothing is moved or copied. Existing
sidecars are left untouched (Lightroom keeps develop settings in them) unless
you pass `--xmp-overwrite`. Lightroom reads sidecars for RAW files; for JPEGs
it expects embedded metadata, so there use Capture One, Bridge or digiKam, or
embed with `exiftool -tagsfromfile %d%f.xmp -all:all photo.jpg`.

### Enhancing the picks

```bash
fotosort /path --copy --enhance                  # the enhanced version is the copy
fotosort /path --move --enhance                  # originals moved, enhanced versions in Highlights/Enhanced/
fotosort /path --copy --enhance --enhance-strength 0.8
fotosort enhance IMG_0042.jpg --style landscape --strength 0.8
```

`--enhance` detects the style of each pick (wildlife, landscape, golden hour,
portrait, night, black and white, food, urban, water, macro) and applies a
matching recipe: white balance (skipped for golden hour and strongly tinted
scenes), auto levels, shadow/highlight tone curve, an S-curve for contrast,
vibrance (skin protected for portraits), local contrast, colour denoise for
night shots, sharpening with a noise threshold, and a subtle vignette for
wildlife. A detected person always gets the gentle portrait recipe. Every
amount is scaled by the image's own statistics, so an already colourful or
contrasty frame gets a lighter touch, and a colour photo is never turned
monochrome. Output is full-resolution JPEG (quality 92) with the original
EXIF and colour profile.

### All options

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | standard | `standard`: highlights per day/subject; `award-roll`: one wildlife portfolio across all dates, requires `--judge` |
| `--roll-size` | 12 | award-roll target and maximum (10–15); fewer picks are allowed when warranted |
| `--award-candidates` | 480 | strict global candidate cap for award-roll preselection; full coverage explicitly bypasses it |
| `--award-plan` | off | local-only award candidate plan and comparison bounds; no judge calls or photo exports; writes award_roll_plan.csv |
| `--move` / `--copy` | dry run | move or copy the picks into the output folder |
| `--highlights` | Highlights (AwardRoll in award-roll mode) | name of the output subfolder |
| `--layout` | flat | `species`, `day` or `day-species` subfolders inside the output folder |
| `--dxo` | off | `all`: PureRAW on every RAW before analysis; `picks`: only on the selected frames |
| `--dxo-app` / `--dxo-dir` / `--dxo-wait` | PureRAW 6 / `DxO` next to the RAWs / 8 h | hand-off application, output folder, wait time (0 = existing outputs only) |
| `--recursive` | | also scan subfolders |
| `--no-raw` | | ignore RAW files (by default RAW files without a same-named JPEG are analysed) |
| `--with-sidecars` | | move/copy RAW and XMP files with the same name |
| `--max-per-group` | 5 | base picks per subject per day |
| `--extra-per` | 40 | one extra pick per this many photos in a group, capped at 3x base (0 = off) |
| `--gap-seconds` | 120 | time gap that starts a new scene |
| `--scene-sim` | 0.80 | min CLIP similarity to stay in a scene |
| `--moment-gap-seconds` | 8 | capture gap that starts a finer moment within a scene |
| `--moment-sim` | 0.92 | whole-frame or subject DINO similarity below which a new moment starts |
| `--subject-embeddings` / `--no-subject-embeddings` | on | use cached local DINO subject views for moments, diversity and duplicate comparisons |
| `--burst-alternatives` / `--no-burst-alternatives` | off | experimental burst nomination and packing, with additional cached local crops for judge preselection |
| `--focus-refine-per-group` | 12 | maximum frames per day/subject for extra local focus comparisons; 0 disables |
| `--label-mode` | scene | `image` labels every frame on its own |
| `--labels` | built-in list | text file with one subject label per line |
| `--skip-labels` | document or screenshot | comma-separated subjects never selected |
| `--dup-sim` | 0.89 | DINOv2 threshold for local near-duplicates; thumbnail framing must also agree |
| `--dup-pixel` | 0.90 | thumbnail framing threshold used together with DINOv2 |
| `--dup-reframe-sim` | 0.98 | strong whole-image DINO match allowing reframed local duplicates within a scene when subject DINO also agrees; 0 disables |
| `--min-score` | disabled | optional relative score cutoff for selection without the judge |
| `--aesthetic-weight` | 0.6 | learned aesthetics vs sharpness weight in the score |
| `--composition-weight` | 0.2 | maximum bonus for thirds, golden-ratio or central placement; 0 disables |
| `--editorial-weight` | 0.15 | maximum local CLIP preference bonus for light, composition, subject and moment; 0 disables |
| `--subject-weight` | 0.4 | bonus for a large detected subject, penalty for a small cut-off one |
| `--blur-ratio` / `--blur-floor` | 0.15 / 20 | local blur rejection thresholds (relative to a known scene / absolute); the judge assesses blur visually |
| `--person-area` | 0.02 | min person box fraction for the people group (0 disables the detector) |
| `--detector` | yolov8m.pt | YOLOv8 weights (`yolov8n.pt` is 3x faster, m is better on birds) |
| `--judge` | off | let a vision model choose the final picks |
| `--judge-provider` / `--judge-model` | openai / gpt-5.6-sol | API and model (`anthropic` / `claude-opus-5`) |
| `--judge-coverage` / `--judge-chunk` | preselect / 24 | `preselect`: local shortlist; `full`: every distinct eligible file, bypassing the award candidate cap. Standard winner comparisons use at most the larger of chunk size and twice the keeper budget; award-roll uses four times the roll target |
| `--preselect` / `--preselect-min` / `--preselect-share` | 3 / 12 / 0.5 | standard-mode shortlist: keeper-budget multiple, minimum and group-share floor. Scene coverage may expand it; award-roll uses its own strict global cap |
| `--preselect-explore` | 2 | maximum existing shortlist slots for unusual/uncertain alternatives; 0 disables |
| `--include` | | frames that must be picked regardless (stems, names, or `@file`) |
| `--judge-detail` | high | OpenAI full-image detail: high uses a 1024 px overview, low a 512 px overview |
| `--judge-crops` | single | `single`: one native subject detail; `multi`: overview plus several native details; `none`: full image only |
| `--judge-hint` | | a sentence or two about this shoot for the judge |
| `--key-file` | | read the judge API key from a `.env` or YAML file |
| `--judge-base-url` | | OpenAI-compatible server for a local judge (needs `--judge-model`) |
| `--xmp` / `--xmp-overwrite` | off | write XMP sidecars for `picks` or `all`; replace existing ones |
| `--raw-cull` / `--raw-keep-benefit` / `--raw-cull-move` | off / low / off | list RAWs to keep vs cull; keep threshold; move the culled ones to `_DELETE_ME_raw` |
| `--taste-dir` / `--taste-weight` | off / 0.5 | folder of photos you love; frames resembling them get up to this bonus |
| `--enhance` | | enhance the picks (see above) |
| `--enhance-strength` / `--enhance-style` | 1.0 / auto | global strength; force one style |
| `--apply-report` | | skip analysis; act on the `selected=1` rows of the existing report |
| `--report` | fotosort_report.csv (award_roll_report.csv in award-roll mode) | CSV report path, relative to the photo folder |
| `--no-cache` | | ignore and do not write the feature cache |
| `--limit N` | | only process the first N files |
| `--batch-size` / `--workers` | 32 / 4 | GPU batch size and decoder threads |

## Project layout

```
fotosort/
  cli.py      argument parsing and the pipeline: scan, features, scenes, score, select, judge, move
  quality.py  sharpness, exposure, thumbnail signature, JPEG/RAW decoding (PIL, numpy, rawpy)
  composition.py  geometric composition guides and local CLIP editorial preferences
  embed.py    CLIP embeddings, aesthetic head, zero-shot labels, DINOv2 sameness, YOLO subject detection
  burst.py    experimental local expression/interaction cues and bounded subject views
  xmp.py      XMP sidecar writer
  dxo.py      DxO PureRAW twins: find, hand off, wait
  editing.py  edit-benefit score and the preset menu
  group.py    scene / burst clustering by time and similarity
  selection.py  score-based selection, judge preselection and tournament chunking
  judge.py    the vision-model judge (OpenAI / Anthropic), prompts and verdict parsing
  enhance.py  style detection and enhancement recipes, also the `enhance` subcommand
  scan.py     JPEG discovery, EXIF capture time, RAW/XMP sidecars
  labels.py   default subject list and the people group
  cache.py    per-folder feature cache (.fotosort_cache.npz)
labels/       ready-made label sets, e.g. whale_watching.txt
tests/        pytest unit tests + make_testset.py (synthetic EXIF-dated images)
```

## Checking selection quality

The regression suite checks that distinctive scenes reach preselection, all
of a group's strongest candidates can advance from one batch, subject focus
is not replaced by background texture, and unrelated days do not change scores.
They also check geometric guide locations, bounded bonuses, unknown detections,
disabled preferences and independent aesthetic nominations at a fixed shortlist
size. These are deterministic checks, not a claim of measured photographic accuracy.

For evaluation on your own photos, keep a separate set of manually reviewed
reports. Compare missed keepers, duplicate picks and best-frame agreement within
bursts, including low light, small subjects, action and unusual compositions.
Keep these reference selections fixed while tuning; use another shoot to check
that improvements transfer. The `decision`, `shortlisted` and `judge_reason`
columns help locate the stage responsible for a missed frame.

## Development

```bash
pip install -e ".[judge,dev]"
ruff check fotosort tests
pytest
python tests/make_testset.py /tmp/testset && fotosort /tmp/testset
```

The unit tests cover the pure logic (metrics, grouping, selection, enhancement,
report parsing) and run without the model weights, so CI needs no GPU.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, conventions and what makes a useful bug report for a culling tool,
and [CHANGELOG.md](CHANGELOG.md) for what has changed and what is planned.

## Credits

- [OpenAI CLIP](https://github.com/openai/CLIP) via [open_clip](https://github.com/mlfoundations/open_clip)
- [LAION improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DINOv2](https://github.com/facebookresearch/dinov2) via [timm](https://github.com/huggingface/pytorch-image-models)
- [rawpy](https://github.com/letmaik/rawpy) (LibRaw) and [ExifRead](https://github.com/ianare/exif-py)

## License

Copyright (c) 2026 Jan R. Seyler. FotoSort is licensed under the GNU Affero
General Public License, version 3 only (`AGPL-3.0-only`); see [LICENSE](LICENSE)
and [NOTICE](NOTICE).

Dependencies and downloaded model weights retain their own licenses. The
required Ultralytics package and YOLO detector weights use AGPL-3.0 by default;
see [Ultralytics' licensing guidance](https://www.ultralytics.com/license).
