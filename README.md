# FotoSort

[![CI](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml/badge.svg)](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Picks the best, most varied photos out of a folder with thousands of JPEGs and
puts them into a subfolder. Built for safari-style shooting: burst mode, many
near-identical frames, several animals per day, and no wish to click through
every frame by hand.

```bash
fotosort /Volumes/SDCARD/DCIM/100OMSYS                # dry run: prints what it would pick
fotosort /Volumes/SDCARD/DCIM/100OMSYS --copy --enhance   # enhanced copies of the picks in Highlights/
```

Originals are never modified or deleted. Without `--move` or `--copy` the tool
only writes a CSV report.

## How it works

Every JPEG gets a handful of signals, computed once and cached in the folder:

| Signal | How |
|---|---|
| **Sharpness** | Variance of the Laplacian, max over a 4x4 tile grid, on a fast reduced-size decode. Taking the max rewards a sharp subject on a soft background. |
| **Exposure** | Fraction of clipped black and white pixels. |
| **Aesthetics** | CLIP ViT-L/14 embedding + the [LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), a 1..10 score. Runs on Apple GPU (MPS), CUDA or CPU. |
| **Subject label** | Zero-shot CLIP label from a list of ~70 subjects (lion, elephant, bird, landscape, sunset, ...). Bring your own list with `--labels`. |
| **Subject box** | YOLOv8 detector for people and animals: how big the main subject is and whether it is cut off by the frame. A person covering at least 2 % of the frame puts the photo in the "people" group, because CLIP alone files a child in a field under "kudu antelope". |
| **Signature** | A tiny normalised thumbnail. Its correlation between two frames tells identical framings apart from merely similar content, which CLIP cannot. |

Then the folder is organised:

1. **Scenes.** Within one day, consecutive shots closer than `--gap-seconds` and
   visually similar form one scene (a burst). By default each scene gets one
   label from its mean embedding, so a burst stays together. On a boat or
   anywhere the background never changes, `--label-mode image` labels every
   frame on its own instead, so bursts do not merge into one giant group.
2. **Groups.** Photos are grouped by *(day, subject)*, e.g. "2026-09-01 /
   elephant". All people labels share one "people" group. Each group gets a
   pick budget: `--max-per-group` (default 5) plus one per `--extra-per`
   photos, capped at three times the base.
3. **Selection by score.** Per day, candidates are visited best-first. One is
   taken if its score is above `--min-score`, its group still has budget, and
   it is not a near-duplicate (CLIP similarity or thumbnail correlation) of
   anything already taken that day, in any group. The score is a weighted
   z-score of aesthetics and log-sharpness, plus a bonus for a large detected
   subject, minus penalties for clipping and for a small subject cut off by
   the frame.
4. **Judge (optional, `--judge`).** A vision model picks the keepers the way a
   photo editor would: subject visible and sharp, faces not in shadow, best of
   each burst, as varied as possible. It sees every frame: a group is judged in
   time-ordered chunks of `--judge-chunk` frames, each chunk sends on its share
   of the budget, and the winners meet in knock-out rounds until a final. Every
   frame goes over twice, the whole picture plus a native-resolution crop of the
   detected subject, so the judge can see whether the eye is actually sharp.
   Each pick gets a one-line reason in the report. `--judge-hint` tells the
   judge what counts as a keeper on this particular shoot.

The CSV report lists every score, label, group, whether the judge saw the frame,
and why it was picked, so you can see exactly why a frame was or was not chosen.
Edit the `selected` column and run `--apply-report` to override any decision.

## Install

Requires Python 3.11+ and about 1 GB of disk for the model weights.

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
(~50 MB) into `~/.cache/fotosort`. Per-image features are cached in
`.fotosort_cache.npz` inside the photo folder, so re-running with different
settings takes seconds. Throughput on an Apple M4 is roughly 7 images/s
including detection.

## Usage

```bash
fotosort /path                                   # dry run: picks + fotosort_report.csv
fotosort /path --copy --enhance                  # enhanced copies of the picks in /path/Highlights
fotosort /path --move                            # move originals into /path/Highlights
fotosort /path --recursive --with-sidecars       # include subfolders, bring RAW/XMP files along
fotosort /path --max-per-group 3                 # tighter selection
fotosort /path --labels labels/whale_watching.txt --label-mode image --judge \
    --judge-hint "Whale-watching trip: keepers show the animal itself, not just a blow or splash"
fotosort /path --apply-report --copy --enhance   # re-apply (possibly hand-edited) report picks, no re-analysis
fotosort enhance /any/folder                     # enhance any folder into <folder>/Enhanced
```

### The judge

```bash
export OPENAI_API_KEY=sk-...                     # or --key-file path/to/.env (a KEY=value or key: value line)
fotosort /path --judge                           # OpenAI gpt-5.6-sol, every frame at high detail
fotosort /path --judge --judge-model gpt-5.6-terra
fotosort /path --judge --judge-provider anthropic # claude-opus-5 via ANTHROPIC_API_KEY
fotosort /path --judge --judge-coverage shortlist --judge-detail low   # the cheap variant
```

The judge sends downsized JPEGs (and subject crops) of every distinct frame to
the provider's API. Do not use it on photos you must not upload. Cost scales
with the number of frames: full coverage of a 2,000-photo folder is roughly
2M input tokens, around $4 with `gpt-5.6-terra` and $8 with `gpt-5.6-sol`;
`--judge-coverage shortlist --judge-detail low` is about a tenth of that. The
report's `shortlisted` column shows what the judge saw and `judge_reason` why
it picked a frame; frames that were pixel-identical to a sibling are marked
"twin of X" and never sent twice.

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
| `--move` / `--copy` | dry run | move or copy the picks into the output folder |
| `--highlights` | Highlights | name of the output subfolder |
| `--recursive` | | also scan subfolders |
| `--with-sidecars` | | move/copy RAW and XMP files with the same name |
| `--max-per-group` | 5 | base picks per subject per day |
| `--extra-per` | 40 | one extra pick per this many photos in a group, capped at 3x base (0 = off) |
| `--gap-seconds` | 120 | time gap that starts a new scene |
| `--scene-sim` | 0.80 | min CLIP similarity to stay in a scene |
| `--label-mode` | scene | `image` labels every frame on its own |
| `--labels` | built-in list | text file with one subject label per line |
| `--skip-labels` | document or screenshot | comma-separated subjects never selected |
| `--dup-sim` | 0.95 | CLIP similarity above which two picks count as duplicates |
| `--dup-pixel` | 0.90 | thumbnail correlation above which two picks count as duplicates |
| `--min-score` | -0.5 | never pick a photo scoring below this |
| `--aesthetic-weight` | 0.6 | aesthetics vs sharpness weight in the score |
| `--subject-weight` | 0.4 | bonus for a large detected subject, penalty for a small cut-off one |
| `--blur-ratio` / `--blur-floor` | 0.15 / 20 | blur rejection thresholds (relative to the day's median / absolute); with the judge on, only the floor applies |
| `--person-area` | 0.02 | min person box fraction for the people group (0 disables the detector) |
| `--detector` | yolov8m.pt | YOLOv8 weights (`yolov8n.pt` is 3x faster, m is better on birds) |
| `--judge` | off | let a vision model choose the final picks |
| `--judge-provider` / `--judge-model` | openai / gpt-5.6-sol | API and model (`anthropic` / `claude-opus-5`) |
| `--judge-coverage` / `--judge-chunk` | full / 24 | every frame in chunks of this size, or `shortlist` |
| `--shortlist-max` | 30 | frames per group in shortlist coverage (or 3x the budget) |
| `--judge-detail` | high | image detail for the OpenAI judge (`low` is ~10x cheaper) |
| `--judge-hint` | | a sentence or two about this shoot for the judge |
| `--key-file` | | read the judge API key from a `.env` or YAML file |
| `--taste-dir` / `--taste-weight` | off / 0.5 | folder of photos you love; frames resembling them get up to this bonus |
| `--enhance` | | enhance the picks (see above) |
| `--enhance-strength` / `--enhance-style` | 1.0 / auto | global strength; force one style |
| `--apply-report` | | skip analysis; act on the `selected=1` rows of the existing report |
| `--report` | fotosort_report.csv | CSV report path, relative to the photo folder |
| `--no-cache` | | ignore and do not write the feature cache |
| `--limit N` | | only process the first N files |
| `--batch-size` / `--workers` | 32 / 4 | GPU batch size and decoder threads |

## Project layout

```
fotosort/
  cli.py      argument parsing and the pipeline: scan, features, scenes, score, select, judge, move
  quality.py  sharpness, exposure and thumbnail signature (PIL + numpy)
  embed.py    CLIP embeddings, aesthetic head, zero-shot labels, YOLO subject detection
  group.py    scene / burst clustering by time and similarity
  select.py   score-based selection, judge shortlists and tournament chunking
  judge.py    the vision-model judge (OpenAI / Anthropic), prompts and verdict parsing
  enhance.py  style detection and enhancement recipes, also the `enhance` subcommand
  scan.py     JPEG discovery, EXIF capture time, RAW/XMP sidecars
  labels.py   default subject list and the people group
  cache.py    per-folder feature cache (.fotosort_cache.npz)
labels/       ready-made label sets, e.g. whale_watching.txt
tests/        pytest unit tests + make_testset.py (synthetic EXIF-dated images)
```

## Development

```bash
pip install -e ".[judge,dev]"
ruff check fotosort tests
pytest
python tests/make_testset.py /tmp/testset && fotosort /tmp/testset
```

The unit tests cover the pure logic (metrics, grouping, selection, enhancement,
report parsing) and run without the model weights, so CI needs no GPU.

## Credits

- [OpenAI CLIP](https://github.com/openai/CLIP) via [open_clip](https://github.com/mlfoundations/open_clip)
- [LAION improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## License

MIT, see [LICENSE](LICENSE).
