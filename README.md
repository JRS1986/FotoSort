# FotoSort

[![CI](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml/badge.svg)](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Picks the best, most varied photos out of a folder with thousands of JPEGs or
RAW files and puts them into a subfolder, or tells Lightroom about them via
XMP sidecars. Built for safari-style shooting: burst mode, many
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
| **Sameness** | A DINOv2 embedding. Unlike CLIP, which is semantic (two poses of one animal score 0.95), DINOv2 measures visual sameness: burst twins score above 0.9, different compositions of the same subject below 0.88 on a calibrated safari set. A tiny normalised thumbnail additionally catches pixel-identical frames. |

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
   it is not a near-duplicate (DINOv2 similarity or thumbnail correlation) of
   anything already taken that day, in any group. The score is a weighted
   z-score of aesthetics and log-sharpness, plus a bonus for a large detected
   subject, minus penalties for clipping and for a small subject cut off by
   the frame.
4. **Judge (optional, `--judge`).** A vision model picks the keepers the way a
   photo editor would: subject visible and sharp, faces not in shadow, best of
   each burst, as varied as possible. By default (`--judge-coverage
   preselect`) the scores first narrow each group to `--preselect` times its
   budget (default 3x, at least 12 frames and at least half the group, so a
   long sighting keeps its distinct moments), taking the best frame of every
   burst first and then the frames most unlike what is already in, and the
   judge runs its tournament on those: time-ordered
   chunks of `--judge-chunk` frames, each sending on its share of the budget,
   knock-out rounds until a final. That is roughly a quarter of the tokens of
   `--judge-coverage full`, where the judge sees every distinct frame. Every
   frame goes over twice, the whole picture plus a native-resolution crop of the
   detected subject, so the judge can see whether the eye is actually sharp.
   Each pick gets a one-line reason in the report. `--judge-hint` tells the
   judge what counts as a keeper on this particular shoot.

The CSV report lists every score, label, group, whether the judge saw the frame,
and why it was picked, so you can see exactly why a frame was or was not chosen.
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
outputs only.

**RAW files** (ORF, NEF, CR2, CR3, ARW, RAF, RW2, DNG, PEF, SRW) are analysed
through their embedded camera preview, so a RAW-only card works directly. A
RAW that sits next to a JPEG with the same name is skipped, the JPEG stands
for both. Enhanced output of a RAW pick is a full-resolution JPEG demosaiced
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
settings takes seconds. Throughput on an Apple M4 is roughly 7 images/s
including detection.

## Usage

```bash
fotosort /path                                   # dry run: picks + fotosort_report.csv
fotosort /path --copy --enhance                  # enhanced copies of the picks in /path/Highlights
fotosort /path --move                            # move originals into /path/Highlights
fotosort /path --recursive --with-sidecars       # include subfolders, bring RAW/XMP files along
fotosort /path --max-per-group 3                 # tighter selection
fotosort /path --labels labels/whale_watching.txt --label-mode image --judge \   # label file from the checkout
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
fotosort /path --judge                           # OpenAI gpt-5.6-sol, every frame at high detail
fotosort /path --judge --judge-model gpt-5.6-terra
fotosort /path --judge --judge-provider anthropic # claude-opus-5 via ANTHROPIC_API_KEY
fotosort /path --judge --judge-coverage full     # the judge sees every distinct frame (about 4x the tokens)
fotosort /path --judge --judge-detail low        # the cheap variant
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

With a cloud provider, the judge sends downsized JPEGs (and subject crops) of
every distinct frame to the provider's API. Do not use it on photos you must
not upload. Cost scales
with the number of frames: full coverage of a 2,000-photo folder is roughly
500k input tokens with the default preselection, around $1 with
`gpt-5.6-terra` and $2 with `gpt-5.6-sol`, and about four times that with
`--judge-coverage full`;
`--judge-detail low` is about a tenth of that. The
report's `shortlisted` column shows what the judge saw and `judge_reason` why
it picked a frame; frames that were pixel-identical to a sibling are marked
"twin of X" and never sent twice.

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
`_DELETE_ME_raw` folder for you to empty; nothing is ever deleted.

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
| `--move` / `--copy` | dry run | move or copy the picks into the output folder |
| `--highlights` | Highlights | name of the output subfolder |
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
| `--label-mode` | scene | `image` labels every frame on its own |
| `--labels` | built-in list | text file with one subject label per line |
| `--skip-labels` | document or screenshot | comma-separated subjects never selected |
| `--dup-sim` | 0.89 | DINOv2 similarity above which two picks count as duplicates |
| `--dup-pixel` | 0.90 | thumbnail correlation above which two picks count as duplicates |
| `--min-score` | -0.5 | never pick a photo scoring below this |
| `--aesthetic-weight` | 0.6 | aesthetics vs sharpness weight in the score |
| `--subject-weight` | 0.4 | bonus for a large detected subject, penalty for a small cut-off one |
| `--blur-ratio` / `--blur-floor` | 0.15 / 20 | blur rejection thresholds (relative to the day's median / absolute); with the judge on, only the floor applies |
| `--person-area` | 0.02 | min person box fraction for the people group (0 disables the detector) |
| `--detector` | yolov8m.pt | YOLOv8 weights (`yolov8n.pt` is 3x faster, m is better on birds) |
| `--judge` | off | let a vision model choose the final picks |
| `--judge-provider` / `--judge-model` | openai / gpt-5.6-sol | API and model (`anthropic` / `claude-opus-5`) |
| `--judge-coverage` / `--judge-chunk` | preselect / 24 | `preselect`: score-based preselection then the tournament; `full`: every frame |
| `--preselect` / `--preselect-min` / `--preselect-share` | 3 / 12 / 0.5 | preselection size: a multiple of the group budget, a minimum, and a minimum share of the group |
| `--include` | | frames that must be picked regardless (stems, names, or `@file`) |
| `--judge-detail` | high | image detail for the OpenAI judge (`low` is ~10x cheaper) |
| `--judge-hint` | | a sentence or two about this shoot for the judge |
| `--key-file` | | read the judge API key from a `.env` or YAML file |
| `--judge-base-url` | | OpenAI-compatible server for a local judge (needs `--judge-model`) |
| `--xmp` / `--xmp-overwrite` | off | write XMP sidecars for `picks` or `all`; replace existing ones |
| `--raw-cull` / `--raw-keep-benefit` / `--raw-cull-move` | off / low / off | list RAWs to keep vs cull; keep threshold; move the culled ones to `_DELETE_ME_raw` |
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
  quality.py  sharpness, exposure, thumbnail signature, JPEG/RAW decoding (PIL, numpy, rawpy)
  embed.py    CLIP embeddings, aesthetic head, zero-shot labels, DINOv2 sameness, YOLO subject detection
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

MIT, see [LICENSE](LICENSE).
