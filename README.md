# FotoSort

Preselects the best, most varied photos out of a folder with thousands of JPEGs
and moves them into a `Highlights/` subfolder. Built for safari-style shooting:
burst mode, many near-identical frames, several animals per day, and no wish to
click through every frame by hand.

```bash
./fotosort.sh /Volumes/SDCARD/DCIM/100CANON          # dry run: shows what it would pick
./fotosort.sh /Volumes/SDCARD/DCIM/100CANON --move   # move the picks into Highlights/
```

## How it picks

Every JPEG gets four signals:

| Signal | How |
|---|---|
| **Sharpness** | Variance of the Laplacian, max over a 4x4 tile grid. Taking the max rewards a sharp subject on a soft background. Computed on a fast reduced-size decode. |
| **Exposure** | Fraction of clipped black and white pixels. |
| **Aesthetics** | CLIP ViT-L/14 embedding + the [LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (score 1..10). Runs on the Apple GPU via MPS, CUDA, or CPU. |
| **Subject** | Zero-shot CLIP label from a list of ~70 subjects (lion, elephant, bird, landscape, sunset, ...). |
| **People** | YOLOv8-nano person detector. A person covering at least 2 % of the frame puts the photo in the "people" bucket, because CLIP alone happily files a child in a field under "kudu antelope". |

Then the folder is organised:

1. **Scenes.** Within one day, consecutive shots closer than `--gap-seconds` and
   visually similar form one scene (a burst). Each scene is labelled by its
   mean embedding, so a whole burst lands in the same subject bucket even if
   single frames are ambiguous.
2. **Buckets.** Photos are grouped by *(day, subject)*, e.g. "2026-09-01 /
   elephant". All people labels (child, man, woman, group, selfie, ...) share
   one "people" bucket.
3. **Selection.** In each bucket the tool first takes the best frame of every
   scene, then fills the pick budget (`--max-per-group`, default 5, plus one
   per `--extra-per` photos in the group, capped at 3x) with further frames
   that are not near-duplicates of an existing pick. Duplicates are detected
   two ways: CLIP similarity (same content) and a tiny normalised thumbnail
   correlation (same framing). A bucket of identical frames yields one pick. Blurry frames (below 30 %
   of the day's median sharpness) and badly clipped frames are never picked.

The combined score is a weighted z-score of aesthetics and log-sharpness minus
an exposure penalty. A CSV report lists every score so you can see why a frame
was or was not chosen.

## Install

Requires Python 3.11+ and about 1 GB of disk for the CLIP weights.

```bash
git clone git@github.com:JRS1986/FotoSort.git
cd FotoSort
./fotosort.sh /path/to/photos     # creates .venv and installs deps on first run
```

Or manually:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m fotosort /path/to/photos
```

The first run downloads the CLIP weights (~900 MB, Hugging Face cache), the
aesthetic head (~4 MB) and the YOLOv8n detector (~6 MB), both into `~/.cache/fotosort`. Per-image features are cached in
`.fotosort_cache.npz` inside the photo folder, so re-running with different
thresholds takes seconds. Throughput on an Apple M4 is roughly 10-20 images/s
after model load.

## Usage

```bash
./fotosort.sh /path                              # dry run, prints picks + writes fotosort_report.csv
./fotosort.sh /path --move                       # move picks into /path/Highlights
./fotosort.sh /path --copy                       # copy instead of move
./fotosort.sh /path --recursive --with-sidecars  # include subfolders, bring RAW/XMP files along
./fotosort.sh /path --max-per-group 3
./fotosort.sh /path --labels my_labels.txt       # own subject list, one label per line
./fotosort.sh /path --blur-ratio 0               # keep soft bursts (e.g. the only leopard of the trip)
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-per-group` | 5 | base picks per subject per day |
| `--extra-per` | 40 | one extra pick per this many photos in a group, capped at 3x base (0 = off) |
| `--gap-seconds` | 120 | time gap that starts a new scene |
| `--scene-sim` | 0.80 | min cosine similarity to stay in a scene |
| `--dup-sim` | 0.985 | CLIP similarity above which two picks count as duplicates |
| `--dup-pixel` | 0.93 | thumbnail correlation above which two picks count as duplicates |
| `--aesthetic-weight` | 0.6 | aesthetics vs sharpness weight in the score |
| `--blur-ratio` / `--blur-floor` | 0.3 / 20 | blur rejection thresholds (relative to day median / absolute) |
| `--skip-labels` | document or screenshot | subjects never selected |
| `--person-area` | 0.02 | min person box fraction for the people bucket (0 = no detector) |
| `--highlights` | Highlights | name of the output subfolder |
| `--report` | fotosort_report.csv | CSV report path, relative to the photo folder |
| `--no-cache` | | ignore and do not write the feature cache |
| `--limit N` | | only process the first N files (for testing) |
| `--enhance` | | enhance the picks: with `--copy` the enhanced version is the copy, with `--move` it goes to `Highlights/Enhanced/` |
| `--enhance-strength` | 1.0 | 0 = untouched, 1.5 = punchy |
| `--enhance-style` | auto | force one style for all picks |

### Enhancing the picks

```bash
./fotosort.sh /path --copy --enhance                  # enhanced copies of the picks in Highlights/, originals untouched
./fotosort.sh /path --move --enhance                  # originals moved to Highlights/, enhanced versions in Highlights/Enhanced/
./fotosort.sh /path --copy --enhance --enhance-strength 1.3
./fotosort.sh enhance /any/folder                     # enhance any folder into <folder>/Enhanced
./fotosort.sh enhance IMG_0042.jpg --style landscape --strength 0.8
```

`--enhance` detects the style of each pick with CLIP (wildlife, landscape,
golden hour, portrait, night, black and white, food, urban, water, macro) and
applies a matching recipe: white balance (skipped for golden hour), auto
levels, shadow/highlight tone curve, an S-curve for contrast, vibrance (skin
protected for portraits), local contrast ("clarity"), colour denoise for
night shots, sharpening with a noise threshold, and a subtle vignette for
wildlife and portraits. Every amount is scaled by the image's own statistics,
so an already colourful or contrasty frame gets a lighter touch. Output is
full resolution JPEG (quality 92) with the original EXIF and colour profile.
Originals are never modified.

Nothing is deleted, ever. Without `--move` or `--copy` the tool only writes the
report. `Highlights/` is excluded from scanning, so running twice is safe.

## Project layout

```
fotosort/
  cli.py      argument parsing and the pipeline: scan, features, scenes, score, select, move
  quality.py  sharpness and exposure metrics (PIL + numpy)
  embed.py    CLIP embeddings, aesthetic head, zero-shot classification
  group.py    scene / burst clustering by time and similarity
  select.py   diverse top-k selection inside a bucket
  scan.py     JPEG discovery, EXIF capture time, RAW/XMP sidecars
  labels.py   default subject list
  cache.py    per-folder feature cache (.fotosort_cache.npz)
  enhance.py  style detection + enhancement recipes, also the `enhance` subcommand
tests/        pytest unit tests + make_testset.py (synthetic EXIF-dated images)
```

## Development

```bash
.venv/bin/python -m pytest tests
.venv/bin/python tests/make_testset.py /tmp/testset && ./fotosort.sh /tmp/testset
```

## Credits

- [OpenAI CLIP](https://github.com/openai/CLIP) via [open_clip](https://github.com/mlfoundations/open_clip)
- [LAION improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
