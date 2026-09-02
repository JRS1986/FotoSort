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
| **Subject** | Zero-shot CLIP label from a list of ~60 subjects (lion, elephant, bird, landscape, sunset, people, ...). |

Then the folder is organised:

1. **Scenes.** Within one day, consecutive shots closer than `--gap-seconds` and
   visually similar form one scene (a burst). Each scene is labelled by its
   mean embedding, so a whole burst lands in the same subject bucket even if
   single frames are ambiguous.
2. **Buckets.** Photos are grouped by *(day, subject)*, e.g. "2026-09-01 /
   elephant".
3. **Selection.** In each bucket the tool first takes the best frame of every
   scene, then fills up to `--max-per-group` (default 5) with further frames
   that are not near-duplicates of an existing pick. Blurry frames (below 30 %
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

The first run downloads the CLIP weights (~900 MB, Hugging Face cache) and the
aesthetic head (~4 MB, `~/.cache/fotosort`). Per-image features are cached in
`.fotosort_cache.npz` inside the photo folder, so re-running with different
thresholds takes seconds. Throughput on an Apple M4 is roughly 10-20 images/s
after model load.

## Usage

```bash
./fotosort.sh /path                              # dry run, prints picks + writes fotosort_report.csv
./fotosort.sh /path --move                       # move picks into /path/Highlights
./fotosort.sh /path --copy                       # copy instead of move
./fotosort.sh /path --recursive --with-sidecars  # include subfolders, bring RAW/XMP files along
./fotosort.sh /path --max-per-group 3 --min-per-group 1
./fotosort.sh /path --labels my_labels.txt       # own subject list, one label per line
./fotosort.sh /path --blur-ratio 0               # keep soft bursts (e.g. the only leopard of the trip)
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-per-group` / `--min-per-group` | 5 / 2 | picks per subject per day |
| `--gap-seconds` | 120 | time gap that starts a new scene |
| `--scene-sim` | 0.80 | min cosine similarity to stay in a scene |
| `--dup-sim` | 0.95 | similarity above which two picks count as duplicates |
| `--aesthetic-weight` | 0.6 | aesthetics vs sharpness weight in the score |
| `--blur-ratio` / `--blur-floor` | 0.3 / 20 | blur rejection thresholds (relative to day median / absolute) |
| `--skip-labels` | document or screenshot | subjects never selected |
| `--highlights` | Highlights | name of the output subfolder |
| `--report` | fotosort_report.csv | CSV report path, relative to the photo folder |
| `--no-cache` | | ignore and do not write the feature cache |
| `--limit N` | | only process the first N files (for testing) |

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
