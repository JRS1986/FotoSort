# FotoSort

[![CI](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml/badge.svg)](https://github.com/JRS1986/FotoSort/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

A local-first command-line assistant for making a **reviewable first pass through
large photo collections**. FotoSort suggests varied highlights from JPEGs and
RAW files, records its decisions in a CSV, and lets you review or change the picks
before exporting them.

Built around wildlife and travel shoots: long bursts, similar frames, and several
subjects in a day. Local models assess sharpness, exposure, aesthetics, subjects,
and similarity. An optional vision-model judge can compare shortlisted photos.

**v0.1.0 is an experimental preview.** It can miss your favourite frame or prefer
a weaker expression. Review the results before discarding anything; the
[known limitations](#known-limitations) include real examples of missed favourites.

[Download v0.1.0](https://github.com/JRS1986/FotoSort/releases/tag/v0.1.0) ·
[How selection works](docs/how-it-works.md) ·
[All CLI options](docs/cli-reference.md)

## Quick start

Requires Python 3.11+. CI covers Python 3.11 and 3.12 on Linux; the local-model
pipeline has also been exercised on Apple silicon. The first analysis downloads
roughly 1.1 GB of model weights, in addition to Python dependencies. A GPU helps;
CPU execution is supported but can be slow.

```bash
git clone https://github.com/JRS1986/FotoSort.git
cd FotoSort
python3.11 -m venv .venv
source .venv/bin/activate
pip install .

# Analyse a small sample first. Writes a report and cache; exports no photos.
fotosort /path/to/photos --limit 100

# Analyse the whole folder, then review fotosort_report.csv.
fotosort /path/to/photos

# After reviewing, copy the selected photos into Highlights/.
fotosort /path/to/photos --apply-report --copy
```

Set `selected` to `1` or `0` in the CSV to override a choice. Keep the file/path
columns intact. `--apply-report` uses those selections without repeating the
selection process. Use the same `--recursive` setting if the original run
included subfolders.

For choices that survive analysis reruns, use
[`fotosort decisions`](docs/review.md). It preserves the automated recommendation
separately and exports reviewed selections for the same `--apply-report` workflow.

Run `fotosort review /path/to/photos` to compare picks with excluded burst
neighbours in a local browser. Keep/reject shortcuts, synchronized detail views,
undo, and reviewed CSV export use the same [persistent decisions](docs/review.md).

You can also extract the release's source archive and run the installation
commands from that directory, or install its wheel with `pip install /path/to/file.whl`.
For judge support, install `pip install ".[judge]"` from the source directory.
`./fotosort.sh /path/to/photos` creates a virtual environment and installs with
judge support automatically on first use.

## What gets written

A normal run writes `fotosort_report.csv` and `.fotosort_cache.npz` in the photo
folder. It uses local models and needs no judge API key. Initial model downloads
require an internet connection; cached features speed up subsequent runs.

Photo contents are preserved. File operations require explicit flags:

| Action | Flag |
|---|---|
| Copy picks to `Highlights/` | `--copy` |
| Create enhanced JPEG copies of picks | `--copy --enhance` |
| Relocate selected originals | `--move` |
| Write rating/keyword sidecars next to originals | `--xmp picks` or `--xmp all` |
| Replace existing XMP sidecars | `--xmp-overwrite` with `--xmp` |
| List RAWs suggested for retention or culling | `--raw-cull` |
| Relocate suggested RAW rejects to `_DELETE_ME_raw/` | `--raw-cull --raw-cull-move` |

FotoSort does not delete photos. RAW culling follows the selected frames and
edit-benefit estimates, so review those choices before removing any files yourself.

## Common workflows

```bash
# Include subfolders and group copied picks by subject.
fotosort /path/to/trip --recursive --copy --layout species

# Export enhanced versions of an already reviewed selection.
fotosort /path/to/photos --apply-report --copy --enhance

# Write XMP ratings and keywords for RAW-aware photo editors.
fotosort /path/to/photos --xmp picks

# Use a marine subject list from the checkout.
fotosort /path/to/photos --labels labels/whale_watching.txt --label-mode image

# Force known favourites into a standard-mode selection.
fotosort /path/to/photos --include IMG_0042,IMG_0108

# Enhance a folder using image statistics for style detection.
fotosort enhance /path/to/photos --no-clip
```

Standard mode groups photos by day and subject. The default budget starts at
five picks per group, increases for larger groups, and is capped at fifteen.
`--max-per-group` and `--extra-per` control it. `--layout day`, `species`, or
`day-species` organizes exported files; the default output is flat.

RAW inputs are analysed using embedded previews, with decoding as a fallback.
A same-stem JPEG represents a RAW+JPEG pair. `--with-sidecars` carries matching
companions during copying or moving. Generated output folders are excluded from
recursive scans. Supported extensions and pairing rules are described in
[RAW input](docs/how-it-works.md#raw-input); camera-specific compatibility varies.

On macOS, `--dxo picks --copy --enhance` can hand RAWs to an installed DxO PureRAW
app and wait for you to process them. `--dxo-wait 0` uses existing outputs without
launching the app. See [DxO integration](docs/how-it-works.md#dxo-pureraw).

XMP files carry ratings, labels, keywords, and available judge reasons. Existing
sidecars are preserved unless explicitly overwritten. Editor support differs
between RAW and JPEG files; check how your editor imports sidecars.

## Optional vision-model judge

Install the `judge` extra first. To use a cloud provider, configure its API key
and an image-capable model available to your account:

```bash
export OPENAI_API_KEY="your-api-key"
fotosort /path/to/photos --judge --judge-model YOUR_VISION_MODEL

# Or use Anthropic with its own model ID and key.
export ANTHROPIC_API_KEY="your-api-key"
fotosort /path/to/photos --judge --judge-provider anthropic --judge-model YOUR_VISION_MODEL
```

Replace `YOUR_VISION_MODEL` with your provider's actual model ID. The configured
model defaults are listed in the [CLI reference](docs/cli-reference.md); access
and API compatibility depend on the provider and account.

**Cloud judging sends image overviews and subject details to the selected API
and can incur charges.** By default, standard mode shortlists at least half of
each eligible day/subject group; group size, keeper budgets, and scene coverage
can make it larger. Final comparisons and subject naming add requests. Token
usage is reported after judging; candidate limits are not spending limits.

`--judge-coverage full` includes every distinct eligible file, increasing coverage
and usually cost. `--judge-detail low` reduces overview resolution, and
`--judge-crops none` omits subject detail crops. Full coverage can still miss a
preferred final choice.

For a local server that supports OpenAI-compatible image requests:

```bash
fotosort /path/to/photos --judge \
    --judge-base-url http://localhost:11434/v1 --judge-model YOUR_LOCAL_VISION_MODEL \
    --judge-detail low --judge-chunk 6
```

Start your server and load an image-capable model first. No provider API key is
required for a local server without authentication. Images go to the configured
endpoint; a remote URL sends them off your machine. Compatibility, context limits,
speed, and judgment quality vary by model and server.

## Wildlife portfolio mode

`--mode award-roll` asks the judge to curate one portfolio across all dates,
targeting 12 wildlife photos with limited repetition per species. Its name
expresses an editorial ambition; it does not predict awards.

```bash
# Inspect up to 480 locally shortlisted candidates without judge requests.
fotosort /path/to/trip --recursive --mode award-roll --award-plan

# With the judge configured, generate the portfolio report for review.
fotosort /path/to/trip --recursive --mode award-roll --judge --judge-model YOUR_VISION_MODEL

# Export the reviewed portfolio.
fotosort /path/to/trip --recursive --mode award-roll --apply-report --copy
```

The normal report is `award_roll_report.csv`, and exports go to `AwardRoll/`.
`--roll-size` accepts targets from 10 to 15. The judge may select fewer, including
zero. Judge failure stops this mode; it does not fill the portfolio with local
score fallbacks. See the [selection details](docs/how-it-works.md#wildlife-portfolio---mode-award-roll---judge).

## Evaluate reviewed selections

Create a private feedback manifest and evaluate saved results without model calls:

```bash
fotosort evaluate feedback /path/to/photos --output fotosort_evaluation.json --split diagnostic
fotosort evaluate run fotosort_evaluation.json --output summary.json
```

The [evaluation guide](docs/evaluation.md) defines exact-favourite and acceptable-moment
recall, missing-data handling, measured review effort, and frozen baseline comparisons.

## Known limitations

- **Preferred moments can be missed.** In a diagnostic replay, all three
  photographer-selected favourites missed the default 50% shortlist. Read the
  [preselection evaluation](docs/evaluations/botlierskop-preselection.md).
- **The burst experiment has mixed results.** `--burst-alternatives` is off by
  default. It recovered some alternatives while displacing other previous picks;
  the [evaluation](docs/evaluations/burst-alternatives.md) does not establish a
  general improvement.
- **Scores and labels are fallible.** Sharpness measures respond to texture,
  detectors can confuse animals with people, and similarity does not establish
  the best expression. There is no dedicated eye-focus or expression detector.
- **Validation is limited.** Automated tests check logic and resource bounds.
  A synthetic local-model smoke test exercises the pipeline, not photographic
  taste. Live cloud judging and camera-specific RAW/DxO behavior were not
  revalidated for the initial release. Broader accuracy across genres is unknown.

The CSV exposes scores, labels, nominations, judge exposure, and decision reasons
so you can inspect where a frame was excluded. Reports of missed favourites are
especially useful: include the relevant rows and, if you can share them, downsized
examples. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
pip install -e ".[judge,dev]"
ruff check fotosort tests
pytest
python tests/make_testset.py /tmp/fotosort-testset
fotosort /tmp/fotosort-testset
```

Tests use model stand-ins and need no weights or GPU. The synthetic pipeline run
uses real local models. CI also builds and validates source/wheel packages and
checks the installed CLI outside the checkout.

[Algorithm and report details](docs/how-it-works.md) ·
[CLI reference](docs/cli-reference.md) ·
[Changelog](CHANGELOG.md) ·
[Issues](https://github.com/JRS1986/FotoSort/issues)

## Credits and license

Built with [OpenAI CLIP](https://github.com/openai/CLIP) through
[OpenCLIP](https://github.com/mlfoundations/open_clip), the
[LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor),
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
[DINOv2](https://github.com/facebookresearch/dinov2) through
[timm](https://github.com/huggingface/pytorch-image-models),
[rawpy](https://github.com/letmaik/rawpy), and
[ExifRead](https://github.com/ianare/exif-py).

Copyright (c) 2026 Jan R. Seyler. FotoSort is licensed under the GNU Affero General
Public License, version 3 only (`AGPL-3.0-only`). See [LICENSE](LICENSE) and
[NOTICE](NOTICE). Dependencies and downloaded model weights retain their own
licenses; the required Ultralytics package and YOLO weights use AGPL-3.0 by default.
