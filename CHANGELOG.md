# Changelog

All notable changes to FotoSort are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Planned, in rough order of value: a `--judge-base-url` option so the judge can
run against a local OpenAI-compatible vision model (Ollama, LM Studio, MLX),
RAW input via the embedded preview, XMP sidecar export of picks and reasons for
Lightroom and Capture One, DINOv2 embeddings for near-duplicate detection.

## [0.1.0] - 2026-09-04

First public release, developed and calibrated on a 2,000-frame safari folder
and a 2,700-frame whale-watching burst.

### Added
- Scoring of every JPEG: Laplacian sharpness (max over a tile grid), exposure
  clipping, CLIP ViT-L/14 embedding with the LAION aesthetic predictor,
  zero-shot subject labels, and a YOLOv8 person/animal detector for subject
  size, frame-edge contact and the "people" group.
- Scene (burst) clustering by time gap and embedding similarity; one label per
  scene by default, `--label-mode image` for boats and other static backdrops.
- Day-wide, score-based selection with a per-group budget that grows with the
  group size, a score floor, and near-duplicate rejection by CLIP similarity
  and thumbnail correlation across all groups of a day.
- Optional vision-model judge (`--judge`) via OpenAI (default `gpt-5.6-sol`)
  or Anthropic (`claude-opus-5`): every distinct frame is judged in chunks with
  knock-out rounds and a final, each frame sent as the whole picture plus a
  native-resolution crop of the detected subject, a second look when a group
  is rejected entirely, a written reason per pick, and `--judge-hint` for
  shoot-specific guidance.
- Style-aware enhancement (`--enhance`, `fotosort enhance`): style detection
  (wildlife, landscape, golden hour, portrait, night, mono, food, urban, water,
  macro), recipes scaled by the image's own statistics, portrait recipe for
  detected people, EXIF and ICC preserved, originals never modified.
- CSV report with every score, label, group, judge exposure and reason;
  `--apply-report` to copy, move or enhance a hand-edited report.
- Per-folder feature cache, so re-runs with new settings take seconds.
- `--copy` / `--move` / dry run, `--with-sidecars` for RAW/XMP companions,
  `--recursive`, custom label lists (`labels/whale_watching.txt` included),
  `--taste-dir` bonus for frames resembling your own favourites.
- Robustness: unreadable files and files deleted mid-run are skipped, the
  judge fails fast without a key and aborts after repeated failures, the
  cache is invalidated on a version or detector change.
- Packaging (`pip install -e ".[judge]"`, `fotosort` console script), MIT
  license, GitHub Actions CI with ruff and pytest.

[Unreleased]: https://github.com/JRS1986/FotoSort/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JRS1986/FotoSort/releases/tag/v0.1.0
