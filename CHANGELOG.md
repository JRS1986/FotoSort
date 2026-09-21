# Changelog

All notable changes to FotoSort are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Persist explicit keep/reject choices, pairwise preferences, and acceptable
  alternatives with `fotosort decisions`; retain automatic report provenance,
  protect concurrent edits, and export reviewed CSVs without model calls.
- Verify the content identities of picks before applying new-format reports;
  changed files cannot silently inherit reviewed selections.
- Record `--include` picks and bulk CSV imports separately from automatic picks
  and one-by-one review choices. Warn when a decision's photo has moved or
  left the collection, and validate the review record before analysis starts.

### Documentation
- Add an issue-first contribution workflow, bug/feature and pull-request
  templates, and repository instructions for agent-assisted development
  (`AGENTS.md`, imported by `CLAUDE.md` and shipped in the source distribution).
- Link the prioritized project roadmap and correct the CLI documentation
  location in the contribution guide.
- Rework the README around installation, reviewing selections and explicit file
  operations; highlight experimental status and documented missed favourites.
- Move algorithm details and the complete option reference into linked guides;
  clarify person detection, judge endpoints and model-dependent compatibility.

## [0.1.0] - 2026-09-19

Initial preview release. The entries below describe the work accumulated before
the first published version.

### Release preparation
- Include label examples, evaluation documents, contribution notes and the
  synthetic test generator in source distributions; remove tracked build output.
- Build and validate distributions in CI and smoke-test the installed CLI.
- Load the aesthetic predictor with PyTorch's restricted weights-only loader.
- Correct the unreleased version history, CLI examples and file-operation docs.
- Adopt AGPL-3.0-only for the initial release, retaining the Ultralytics detector.
  Include the full license and copyright notice in source and wheel packages.

### Wildlife camera roll
- Added `--mode award-roll --judge`: a single wildlife portfolio across all dates,
  targeting 12 photos (`--roll-size` accepts 10–15). One photo per species is the
  norm; a second needs a distinct exceptional moment and a concrete explanation.
- Defaults to local preselection with a strict 480-photo global candidate cap
  (`--award-candidates`). Scene counts and per-group share floors cannot expand
  it; full coverage is explicit. Local subject groups share the budget with
  diminishing weight for burst volume; within-day score ranks, independent
  nominations and visual alternatives determine candidates. This policy needs
  real-photo recall evaluation. Final choices still come from visual comparisons.
- Packs small groups into shared initial requests. `--award-plan` lists candidates
  and comparison-work bounds without a judge or API key; it writes a separate
  plan report and reuses the local feature cache. Limits include repeat comparison
  work but exclude retries and extra detail crops; they are not money limits.
- The separate wildlife brief excludes people-led photos, pets and empty scenery,
  and does not force weak picks to fill the count. Judge failures stop the run.
  Reports retain species, rank, visible moment and exceptional-moment evidence.
- Defaults to `AwardRoll/` and `award_roll_report.csv`; recursive scans exclude
  the generated roll. Standard selection defaults remain unchanged. Logic and
  mocked pipeline tests cover caps, coverage, species exceptions and export;
  real-photo curation quality remains to be evaluated.

### Selection quality and reliability
- Added opt-in `--burst-alternatives`: continuous capture grouping, real quality
  allocation, general expression/interaction/gesture nominations and separate
  visual pose alternatives, all within the existing shortlist cap. Burst-aware
  packing does not add initial requests or repeat images. Model weights and
  overall photo scores are unchanged; no species or filename rules are added.
- Cache v8 retains v6/v7 full-image features and supports up to six subject boxes,
  two local CLIP crops and one interaction DINO crop. Additional extraction is
  only requested by the experimental judge-preselection mode and is cached.
- Kept the burst policy off by default after a local replay found mixed results:
  the exact three Botlierskop favourites still missed the shortlist; the
  meerkat hug alternatives entered, but five previous Buffelsdrift judge picks
  were displaced. This is candidate recall evidence, not a new judge verdict.
- Restored `--preselect-share 0.5` as the default after reviewed Botlierskop
  favourites were excluded before judging with share 0. The share floor rounds
  up, so odd-sized groups also send at least half their distinct eligible files.
  Share 0 remains an explicit cheaper option; higher coverage improves the
  opportunity for recall but does not guarantee every favourite survives.
- Reviewed real shoot examples exposed a thumbnail-alignment veto on similar
  reframed photos. Local duplicate matching now accepts strong whole-image
  DINO (default 0.98) plus subject-DINO agreement within the same scene and
  subject bucket. `--dup-reframe-sim 0` disables this route. Measured reviewed
  giraffe/ostrich pairs had whole-image similarities 0.946–0.991 and subject
  similarities 0.916–0.983, but thumbnail correlations only 0.188–0.846.
  An aggressive 0.94 trial dropped two preferred giraffe variants because
  the scorer favoured a weaker pose; 0.98 preserves these alternatives while
  catching the near-identical ostrich pair. Non-identical frames remain eligible
  for the visual judge. This calibrates similarity, not general pose preference.
- Finer moment grouping uses capture gaps, full-image DINO and detected subject
  views. Moment representatives compete within the existing shortlist cap;
  only the existing scene-coverage guarantee can expand it.
- Independent focus and distinctive-moment nominations join aesthetics,
  composition and editorial preferences. Quality-aware visual coverage replaces
  pure farthest-point selection, with up to two existing exploration slots for
  unusual or uncertain frames. Reports explain nomination routes.
- Cached subject DINO views reveal pose changes obscured by the background and
  can prevent false local duplicate matches. No new model is required. Cache v7
  reads v6 data and adds missing subject features without rerunning full-image
  models or detection; `--no-subject-embeddings` disables this local work.
- Adaptive local focus checks examine at most 12 frames per day/subject and
  three close candidates per moment at 512 px subject scale from full sources.
  Comparisons are bounded, cached and reported; failed/tiny regions do not
  replace preview scores. `--focus-refine-per-group 0` disables refinement.
- New moment and exploration controls document starting heuristics. Synthetic
  tests cover missed action, poor visual outliers, cache upgrades, missing
  detections, independent nominations and processing/upload bounds. A local
  replay of three reviewed Botlierskop favourites reproduced their exclusion
  at share 0 and still excluded them at 0.5; broader accuracy is not established.
- Added gentle, alternative thirds/golden-ratio/central-placement bonuses and
  local CLIP editorial contrasts for light, composition, subject clarity and
  compelling moments. The existing learned aesthetic predictor remains separate.
- Aesthetic, editorial and composition candidates can nominate frames within
  the existing shortlist allocation after scene coverage; no extra upload slots.
- New `--composition-weight` (0.2) and `--editorial-weight` (0.15) controls, with
  zero disabling each preference. Geometry never rejects unconventional frames.
  Reports expose all measurements and their bounded score contributions.
- These preferences reuse cached boxes and embeddings; no cache rebuild or new
  model download is required. Award-related criteria are heuristics inspired by
  judging guidance, not a classifier trained on verified award winners.
- Judge coverage defaults to local preselection; full coverage remains opt-in.
  Shortlists scale with keeper budgets and scene coverage, with a default
  half-group floor (`--preselect-share 0.5`). Difficult light and
  low-texture candidates are not automatically rejected before shortlisting.
- Each tournament batch can advance the full keeper budget; winner pools are
  combined into fewer comparisons while preserving strong batches. Cross-label
  winners receive duplicate review.
- Preselection reserves every scene and allows low-scoring distinctive moments.
  Only SHA-256-identical sources collapse before judging; local near-duplicate
  checks require agreement between DINOv2 and thumbnail framing.
- Subject-region focus contributes to ranking, cached alongside the detector
  box. Scores are normalized per day and subject. The default relative score
  cutoff is disabled; explicit `--min-score` still works without the judge.
- Default judge payloads contain a 1024 px full image and one native subject
  detail. `--judge-crops multi` enables extra detail windows; `none` omits crops.
  Images use the shared RAW/JPEG loader and cached encodings. Prompts consider
  exceptional behaviour, expression, light and storytelling.
- The judge uses the processed DxO source; RAW+JPEG pairs and external DxO
  output folders work consistently. RAWs lacking previews fall back to decoding.
- Cache loading retains each feature array once, with atomic saves and
  nanosecond file timestamps. Format v6 requires one feature re-analysis.
- RAW keep decisions take precedence across representations. Recursive scans
  recognize RAW-subfolder pairs and exclude generated output directories.
- Reports record relative paths, decode sources, subject focus and decisions.
  Applying moved reports preserves subfolder identity and refuses ambiguous
  filename matches before file operations.
- Added deterministic regressions for the review findings. Real-photo accuracy
  and model-specific behaviour still require evaluation on reviewed shoots.

### Added
- **Judge names the species of every pick** (`--judge-species`, on by default
  with `--judge`). Species folders follow the judge's answer instead of the
  CLIP label; a `judge_label` column records it and `--apply-report --judge`
  names picks of an existing report. Zero-shot CLIP confuses seal flippers,
  dolphin fins and spray at sea; the judge sees the frame at full detail.
- **Detector person override needs CLIP agreement** (`--person-agree 0.2`).
  Swimming seals trigger the detector's person class; a person filling half
  the frame is still trusted without agreement.
- `labels/shark_diving.txt`: marine label set for shark-cage and boat trips
  (seals on rocks and in the water, fish and chum, shark fins and teeth).
- Progress bars for the judge stage (per day and group) and for the local
  focus refinement.
- **Edit benefit and preset per frame.** `edit_score` (0..100) from blown
  highlights, crushed shadows, exposure and ISO; judged picks get the judge's
  `edit_benefit` (low/medium/high) with a reason and a preset from a fixed
  menu (Natural, Color Pop, Wildlife Crisp, Golden Hour Warmth, Moody Colors,
  Low Key Dramatic, High Key, Black & White, Vignette, Portrait Soft). Both
  in the report and as XMP keywords.
- **RAW cull** (`--raw-cull`, `--raw-keep-benefit`, `--raw-cull-move`):
  keep the RAWs of picks worth editing, list or move the rest into
  `_DELETE_ME_raw`. RAW siblings are found next to the JPEG or in `RAW/`.
- ISO is read from EXIF (JPEG and RAW) and reported.
- `--include`: frames that must be picked regardless of scores or judge.
- Preselection reserves a representative of each scene, including weaker
  scored bursts of a long sighting, in addition to the configured share floor.
- Diverse preselection, judge retries and a duplicate-free fallback (see
  the commit "Preselection covers a sighting's variety").
- **DxO PureRAW as a pre-processing step** (`--dxo all|picks`). Processed
  twins in the `DxO` folder are found under any PureRAW naming template;
  missing ones are handed to the app (`open -a`) and awaited. The twin is
  scored, enhanced and copied; the RAW keeps its identity.
- **Output layout** (`--layout species|day|day-species`): one subfolder per
  subject and/or day inside the output folder.
- **Preselection before the judge** is now the default
  (`--judge-coverage preselect`, `--preselect 3`, `--preselect-min 12`,
  `--preselect-share 0.5`): scores and visual coverage narrow each group to
  at least three times its budget and half its distinct eligible frames,
  retaining every scene. The judge tournament runs on those; actual token
  savings depend on the groups and repeated winner comparisons. `full`
  remains available. The old
  `shortlist` mode and `--shortlist-max` are gone.
- **RAW input.** ORF, NEF, CR2, CR3, ARW, RAF, RW2, DNG, PEF and SRW files
  are analysed through their embedded camera preview; a RAW next to a JPEG
  of the same name is skipped. Capture time is read from the RAW's EXIF.
  Enhanced output of a RAW pick is a full-resolution JPEG (camera white
  balance, no auto-brightening) carrying camera model and capture time.
  `--no-raw` ignores RAW files.
- **Local judge.** `--judge-base-url` points the judge at any
  OpenAI-compatible server (Ollama, LM Studio, vLLM, MLX); no key, no upload.
  Falls back automatically when the server rejects JSON mode.
- **XMP sidecars.** `--xmp picks|all` writes `<stem>.xmp` next to the
  originals with rating, colour label, keywords and the judge's reason, for
  Lightroom (RAW), Capture One, Bridge and digiKam. Existing sidecars are
  never overwritten unless `--xmp-overwrite`.

### Changed
- **Near-duplicate detection uses DINOv2** (small, 224 px) instead of CLIP.
  On a calibrated safari set burst twins score above 0.90 and different
  compositions of the same subject below 0.88, where CLIP overlapped.
  `--dup-sim` now refers to DINOv2 similarity (default 0.89). The thumbnail
  correlation still catches pixel-identical frames. Cache format v5.
- New dependencies: timm, rawpy, exifread.

### Initial implementation — 2026-09-04

Developed and calibrated on a 2,000-frame safari folder and a 2,700-frame
whale-watching burst. This development snapshot was not separately released.

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
- Packaging (`pip install -e ".[judge]"`, `fotosort` console script), the
  development snapshot's MIT license (superseded by AGPL-3.0-only for this
  release), GitHub Actions CI with ruff and pytest.

[Unreleased]: https://github.com/JRS1986/FotoSort/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JRS1986/FotoSort/releases/tag/v0.1.0
