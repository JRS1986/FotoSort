# Command-line reference

[Back to the README](../README.md) · [How selection works](how-it-works.md)

Run `fotosort --help` for the installed version's options. The defaults below
match v0.2.0. Provider model access depends on your account or server.

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
| `--dup-sim` | 0.89 | DINOv2 threshold for local near-duplicates; framing or the reframe rule must also agree |
| `--dup-pixel` | 0.90 | thumbnail framing threshold used together with DINOv2 |
| `--dup-reframe-sim` | 0.98 | strong whole-image DINO match allowing reframed local duplicates within a scene when subject DINO also agrees; 0 disables |
| `--min-score` | disabled | optional relative score cutoff for selection without the judge |
| `--aesthetic-weight` | 0.6 | learned aesthetics vs sharpness weight in the score |
| `--composition-weight` | 0.2 | maximum bonus for thirds, golden-ratio or central placement; 0 disables |
| `--editorial-weight` | 0.15 | maximum local CLIP preference bonus for light, composition, subject and moment; 0 disables |
| `--subject-weight` | 0.4 | bonus for a large detected subject, penalty for a small cut-off one |
| `--blur-ratio` / `--blur-floor` | 0.15 / 20 | local blur rejection thresholds (relative to a known scene / absolute); the judge assesses blur visually |
| `--person-area` | 0.02 | min person box fraction for the people group, subject to CLIP agreement (0 disables the detector) |
| `--person-agree` | 0.2 | required CLIP probability mass on people labels for the person override; a person covering half the frame bypasses this check |
| `--detector` | yolov8m.pt | YOLOv8 weights; smaller variants trade detection quality for speed |
| `--judge` | off | let a vision model choose the final picks |
| `--judge-provider` / `--judge-model` | openai / gpt-5.6-sol | configured API/model defaults (`anthropic` / `claude-opus-5`); override with an image-capable model available to your account |
| `--judge-species` / `--no-judge-species` | on | with the standard judge, name picked subjects for species folders; adds requests (about one per 12 picks) |
| `--judge-coverage` / `--judge-chunk` | preselect / 24 | `preselect`: local shortlist; `full`: every distinct eligible file, bypassing the award candidate cap. Standard winner comparisons use at most the larger of chunk size and twice the keeper budget; award-roll uses four times the roll target |
| `--preselect` / `--preselect-min` / `--preselect-share` | 3 / 12 / 0.5 | standard-mode shortlist: keeper-budget multiple, minimum and group-share floor. Scene coverage may expand it; award-roll uses its own strict global cap |
| `--preselect-explore` | 2 | maximum existing shortlist slots for unusual/uncertain alternatives; 0 disables |
| `--include` | | force picks in standard mode (stems, names, or `@file`); unavailable for award-roll judging |
| `--judge-detail` | high | OpenAI full-image detail: high uses a 1024 px overview, low a 512 px overview |
| `--judge-crops` | single | `single`: one native subject detail; `multi`: overview plus several native details; `none`: full image only |
| `--judge-hint` | | a sentence or two about this shoot for the judge |
| `--key-file` | | read the judge API key from a `.env` or YAML file |
| `--judge-base-url` | | OpenAI-compatible endpoint with image support (needs `--judge-model`); images go to this address |
| `--xmp` / `--xmp-overwrite` | off | write XMP sidecars for `picks` or `all`; replace existing ones |
| `--raw-cull` / `--raw-keep-benefit` / `--raw-cull-move` | off / low / off | list RAWs to keep vs cull; keep threshold; move the culled ones to `_DELETE_ME_raw` |
| `--taste-dir` / `--taste-weight` | off / 0.5 | folder of photos you love; frames resembling them get up to this bonus |
| `--enhance` | off | write enhanced JPEG copies of picks, with local RightWayUp straightening and corner cropping |
| `--enhance-strength` / `--enhance-style` | 1.0 / auto | global strength; force one style |
| `--no-orientation` | off | disable RightWayUp during enhancement; has no effect on selection |
| `--orientation-tier` | max | RightWayUp model: `pico`, `nano`, `fast`, `balanced`, `pro`, or `max`; uses strict abstention |
| `--orientation-snap` | 0 | `0`: correct any angle and crop empty corners; `90`: quarter turns without cropping |
| `--orientation-min-angle` | 1.0 | treat residual tilts below this many degrees as level (the `rightwayup fix` default); `0` applies every estimate |
| `--apply-report` | | skip analysis; act on the `selected=1` rows of the existing report |
| `--report` | fotosort_report.csv (award_roll_report.csv in award-roll mode) | CSV report path, relative to the photo folder |
| `--no-cache` | | ignore and do not write the feature cache |
| `--limit N` | | only process the first N files |
| `--batch-size` / `--workers` | 32 / 4 | GPU batch size and decoder threads |

## Visual review

`fotosort review FOLDER` opens the existing report in a local browser reviewer.
See [the review guide](review.md) for controls, exports, and local access rules.

| Flag | Default | Meaning |
|---|---|---|
| `--report` | fotosort_report.csv | Existing CSV inside the collection |
| `--port` | 0 | Choose an available loopback port; set a number to use a fixed port |
| `--no-browser` | off | Print the session URL without opening a browser |

## Persistent decisions

`fotosort decisions FOLDER [--report fotosort_report.csv] COMMAND` edits local
human choices without models. See [review decisions](review.md) for the format
and identity rules.

| Command / option | Default | Meaning |
|---|---|---|
| `--report FILE` | fotosort_report.csv | Report inside the photo folder to review |
| `set FILE... --choice keep\|reject\|clear` | choice required | Set or clear explicit overrides using relative paths |
| `set` / `prefer --note TEXT` | empty | Record a human explanation |
| `prefer WINNER LOSER` | | Record a pairwise preference without changing selection |
| `alternatives NAME FILE...` | | Record acceptable alternatives for a moment |
| `undo` | last edit | Restore previous decisions and feedback, up to 50 edits |
| `show` | | Print the current review record without undo history |
| `import` | all report rows | Explicitly convert this report's selected column into keep/reject choices, recorded with the origin `csv import` |
| `export --output FILE` | fotosort_reviewed.csv | Export effective selections to a CSV inside the photo folder; never the report being reviewed |

## Standalone enhancement

Run `fotosort enhance --help` for its separate options:

```bash
fotosort enhance /path/to/photos --no-clip
fotosort enhance IMG_0042.jpg --style landscape --strength 0.8
```

This writes JPEGs to an `Enhanced/` subfolder by default; `--out` chooses
another destination. Filename collisions get numeric suffixes, preserving
existing photos and enhanced copies. `--no-clip` uses image statistics for style
detection, and `--style` chooses a recipe explicitly.

The four orientation flags in the table above also apply to `fotosort enhance`,
with the same defaults. For example:

```bash
# Correct tilted horizons, crop empty corners, and enhance the colour/tone.
fotosort enhance /path/to/photos --no-clip

# Fix sideways/upside-down images without cropping the frame.
fotosort /path/to/photos --apply-report --copy --enhance --orientation-snap 90

# Use only the existing tone/colour recipes, without downloading either model.
fotosort enhance /path/to/photos --no-clip --no-orientation
```

RightWayUp runs locally using the Max tier and its upstream **strict** abstention
thresholds. An abstention skips model rotation, but EXIF orientation and the
tone/colour recipe are still applied. The saved EXIF orientation is normalized so
viewers do not rotate the pixels twice; other EXIF data and RGB ICC profiles are
retained. EXIF image dimensions are updated to match the output. RAW and DxO
inputs use the same decoded-image path and produce JPEGs.

Model estimates are continuous, so a level photo rarely scores exactly 0°.
Residual tilts below `--orientation-min-angle` (1.0° by default, matching
`rightwayup fix --min-angle`) are treated as model error: a nearly upright photo
is left unrotated, and a nearly sideways one receives a lossless quarter turn.
The threshold follows upstream and has not been calibrated on reviewed photos.

Continuous correction crops a centered rectangle with the aspect ratio of the
nearest upright quarter turn. It trims edges to exclude empty corners, with a
small allowance for bicubic interpolation. It can remove substantial content
on heavily tilted images. Quarter turns require no crop. Model predictions can
be wrong, including confident ones; review enhanced copies before using them.

`--no-orientation` keeps the previous tone/colour-only behavior. Zero strength
(`--strength 0` or `--enhance-strength 0`) skips both the recipe and RightWayUp;
JPEG output is still re-encoded. Ordinary analysis, dry runs, and exports without
`--enhance` never initialize RightWayUp or download its weights. Model setup
errors stop enhanced export before copying/moving picks and give an actionable
message; they do not silently produce uncorrected images.

RightWayUp is included in the regular installation. Its ONNX weights download
from `ortusai/rightwayup` on Hugging Face on first use, in addition to FotoSort's
analysis models (about 300 MB for Max INT8). Upstream selects INT8 on CPU and FP16 when ONNX Runtime provides
CUDA. Choose `--orientation-tier fast` for lower resource use. To run offline,
cache the model first or set `RIGHTWAYUP_MODEL_DIR` to a folder containing that
tier's ONNX files (for example `max-l280-int8.onnx` for Max on CPU). Photo pixels
are never sent to a service for this operation.

Each export writes a new `fotosort_enhancement.csv` in its output root (or each
standalone output folder), adding a numeric suffix on later runs. The selection
report and its review identities are unchanged. Each row records source/output
paths, style, strength, status (`corrected`, `upright`, `abstained`, `disabled`, or
`zero_strength`), estimated clockwise angle, confidence, abstention, actual
counter-clockwise correction, tier, cascade routing, precision, device, snap
setting, minimum angle, fraction of source area cropped, and output dimensions. Disabled cases
have no estimate/confidence. Successful rows are flushed as each output is saved,
so a later failure does not erase earlier records. These local reports contain
photo paths; sanitize them before publishing.


## Evaluation harness

`fotosort evaluate` runs offline on saved CSV reports. See the [format and metric definitions](evaluation.md).

| Command / option | Default | Purpose |
|---|---|---|
| `feedback FOLDER` | — | Export explicit review feedback to a private manifest |
| `feedback --report` | `fotosort_report.csv` | Report inside the collection |
| `feedback --output` | Required | New private JSON manifest; existing files are refused |
| `feedback --split` | `diagnostic` | `development`, `heldout`, or `diagnostic` |
| `feedback --review-seconds` | Unknown | Explicit measured baseline review time |
| `feedback --manual-replacements` | Unknown | Explicit observed baseline replacements |
| `feedback --include-imported` | off | Also export decisions recorded by `decisions import` as favourites and rejects |
| `run MANIFEST --run` | `baseline` | Evaluate a named saved run in every shoot |
| `compare MANIFEST --run` | `candidate` | Candidate run to compare |
| `compare --baseline` | `baseline` | Frozen baseline with the same eligible inputs and budget |
| `run/compare --output` | stdout | Shareable JSON summary; cannot overwrite inputs |
