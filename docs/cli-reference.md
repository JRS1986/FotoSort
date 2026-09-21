# Command-line reference

[Back to the README](../README.md) · [How selection works](how-it-works.md)

Run `fotosort --help` for the installed version's options. The defaults below
match v0.1.0. Provider model access depends on your account or server.

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
| `--enhance` | | write enhanced JPEG copies of picks |
| `--enhance-strength` / `--enhance-style` | 1.0 / auto | global strength; force one style |
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
| `set FILE... --choice keep\|reject\|clear` | choice required | Set or clear explicit overrides using relative paths |
| `set` / `prefer --note TEXT` | empty | Record a human explanation |
| `prefer WINNER LOSER` | | Record a pairwise preference without changing selection |
| `alternatives NAME FILE...` | | Record acceptable alternatives for a moment |
| `undo` | last edit | Restore previous decisions and feedback, up to 50 edits |
| `show` | | Print the current review record without undo history |
| `import` | all report rows | Explicitly convert this report's selected column into keep/reject choices |
| `export --output FILE` | fotosort_reviewed.csv | Export effective selections to a CSV inside the photo folder |

## Standalone enhancement

Run `fotosort enhance --help` for its separate options:

```bash
fotosort enhance /path/to/photos --no-clip
fotosort enhance IMG_0042.jpg --style landscape --strength 0.8
```

This writes JPEGs to an `Enhanced/` subfolder by default; `--out` chooses
another destination. Existing enhanced files with the same name can be
replaced on subsequent runs. `--no-clip` uses image statistics for style
detection, and `--style` chooses a recipe explicitly.


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
| `run MANIFEST --run` | `baseline` | Evaluate a named saved run in every shoot |
| `compare MANIFEST --run` | `candidate` | Candidate run to compare |
| `compare --baseline` | `baseline` | Frozen baseline with the same eligible inputs and budget |
| `run/compare --output` | stdout | Shareable JSON summary; cannot overwrite inputs |
