# Contributing to FotoSort

Thanks for taking an interest. FotoSort is a small, opinionated tool, and
contributions that keep it small and opinionated are the most welcome ones.

## Getting set up

```bash
git clone https://github.com/JRS1986/FotoSort.git
cd FotoSort
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[judge,dev]"
pytest            # unit tests, no model weights needed
ruff check fotosort tests
```

The first real run downloads about 1 GB of model weights into the Hugging Face
cache and `~/.cache/fotosort`. `python tests/make_testset.py /tmp/testset`
creates a small synthetic folder with EXIF dates, bursts, a blurred burst and
clipped frames, which is enough to exercise the whole pipeline end to end:

```bash
fotosort /tmp/testset
fotosort /tmp/testset --copy --enhance
```

## What makes a good change

- **Keep the pipeline explainable.** Every decision FotoSort makes should be
  visible in the CSV report. If you add a signal, add its column.
- **Preserve photo contents.** Photo writes must go to a new destination.
  Only explicit `--move` or `--raw-cull-move` operations relocate originals;
  replacing existing XMP sidecars requires `--xmp-overwrite`.
- **Calibrate on real photos, then write the number down.** Thresholds such as
  the duplicate similarity or the blur floor were measured against real frames
  a photographer called identical or soft. If you change one, say in the commit
  message what you measured.
- **Prefer a flag to a fork.** Shoot-specific behaviour belongs in label files
  (`labels/`), `--judge-hint`, or a recipe in `enhance.py`, not in a copy of the
  pipeline.
- **Tests for logic, not for models.** Pure logic (selection, grouping, budget,
  enhancement maths, report parsing, judge verdict parsing) gets a unit test.
  Model behaviour is checked by running the tool on photos, not in CI.

## Reporting a bad pick

The most useful bug report for a culling tool is "it picked X and it should
have picked Y". Please include the relevant rows of `fotosort_report.csv`
(they carry the scores, group, whether the judge saw the frame, and its reason)
and, if you can share them, downsized copies of the frames. Do not commit
photos to the repository.

## Pull requests

- Start from an existing issue, or open one using the bug/feature templates.
  Define the problem, scope, acceptance criteria, dependencies and validation
  before implementation. Search existing issues and PRs to avoid duplicates.
- Use a focused topic branch (for example `fix/<topic>` or `docs/<topic>`)
  from the current default branch, in your fork or in this repository. Keep
  unrelated changes separate and preserve any existing local work.
- One topic per pull request, with a short description of what changed and
  why. Reference the report rows or measurements that motivated it.
- Link the issue in the PR description. Use `Closes #N` when all of its
  acceptance criteria are met, or `Refs #N` for a partial implementation.
  Open incomplete work as a draft and list what remains.
- `ruff check` and `pytest` must pass; CI runs both on Python 3.11 and 3.12.
- CI also builds the source distribution and wheel, validates their metadata,
  and checks the installed CLI outside the source tree. Run `python -m build`
  and `python -m twine check --strict dist/*` locally before a release.
- Add an entry under **Unreleased** in `CHANGELOG.md`.
- New command-line flags need a row in [the CLI reference](docs/cli-reference.md)
  with the actual default. Update README workflow examples where useful.
- Review the final diff and include actual validation results and material
  limitations in the PR. Documentation/template changes need content and link
  checks; do not add tests that only repeat the documentation.
- Keep PRs open for maintainer review and merge only after authorization and
  passing checks. Do not push changes directly to `main`. Close issues only
  when their acceptance criteria are satisfied and the relevant PRs have merged.

Agent-assisted work follows the same process; [AGENTS.md](AGENTS.md) adds the
agent-specific rules, including the `codex/<topic>` branch prefix.

## Ideas that are welcome

Useful contributions include broader real-photo evaluations, RAW camera
compatibility reports, and small fixes backed by reproducible examples.
Local judging, RAW input, XMP export, and DINOv2 duplicate detection already
exist; see `CHANGELOG.md` for implemented changes and the issue tracker for
current proposals.

The [project roadmap](https://github.com/JRS1986/FotoSort/issues/6) tracks the
planned improvements, their current priorities and their dependencies. Each
improvement has its own scoped issue; future proposals should follow the same
process.

## License

By contributing you agree that your contribution is licensed under the GNU
Affero General Public License, version 3 only (`AGPL-3.0-only`), like the rest
of the project. See [LICENSE](LICENSE).
