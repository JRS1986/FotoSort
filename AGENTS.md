# Working on FotoSort

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes. The project owner
requires development through GitHub issues and pull requests.

## Issue and branch workflow

- Check existing issues and PRs before starting work. Create or reuse an issue
  that states the problem, scope, acceptance criteria, and validation approach.
- Work on a focused `codex/<topic>` branch from the current default branch.
  Keep unrelated work separate and preserve existing user changes.
- Deliver changes through a PR linked to the issue. Use `Closes #N` only when
  the PR satisfies the whole issue; use `Refs #N` for an implementation slice.
- Keep PRs open for maintainer review. Merge only when the user has authorized
  it and the required checks pass. Do not push changes directly to `main`.
- Close issues when their acceptance criteria have been met and the relevant
  PRs have merged. Track remaining work explicitly rather than declaring it done.

## Validation and review

- Run `ruff check fotosort tests` and `pytest` for changes, and report the actual
  results. CI also validates distributions and the installed CLI on Python
  3.11 and 3.12; resolve relevant failures before marking a PR ready.
- Add focused regression tests for behavior changes. Documentation and template
  changes need content/link checks, not tests that merely repeat their text.
- Inspect the final diff. Keep PR descriptions focused on the problem, resulting
  behavior, linked issue, validation, and material limitations.
- Add an entry under `Unreleased` in `CHANGELOG.md`. Document new CLI options
  and their actual defaults in `docs/cli-reference.md`, with README examples
  when they help explain the workflow.

## Project constraints

- Preserve photo contents. Relocation requires explicit move options; replacing
  existing XMP sidecars requires the existing explicit overwrite option.
- Keep decisions explainable in reports. Keep cloud judging opt-in.
- Calibrate selection changes on reviewed photos and record the evidence.
  Logic tests do not establish photographic accuracy. Keep private photos out
  of Git and sanitize private paths in published reports.

The current proposed roadmap is tracked in
[issue #6](https://github.com/JRS1986/FotoSort/issues/6).
