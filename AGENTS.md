# Working on FotoSort

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes and follow its
pull-request process in full: issue first, focused branch, linked PR,
validation, changelog, and maintainer review. The project owner requires
development through GitHub issues and pull requests. The points below add
what is specific to agent sessions; CONTRIBUTING.md remains the single source
for the shared rules.

## Agent sessions

- Name branches `codex/<topic>` and create them from the current default
  branch. Preserve existing user changes in the working tree.
- Merge only when the user has authorized it in the current session and the
  required checks pass. Do not push changes directly to `main`.
- Run `ruff check fotosort tests` and `pytest`, and report the actual results,
  including checks that could not run. Resolve relevant CI failures before
  marking a PR ready.
- Track remaining work explicitly in the issue or PR rather than declaring it
  done.

## Project constraints

- Preserve photo contents. Relocation requires explicit move options; replacing
  existing XMP sidecars requires the existing explicit overwrite option.
- Keep decisions explainable in reports. Keep cloud judging opt-in.
- Calibrate selection changes on reviewed photos and record the evidence.
  Logic tests do not establish photographic accuracy. Keep private photos out
  of Git and sanitize private paths in published reports.

The roadmap is tracked in
[issue #6](https://github.com/JRS1986/FotoSort/issues/6).
