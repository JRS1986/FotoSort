# OpenAI judge default migration — 2026-10-11

PR #30 proposes `gpt-6.1-sol` as the OpenAI default when `--judge` is enabled.
Use `--judge-model gpt-5.6-sol` to select the previous default. Cloud judging
remains opt-in, and local endpoints still require an explicit model.

## Compatibility and cost

The [GPT-6.1 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
lists image input and Chat Completions support for requests without tools.
FotoSort sends no tools or sampling parameters. Both this model and the
[previous default](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
default to medium reasoning, which FotoSort leaves at the provider default.
The request format, prompts, output-token limits, and Anthropic default are
unchanged.

As of 2026-10-11, published Standard rates for up to 272K input tokens are
$2 input / $10 output per million tokens for GPT-6.1 Sol, versus $4 / $20
for GPT-5.6 Sol. These are uncached input and output token rates, not a
measurement of cost per photo or per run. Token usage, reasoning output,
image accounting, caching, processing tier, and retries can change the
actual bill. No matched-run cost comparison has been recorded.

## Evidence available

The original [PR #30 description](https://github.com/JRS1986/FotoSort/pull/30)
reported a live smoke run on three safari JPEGs using the existing Chat
Completions path: JSON-object output, `max_completion_tokens`, and high-detail
images. It reported a parsed pick, a subject label, and 5,802 input / 184
output tokens. This is author-reported API compatibility evidence; it was
not rerun during this review and does not establish selection quality.

Automated regression coverage exercises the CLI model choice through the
OpenAI SDK with an in-memory HTTP transport, including default and explicit
models, image requests, parsed verdicts, subject naming, and token accounting.
It also checks Anthropic model routing with a stub client. These checks make
no external requests and cannot establish live model behavior or photographic
accuracy.

## Remaining calibration

The owner confirmed on 2026-10-11 that no reviewed evaluation set is available.
The existing archived shortlist evaluations contain no matched final judgments
from these two models, so they cannot validate this migration. Public vision
benchmarks and a successful API response do not replace photographer feedback.
The PR remains a draft pending the calibration required by `AGENTS.md` and
`CONTRIBUTING.md`.

- Obtain explicit favourite, reject, acceptable-alternative, or pairwise
  feedback on representative photos; keep private photos and manifests out
  of Git.
- Compare the old and proposed defaults on identical inputs, prompts,
  preselection, and declared candidate/request budgets. Record the actual
  model identities, token usage, failures, runtime, and cost separately.
- Use the [evaluation workflow](../evaluation.md) to publish a sanitized
  comparison, including unknown outcomes and any regressions. Have the
  maintainer assess the evidence before promoting this default.

Tracked in [issue #29](https://github.com/JRS1986/FotoSort/issues/29).
