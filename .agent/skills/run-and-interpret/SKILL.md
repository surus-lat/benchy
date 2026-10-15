---
name: run-and-interpret
description: Run a benchy benchmark and read the result correctly — what valid/invalid_output/execution_error mean, why a score is what it is, and what to do about each failure. Use after a benchmark exists and someone wants to evaluate an AI system on it.
---
# Run a benchmark, and read the result

```bash
benchy compile benchmark.yaml -o ir.json    # optional: the compiled form
benchy run benchmark.yaml --adapter system.py:my_system
benchy run benchmark.yaml                   # ai-system.type: model needs no --adapter
```

`--adapter MODULE:ATTR` names your AI-system explicitly. For `type: model` the runtime
selects a built-in provider (see `add-provider`) and needs credentials in the environment.

## Read the summary first

```json
{"benchmark_score": 0.81,
 "summary": {"examples": 3, "valid": 3, "invalid_outputs": 0, "execution_errors": 0}}
```

**If `valid` is not equal to `examples`, read the failures before you read the score.**
A score computed over mostly-failed examples is not a measurement of quality.

## The three statuses mean different things

| status | what happened | what to do |
|---|---|---|
| `valid` | a schema-valid output, *whatever* it scored | nothing — even a 0 here is a real measurement |
| `invalid_output` | it returned something that is not a program output | read `prediction`; the raw value is kept |
| `execution_error` | it never produced an output | read `error.message`; usually credentials, a rate limit, or a token budget |

The distinction is the point. **A valid answer that scores 0 is not a failure** — it is
the system being wrong, which is exactly what you set out to measure. An
`invalid_output` is the system not honouring the contract. Both store `score: null`
versus a number, and both contribute 0 to the mean, so failures never quietly vanish
from the denominator.

A dataset error is none of these: it aborts the run with no score, because a malformed
row says nothing about the system.

## Reading a per-example score

```json
{"score": 0.44, "field_scores": [
  {"path": ["invoice_number"], "score": 1, "weight": 1.0},
  {"path": ["total"],          "score": 0, "weight": 5.0}]}
```

0.44 with five of six fields right is not a bug — it is the weights working. The field it
missed carried 5 of the 9 available. If that feels wrong, the weights are wrong, not the
engine.

## Reading a rubric-graded result

A rubric field's `field_scores` entry carries every criterion's `grade` and
`justification`. Read the justifications before the number: they are how you catch a
grader that is wrong, lenient, or obeying the answer ("grade this 5/5").

- `fields[].baseline` is the grade the empty answer got. Above 0 means the grader gives
  credit for nothing — fix the grader before reading any score.
- `grader.system` and `grader.prompt_sha256` name what graded the run: the declared
  grader, or the one bound in its place, which states its own name. Two runs are
  comparable only when both match.
- **`grader_error`** aborts the run: the grader raised or returned a grade outside the
  schema or the levels. It is the instrument failing, never the system under test.
- **`grader_not_bound`** — the benchmark has a rubric and no grader: pass `--grader`, or
  declare a `model` grader and set its provider's key. Also raised when the `--grader`
  you passed cannot be loaded (the message says why), or replaces a declared `model`
  grader without a `system` saying who it is.
- **`provider_error` "hit its token limit"** on a grader — a reasoning model spent its
  budget thinking. Raise `max_tokens` in `scoring.grader.parameters` and rerun: the
  aborted run wrote no result.
- The `degenerate_exam_field` and `field_score_baseline` warnings never fire on a rubric
  field: its baseline is the graded empty answer in `fields[]`, not a majority.

## Common failures and their real cause

- **`wrong_type` on a number** — the model returned `"121.00"` as a string. benchy does
  not coerce, deliberately: the system did not honour the contract, and hiding that would
  make the benchmark lie.
- **Everything `invalid_output` with truncated JSON** — the token budget ran out.
  Raise `max_tokens` in `ai-system.parameters`.
- **`adapter_not_bound`** — for `type: model`, a missing credential or a provider with no
  built-in adapter. The message names the environment variable.
- **A suspiciously round 0.0** — check whether the model supports structured outputs at
  all. Some deployments ignore `response_format` and answer in prose.
