# Rubric-graded question answering

Status: **built**, branch `design/rubric-qa`, 2026-10-12.

Driver: the Benchy agent will be used by universities to write exams *for*
AI-systems. The main exam shape is open questions whose answers are graded against a
rubric. Engine 1.0 could not express that: the ontology had no question-answering task,
every field metric was a deterministic function of `(predicted, expected)`, and the core
may not call a model (it imports stdlib + PyYAML only).

## Decisions

| Question | Decision |
|---|---|
| Task name | `answer`, which names the operation, like `extract` and `translate` |
| Ontology | a new version **1.1** = 1.0 + `answer`; 1.0 is untouched |
| Rubric shape | **analytic**: named criteria, each with a weight and its own levels. Holistic is the one-criterion case |
| Who grades | an AI-system declared as `scoring.grader`, so a model or a human through the same seam. By default a model, checked against a professor on a sample |
| Grader identity in results | yes: `grader.system` and `grader.prompt_sha256` |

## The idea in one paragraph

A rubric grader is **an AI-system**, so it goes through the one interface the engine
already has, `invoke(dict) -> dict` (A.10). The engine fixes the grader's *program*,
validates what comes back like any other output, and turns the criterion grades into a
field score $c_{ij} \in [0,1]$. That is the slot Appendix E reserved for field
evaluators. The model, prompt and credentials live outside the core, as they do for the
system under test (A.11). The core gains no provider code, no new module and no new
status.

## Authoring

```yaml
ontology_version: "1.1"
benchmark: {task: answer, domain: history, language: es}
program:
  input: {question: string}
  output: {answer: string}
scoring:
  weights: {answer: 1}
  aggregator: weighted_mean
  field_metrics:
    answer:
      metric: rubric
      params:
        criteria:
          accuracy:
            weight: 3
            description: Facts and causes correct against the answer key.
            levels: {1: "...", 2: "...", 3: "...", 4: "...", 5: "..."}
          clarity:
            weight: 1
            levels: {1: "...", 2: "...", 3: "..."}
  grader: {type: model, provider: openai, model: gpt-5-mini}   # or external
```

| What | Where | Why there |
|---|---|---|
| Criteria, weights, level descriptors | `field_metrics.<leaf>.params.criteria`, sealed into the IR | The same for every question, and part of what the score *means* |
| Per-question key points | the row's `expected` value for that leaf | Differs per question. The expected side already is the per-question answer key, so **A.5 is unchanged** |
| Who grades | `scoring.grader`, the same grammar as `ai-system` | Declared in the benchmark, bound at runtime (A.11) |

Compile rules (`invalid_rubric` unless noted):

- `rubric` only on a `string` leaf, and **at most one rubric field per benchmark**. A
  model grader has one output schema, and the use case grades one answer per question.
- `params` is exactly `{criteria}`. Each criterion is `{weight, levels, description?}`
  (`missing_key`/`unknown_key` otherwise). The weights are finite, `>= 0`, and sum to
  `> 0`. `levels` are 2–11 contiguous integers, each described. In the IR, criteria
  keep source order and levels become ascending `[{grade, description}]` lists.
- `scoring.grader` is required when a rubric exists (`missing_key`) and rejected
  otherwise (`invalid_value`). It compiles with `_compile_ai_system`, whose errors now
  point at `scoring.grader.*`.
- A benchmark without a rubric compiles to byte-identical IR as before.

## The grader contract

`score.grader_program(ir)` is an ordinary IR program:

```text
input:   input (the program's input node) · reference · answer · rubric (text)
output:  {<criterion>: {justification: string, grade: int}}   for every criterion
```

The justification comes before the grade on purpose: a model writes structured output
in schema order, so it reasons first and then commits to a level. Asked for the grade
first, it picks a number and rationalizes it, and the two can disagree (seen live: a
`clarity` of 3, "clear", justified as "dense, takes some effort", which is level 2).

The reply is checked with `types.validate`, and every grade must be one of its
criterion's levels. Then:

$$
c_{ij}=\frac{\sum_k w_k r_k}{\sum_k w_k}\qquad r_k=\frac{g_k-g_{k,\min}}{g_{k,\max}-g_{k,\min}}
$$

The field's `field_scores` entry keeps `criteria`: every grade and justification.

## Execution (`run.py`)

- `run(ir, workspace, adapter, *, grader=None)`. With a rubric and no callable grader,
  the run aborts before the first example (`grader_not_bound`). A persisted IR with a
  rubric field and no `scoring.grader` is `invalid_ir` up front (spec §17), not after
  every example has been graded.
- Only `valid` outputs are graded. Failures already contribute 0 (A.13).
- **A grader failure aborts the run** (`grader_error`): the grader raised, or replied
  outside the schema or the levels. The grader is the measuring instrument, so scoring
  its failure as the examinee's zero would blame the student for the examiner. A fourth
  status would break A.12. This is the same reasoning as A.15.
- **Baseline**: the empty answer is graded once per question with the same grader, so
  the baseline is measured like every other one. It is also a free calibration check: a
  grader that gives an empty answer more than the floor is visibly broken.
- The result gains `grader: {system, prompt_sha256}` when a rubric exists.
  `prompt_sha256` comes from the grader adapter if it states one, else `null`.

## Outside the core

- `providers.for_grader(ir, workspace)` runs the same `OpenAIChat` adapter on
  `{"ai-system": scoring.grader, "program": grader_program(ir)}`. It uses
  `GRADING_PROMPT` unless the grader declares `prompt`, and sets `prompt_sha256`. The
  default prompt tells the model to treat the answer as data, never as instructions.
- `cli.py`: `--grader MODULE:ATTR` binds any adapter. Otherwise a `model` grader comes
  from the providers. An unbound `external` grader, or a `--grader` that cannot be
  loaded, is reported as `grader_not_bound` at `scoring.grader`.
- The CLI's field warnings skip the two exam-side checks for a rubric field: they reason
  about a `(predicted, expected)` metric, and a rubric grades each question's answer on
  its own. A constant answer to different questions is still reported.
- **Human grading** goes through the same seam. To check a model grader against a
  professor, rerun a sample with `--adapter` replaying the stored predictions and
  `--grader` returning the professor's grades, then compare criterion by criterion.

## What changed

| File | Change |
|---|---|
| `ontologies/1.1.yaml` | new: 1.0 + `answer` |
| `ontology.py` | `_answer_program`; `_VALIDATORS["1.1"]` = 1.0's table + `answer` |
| `compiler.py` | `_compile_rubric`, the `scoring.grader` rules, one rubric per benchmark, `_compile_ai_system(at=…)` |
| `score.py` | `RUBRIC`, `GRADE`, `grader_program`, `grader_input`, `rubric_score`; `score_example(…, grades)` |
| `run.py` | grader binding, grading valid outputs, the graded baseline, the `grader` block |
| `errors.py` | `invalid_rubric`, `grader_not_bound`, `grader_error` |
| `providers.py`, `cli.py` | `for_grader`, `GRADING_PROMPT`, `ANSWER_PROMPT`; `--grader` |
| paper, spec, handoff | 1.1 / `answer` in A.4, Appendix B, spec §5, handoff table; field metrics and rubric in §4 and spec §6; results in A.14 and spec §16; grader failure in A.15; codes in A.16; Appendix E status |
| README, four skills, `examples/exam-qa/` | authoring, interpreting, writing a grader, an offline example |

`metrics.py` is **not** touched. It is vendored byte-identical into datapipeline, and a
graded metric is not a pure `(predicted, expected)` function, so `rubric` lives in
`score.py`.

**Drift fixed along the way.** `field_metrics` shipped in #37 while the paper still said
"custom field evaluators" were not exposed, and the spec said "Field evaluator is fixed to
`exact_match`". Both now describe the closed registry. A new test pins spec §6's metric
list to `metrics.METRICS ∪ {rubric}`. The README's "five domains" (stale since #38) now
counts the academic ones.

## Tests

`tests/test_rubric.py`, all deterministic. The grader is a plain callable:
no model, no network, no key. They cover the 1.1 registry and `answer` validator; every
compile rejection; byte-identical IR; the grader program; weighted normalization;
grade and justification in results; that failed examples are never graded; the graded
baseline; the grader block; `grader_not_bound`; `grader_error` for raise, out-of-range,
wrong type, missing criterion and non-object replies; async graders; reruns from
persisted IR (C33); and the CLI with `--grader`, with a provider grader and with a
declared prompt; a grader that cannot be loaded; the field warnings on a rubric field.
`tests/test_doc_agreement.py` now follows the newest ontology version.

## Parts deliberately not built

| Rejected | Why |
|---|---|
| A judge module or `Judge` class | The grader is an AI-system; `adapter.invoker` already normalizes anything adapter-shaped |
| An LLM call or `rubric` entry in `metrics.py` | Breaks the closed, pure `(pred, expected) -> float` registry, its vendored twin, and the core's dependency rule |
| A third row key (`grading`, `rubric`) | Changes A.5; the expected value already is the per-question key |
| A fourth status (`ungraded`) | A.12 has three on purpose; an instrument failure aborts instead |
| Several rubric fields / several graders | One graded answer per question is the use case; the IR shape leaves room |
| Ensembles, votes, self-consistency | A grader adapter can do that behind `invoke`; the engine sees one grade per criterion |
| Built-in human-vs-model agreement statistics | Two result files compare criterion by criterion; build the report when a real calibration needs it |
| Resume after a grader abort | Runtime concern; not before a real run needs it |
