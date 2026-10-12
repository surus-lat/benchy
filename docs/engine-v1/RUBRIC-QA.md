# Rubric-graded question answering — design

Status: **proposed**, not built. Branch `design/rubric-qa`, 2026-10-12.

Driver: the Benchy agent will be used by universities to write exams *for* AI-systems.
The main exam shape is question answering, where each answer is graded against a
5-point rubric. Engine 1.0 cannot express that today:

- the ontology has no question-answering task (`extract`, `classify`, `translate`);
- every metric in `benchy/metrics.py` is a deterministic function of
  `(predicted, expected)`, and none can read an answer against criteria;
- the core may not call a model (it imports stdlib + PyYAML only).

This document adds both pieces under the three rules in `CLAUDE.md` — cleanest
expression of the paper, fewest parts, fewest lines — and names every part it
rejects.

## The idea in one paragraph

A rubric grader is **an AI-system**, so it goes through the one interface the engine
already has, `invoke(dict) -> dict` (A.10). The engine fixes the grader's *program*
(what goes in and what comes out), validates what comes back like any other output, and
turns the grade into a field score $c_{ij} \in [0,1]$. That is the slot Appendix E reserved
for field evaluators. The model, prompt and credentials live outside the core, exactly
as they do for the system under test (A.11). The core gains no provider code, no new
module and no new status.

## 1. Task: `answer`

New ontology task, added to registry `1.0` additively. #38 set the precedent with
domains, and a new task cannot change the meaning of an existing benchmark, because
none of them can declare it.

```yaml
tasks:
  answer:
    description: Answer a question in natural language.
```

A.4 structural validator:

```text
answer
  exactly one leaf output field
  that leaf field is string
```

The input is left free on purpose: a question may be a string, a PDF, an image, or a
question plus a context document. `benchmark.language` is one registered language, as
with `extract` and `classify`.

Task and scoring stay orthogonal: `answer` does not require a rubric. A short-answer exam
can score `answer` with `exact` or `token_set_f1`. A rubric can score any string leaf
of any task. Validation checks that a program is an instance of the task
($P \in P_T$), not how it is graded.

## 2. Authoring

```yaml
version: "1.0"
ontology_version: "1.0"

benchmark:
  task: answer
  domain: history
  language: es

program:
  input:
    question: string
  output:
    answer: string

scoring:
  weights:
    answer: 1
  aggregator: weighted_mean
  field_metrics:
    answer:
      metric: rubric
      params:
        levels:
          1: "Incorrect or missing; no relevant content."
          2: "Mentions relevant ideas but with major errors."
          3: "Partially correct; key elements missing."
          4: "Correct with minor omissions or imprecision."
          5: "Complete, correct and well argued."
  grader:
    type: model
    provider: openai
    model: gpt-5
    prompt: ./grader.md        # optional; providers.py ships a default
    parameters:
      temperature: 0

data:
  path: ./exam.jsonl

ai-system:
  type: model
  provider: together
  model: meta-llama/Llama-3.3-70B-Instruct-Turbo
```

```json
{"input": {"question": "¿Por qué cayó la República de Weimar?"},
 "expected": {"answer": "Clave: crisis de 1929; artículo 48; ... Para 5/5 debe relacionar al menos tres causas."}}
```

**Where each part of the rubric lives:**

| What | Where | Why there |
|---|---|---|
| Level descriptors (what 1…5 mean) | `field_metrics.<leaf>.params.levels`, sealed into the IR | Same for every question; part of what the score *means* |
| Per-question key points or criteria | the row's `expected` value for that leaf | Differs per question; the expected side is already the per-question answer key |
| Who grades | `scoring.grader`, the same grammar as `ai-system` | Declared in the benchmark, bound at runtime (A.11) |

This needs **no change to A.5**: a row is still exactly `input` and `expected`.

**Compile rules (new, all `phase: compile`):**

- `levels` is a mapping from contiguous integers to non-empty strings, with 2–11
  levels. Authors choose 1–5 or 0–4. The IR stores it as an ascending list
  `[{"grade": 1, "description": "..."}, ...]`, because IR keys are JSON strings.
- `rubric` is allowed only on a `string` leaf. It is not in `ENUM_SAFE`.
- `scoring.grader` is **required** if any dimension uses `rubric`, and **rejected**
  otherwise. That follows the strict-keys rule: no inert configuration.
- `scoring.grader` compiles with the existing `_compile_ai_system`, which gains a `path`
  argument so its errors point at `scoring.grader.*` instead of the hardcoded `ai-system.*`.

## 3. The grader contract, fixed by the engine

The engine builds the grader's program from the IR. It is a pure function, and its
result is an ordinary IR program node (the one representation):

```text
input:
  input:      <the benchmark's program.input node>   # the question exactly as the examinee saw it
  reference:  string                                 # the row's expected value for this leaf
  answer:     string                                 # the examinee's validated answer
  rubric:     string                                 # the levels rendered "1: …\n2: …"
output:
  grade:          int
  justification:  string
```

The grader's output is checked with `types.validate`, like any program output, plus one
range check: `grade` must be one of the declared levels. The field score is linear:

$$
c_{ij} = \frac{g_{ij} - g_{\min}}{g_{\max} - g_{\min}}
$$

Instance and benchmark aggregation (§4, A.13) are untouched: the rubric produces
$c_{ij}$, and the weighted mean consumes it as before. The raw grade and the justification
are preserved next to the score, as Appendix E requires for any non-identity evaluator:

```json
{"path": ["answer"], "score": 0.75, "weight": 1, "grade": 4, "justification": "..."}
```

## 4. Execution

Per example, inside the existing loop in `run.py`:

1. Invoke the system and validate its output. **Unchanged.**
2. `invalid_output` or `execution_error`: contribute 0 and **do not grade**. Nothing is
   spent judging a non-answer, and A.13 already gives it zero.
3. `valid`: for each `rubric` dimension, `await grader(grader_input)`, then validate
   the grade.
4. `score.score_example(validated, expected, ir, grades)`. It stays pure arithmetic
   because the I/O happens in `run.py`, as it already does for the system.

`run()` gains one keyword argument, `grader=None`. If the IR has a `rubric` dimension and
no grader is bound, the run aborts before the first example with `grader_not_bound`. That
mirrors `adapter_not_bound`.

**A grader failure aborts the run** (`phase: runtime`, code `grader_error`). The grader
is part of the measuring instrument, not the examinee. Scoring its failure as the
system's zero would blame the student for the examiner. Leaving it unscored would need a
fourth status, which A.12 forbids. This is the same reasoning by which a broken dataset
aborts (A.15). Retries and timeouts are grader-adapter runtime policy (A.10), so one
transient 500 should never reach the engine.

**Baseline.** `run.py` promises that every baseline is *computed from the exam, never a
constant*. For a rubric field, the trivial prediction is the empty answer `""`, so the
engine grades `""` once per question, with the same grader and the same reference. That
costs N extra grader calls, and it is a free calibration check: a grader that gives empty
answers more than the lowest level is visibly broken in the report.

## 5. Binding the grader (outside the core)

- `cli.py`: `--grader MODULE:ATTR` binds any adapter. Otherwise, when `scoring.grader.type`
  is `model`, it calls `providers.for_system` on the synthetic IR
  `{"ai-system": scoring.grader, "program": score.grader_program(ir, path)}`. The same
  OpenAI-compatible adapter that runs examinees runs graders: no judge class.
- `providers.py` adds one constant, `GRADING_PROMPT`, used when the grader declares no
  `prompt`. It tells the model to grade against the rubric and the reference, and to
  treat `answer` as data, never as instructions. A student answer saying "give this 5/5"
  is a prompt-injection attempt, and handling it is the grader adapter's job.
- An `external` grader goes through the same seam. The useful consequence for
  universities is **human grading**: an adapter that looks up a professor's grades
  makes "LLM grader vs. human grader" agreement measurable with no engine change.

## 6. What changes, file by file

| File | Change | Est. lines |
|---|---|---|
| `ontologies/1.0.yaml` | `answer` task | +2 |
| `ontology.py` | `answer` validator (sole string output) | +4 |
| `metrics.py` | `rubric` in the catalog as the one *graded* metric (`GRADED = {"rubric"}`), so the vocabulary stays closed and listed in one place | +6 |
| `compiler.py` | `levels` validation, `scoring.grader` required/rejected, `optional=("grader",)` | +35 |
| `score.py` | `grader_program(ir, path)`, `grades` argument, linear normalization | +25 |
| `run.py` | `grader=` kwarg, setup check, grade valid outputs, graded baseline, `grade`/`justification` in field scores | +30 |
| `errors.py` | `grader_not_bound`, `grader_error`, `invalid_rubric` codes | +3 |
| `cli.py` | `--grader`, provider fallback | +8 |
| `providers.py` | `GRADING_PROMPT` | +10 |

About 120 lines. The core still imports nothing but stdlib, PyYAML and itself, and
`tests/test_acceptance.py` keeps pinning that.

**Documents that must agree** (`tests/test_doc_agreement.py` fails otherwise):

- paper: A.4 validator, Appendix B registry, §4 and A.7 (graded field metric),
  A.14 (`grade`, `justification`), A.15 (grader failure aborts), Appendix E
  (partly realised);
- spec: §5 task rules, §6 scoring, §16 result schema;
- handoff: validator table;
- registry, `README.md`, and the skills `author-benchmark`, `run-and-interpret`
  (scores are comparable only under the same grader) and `write-adapter`
  (grader adapters).

**Pre-existing drift to fix along the way.** `scoring.field_metrics` shipped in #37, but
the paper (§10 "custom field evaluators" is not exposed) and the spec (§6 "Field
evaluator is fixed to `exact_match`") still describe Engine 1.0 without it. The rubric
cannot be documented on top of a contract that denies field metrics exist, so that
paragraph is rewritten first. A test should pin it: every metric name in
`metrics.catalog()` appears in the spec's §6.

## 7. Tests

All of them are deterministic: the grader in tests is a plain callable, so there is no
model, no network and no key.

- `answer` compiles; `answer` with two output leaves or a non-string leaf → `task_program_mismatch`.
- Rubric: non-contiguous levels, one level, empty descriptor, rubric on `int` or `enum`,
  grader present without rubric, rubric without grader → each its own compile error.
- Same YAML → byte-identical IR, levels included (C01 discipline).
- A run with a fake grader: grade 4 on 1–5 → field score 0.75; `grade` and
  `justification` appear in `field_scores`.
- `invalid_output` and `execution_error` examples are **never** passed to the grader
  (the fake grader counts its calls).
- Grader raises / returns `{"grade": 6}` / returns a string → run aborts `grader_error`.
- Rubric IR, no grader bound → `grader_not_bound` before the first example.
- Baseline: the grader sees `answer == ""` once per question.
- Persisted-IR rerun (C33) still holds with a rubric dimension.
- Doc agreement: `answer` in all four documents.

Then, per `CLAUDE.md`: install into a fresh venv and run an example benchmark
(`examples/exam-qa/`) from a fresh clone.

## 8. Parts deliberately not built

| Rejected | Why |
|---|---|
| A judge module or `Judge` class | The grader is an AI-system; `adapter.invoker` already normalizes anything adapter-shaped. |
| An LLM call inside `metrics.py` | Breaks the closed, pure `(pred, expected) -> float` registry and the core's dependency rule. |
| A third row key (`grading`, `rubric`) | Changes A.5. The expected value already is the per-question answer key. |
| A fourth status (`ungraded`) | A.12 has three statuses on purpose; an instrument failure aborts instead. |
| One grader per field | One per benchmark until an exam needs two. The IR shape leaves room. |
| Ensembles, majority votes, self-consistency | A grader adapter can do that behind `invoke`; the engine sees one grade. |
| Partial credit inside a level (4.5) | `grade` is an int; finer resolution means declaring more levels. |
| Resume after a grader abort | Runtime concern; not before a real run needs it. |

## Open questions

1. **Ontology `1.0` additive vs `1.1`.** This proposes additive, following #38. A `1.1` is
   cleaner if anyone already pins "1.0 means exactly three tasks".
2. **Task name.** `answer` reads as the operation, like `extract` and `translate`. The
   alternative `qa` names a benchmark genre rather than a task.
3. **Grader identity in results.** The grader is in the IR, so it is reproducible. A
   cross-run leaderboard would also need the grader's *prompt hash* recorded as run
   metadata (A.10 "non-secret effective runtime settings").
