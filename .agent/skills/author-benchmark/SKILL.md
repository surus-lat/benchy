---
name: author-benchmark
description: Turn a business problem into a benchy benchmark.yaml — choosing the program's input/output schema, the scoring weights, and the exam. Use when someone wants to measure an AI system on their own task rather than run an existing benchmark.
---
# Author a benchmark

A benchmark is `B = (P, S, D)` — a program, a scoring function, and a dataset. It is
separate from the AI-system that takes it. Your job here is the *exam*, not the taker.

This is the step benchy exists for: VISION says the benchmark is the bridge between an
AI capability and a business problem, so the hard part is judgement, not syntax.

## The four questions, in order

**1. What is the program?** A typed contract, `P : X → Y`. Named input fields in, named
output fields out. Write the output schema first — it is the thing you will score.

```yaml
program:
  input:
    text: string           # or image / audio / document for an artifact
  output:
    invoice_number: string
    date: date
    supplier:
      name: string
      tax_id: string
    total: float
```

Types are semantic, not storage: `date` means a calendar date, and the engine will check
it parses as `YYYY-MM-DD`. Every field is required. No lists — variable-length
collections are outside the language.

**2. What does "good" mean?** Every output leaf needs exactly one explicit weight. This
is where the business lives, and it is worth arguing about:

```yaml
scoring:
  weights:
    invoice_number: 1
    date: 1
    supplier:
      name: 1
      tax_id: 0            # validated and reported, but does not move the score
    total: 5               # getting the money wrong is what actually costs
  aggregator: weighted_mean
```

A weight of `0` is a real choice: the field is still required and still type-checked, it
just does not affect the number. Ask "if the system got only this field wrong, how bad is
that?" — the answers are the weights.

**3. What is the exam?** JSONL, one `{"input": …, "expected": …}` per line. Artifacts are
paths relative to the JSONL file.

**4. Which task, domain, language?** The shared ontology: `extract`, `classify`,
`translate`, or — with `ontology_version: "1.1"` — `answer`. It constrains the program
(`classify` must have exactly one enum output leaf, `answer` exactly one string) so a
mismatch fails compilation rather than scoring nonsense.

## Exams whose answers must be judged

A university exam asks open questions; the answer is prose and has to be graded, not
matched. Use `task: answer` and score the answer with an analytic rubric:

```yaml
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
            levels: {1: "Wrong or empty.", 2: "...", 3: "...", 4: "...", 5: "Complete and correct."}
          argument:
            weight: 2
            levels: {1: "No argument.", 2: "Lists, does not relate.", 3: "Relates causes and effects."}
  grader: {type: model, provider: openai, model: gpt-5-mini}
```

- **The rubric is the professor's grid.** One criterion per row, each with its own
  levels (2–11 contiguous integers, every one described) and a weight. A single
  holistic grade is a rubric with one criterion.
- **The answer key goes in the exam.** Each row's `expected.answer` is what that
  question's grader compares against: the model answer, the points a full answer must
  make, or both. That is where per-question specifics belong.
- **The grader is part of the benchmark.** Scores are comparable only under the same
  grader. Before trusting a model grader, have a professor grade a sample of the *same*
  answers: rerun that sample with `--adapter` replaying the stored predictions from the
  model-graded result and `--grader` returning the professor's grades, then compare the
  two results criterion by criterion.
- **One rubric field per benchmark.**

See `examples/exam-qa/`.

## Where people go wrong

- **Scoring everything equally.** If all weights are 1, you have said nothing about the
  business. Almost every real problem has one or two fields that matter more.
- **Reaching for `string` when the value is closed.** If the answer is one of five
  things, it is an `enum`, and exact match then means something.
- **Expecting partial credit on free text from exact match.** A `string` scored by
  default is right or wrong, nothing in between. For short text, pick a metric from the
  closed registry (`token_set_f1`, `span_recall`, …); for an answer that must be judged,
  use a rubric (above). Transcription is still out of reach — see paper Appendix E.
- **Too few examples.** The score is a mean over the exam; ten rows is a smoke test, not
  a measurement.

## Check it

```bash
benchy compile benchmark.yaml        # rejects, never repairs
```

Compilation is a wall: it tells you exactly what is wrong (`missing_weight`,
`task_program_mismatch`, `invalid_schema`) with the field path. It never guesses.

Then `run-and-interpret` to actually measure something.
