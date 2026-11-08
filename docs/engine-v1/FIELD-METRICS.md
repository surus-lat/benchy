# Field metrics and baselines — `scoring.field_metrics` + the anti-trivial guard

Date: 2026-10-09. Lane `scoring` of the datapipeline / programpipeline / benchy
consolidation. One scoring, benchy principal: the closed per-field metric library
from `program_pipeline/scoring.py` becomes `benchy/metrics.py`, and the mandatory
trivial baselines of datapipeline's derived scoring land in the run's report.

## What a benchmark may declare

```yaml
scoring:
  weights: {...}
  aggregator: weighted_mean
  field_metrics:                      # optional
    supplier.name: {metric: token_set_f1}
    total: {metric: numeric_tolerance, params: {tolerance: 0.5}}
  signal_epsilon: 0.02                # optional, default 0.01, sealed in (0, 0.10]
```

- `field_metrics` maps a dotted leaf path to `{metric: <name>, params: {...}}` drawn
  from the **closed** registry in `benchy/metrics.py`: `exact` (the default),
  `casefold_strip`, `token_set_f1`, `date_flexible`, `numeric_tolerance`
  (param `tolerance`), `span_recall`, `set_f1`.
- The compiler rejects anything outside the registry: an unknown metric is
  `unknown_metric` carrying the field path; an unknown or out-of-range parameter is
  `unknown_param` / `param_out_of_range`.
- **Enum-preserving rule**: a leaf declared `{enum: [...]}` only accepts enum-safe
  metrics (`exact`, `casefold_strip`) — anything else is `enum_unsafe_metric`,
  because relaxing a closed vocabulary into a similarity metric dissolves it.
- A dimension with no entry carries no metric key and scores with the canonical
  exact match, so a benchmark without `field_metrics` compiles to the exact same IR
  as before — and scores identically (verified against `examples/invoices`).

## Per-row routing: mixed exams (2026-11-08)

One benchmark may mix question kinds — some rows open-answer (judged against the
rubric), some multiple-choice (scored by exact match). Two mechanisms, both
backward-compatible:

- **Optional input fields.** `program.input` may declare a field with one `?`
  suffix — `options?: {a: string, b: string, c: string, d: string}`. The IR node
  carries `"optional": true`; a row that omits the field validates (it is skipped,
  not stored), a row that carries it is validated exactly as before, and an
  undeclared field still fails `extra_field`. The marker is rejected in
  `program.output`: every output field is a scoring dimension on every row.
- **`route` on a field-metric entry.** A `scoring.field_metrics` entry may carry
  `route: {<optional input field>: <metric>}`:

```yaml
field_metrics:
  answer:
    metric: rubric
    params: {criteria: {...}}
    route: {options: casefold_strip}   # rows carrying `options` score by exact match
```

  A row whose input carries the routed field scores by the routed metric (drawn
  from the closed registry — never `rubric`); every other row scores by the
  field's own metric. Rows routed off the rubric are never sent to the grader,
  and their baseline is the trivial prediction under the row's own metric instead
  of the graded empty answer. Routing requires the input field to be **optional**:
  a required field is present on every row, so routing on it would decide nothing.

A benchmark without `?` or `route` compiles to byte-identical IR and scores
exactly as before.

## Aggregation does not change

`benchy/score.py` keeps the paper's aggregation: weighted mean per instance, mean of
contributions with failures in the denominator. What a metric changes is the field
value `c_ij`, which is now a float in [0, 1] (partial credit is a first-class score).
`weight: 0` still means "validates and reports, does not score".

## Baselines, computed from the exam

Every run reports, per field, in the top-level `fields` block:

| key | meaning |
|---|---|
| `score` | mean of the field's contributions over *every* example (failures count 0) |
| `baseline` | the score a trivial predictor gets on this exam, under the field's own metric |
| `signal` | `score - baseline` |
| `counts_as_signal` | `score >= baseline + signal_epsilon` |

The trivial predictor is fitted on the exam's own `expected` values: majority for
`enum`/`bool`/artifacts, the mean for `int`/`float`, the empty string for text and
temporals. Everything published is computed — there are no decorative constants.

## Parity notes

- Text normalization (`_fold`/`_tokens`) is ported verbatim from
  `program_pipeline/scoring.py`: NFKD + casefold + accent strip, alphanumeric
  tokens. Tokenization parity with the optimization loop is sacred.
- `span_recall` and `set_f1` are ported from `datapipeline/derived_scoring.py`;
  datapipeline vendors this registry back as `datapipeline/scoring_metrics.py`
  (byte-identical semantics, guarded by a fixed-vector parity test).
