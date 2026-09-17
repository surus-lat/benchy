# Benchy Engine 1.0 — Normative Conformance Specification v1.2

This is the compact normative specification for implementation and tests.

Keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 1. Source

- The source MUST be YAML.
- Duplicate mapping keys MUST be rejected.
- YAML aliases, anchors, merge keys, and custom tags MUST be rejected.
- The root MUST contain exactly:
  - `version`
  - `ontology_version`
  - `benchmark`
  - `program`
  - `scoring`
  - `data`
  - `ai-system`
- Unknown root keys MUST be rejected.

## 2. Versions

- `version` MUST equal `"1.0"`.
- `ontology_version` MUST equal a loaded supported ontology version.
- Engine 1.0 supports ontology `"1.0"`.
- The benchmark MUST NOT need to provide an ontology filepath; runtime resolves the registry by version from its installed/configured registry store.

## 3. Benchmark classification

- `benchmark.task` MUST exist in registry tasks.
- `benchmark.domain` MUST exist in registry domains.
- Non-translation `benchmark.language` MUST be one registered language string.
- Translation language MUST be:
  ```yaml
  language:
    source: <registered language>
    target: <registered language>
  ```

## 4. Program grammar

- `program.input` MUST be a non-empty mapping.
- `program.output` MUST be a non-empty mapping.
- Every field name MUST be a non-empty string.
- Every field MUST be required.
- Null field values MUST be rejected.
- Arrays/lists MUST NOT appear as schema nodes.
- A leaf MUST be:
  - a primitive type token; or
  - an enum declaration.
- Primitive type tokens:
  ```text
  string int float bool date time datetime image audio document
  ```
- Enum:
  ```yaml
  enum: [<non-empty distinct strings>]
  ```
- Nested mappings MUST be recursively interpreted as object schemas.
- Variable-length collections MUST be rejected.

## 5. Task validation

Ontology 1.0 rules:

- `extract`: no additional structural constraint.
- `classify`: exactly one output leaf; it MUST be enum.
- `translate`: at least one string input leaf; exactly one string output leaf; source/target language object required.

Unsupported tasks MUST fail compilation even if present in another registry version.

`transcribe` is NOT a task in ontology 1.0. Exact match (§12) cannot rank transcription
systems, so a benchmark declaring `transcribe` MUST fail compilation as an unknown task
rather than produce uninterpretable scores. The `audio` semantic type is unaffected and
remains a valid input type. See paper Appendix E.

## 6. Scoring

- `scoring.aggregator` MUST equal `weighted_mean`.
- Every output leaf MUST have exactly one weight.
- Intermediate object nodes MUST NOT have weights.
- Extra weights MUST be rejected.
- Missing weights MUST be rejected.
- Every weight MUST be finite and `>= 0`.
- Sum of all leaf weights MUST be `> 0`.
- Field evaluator is fixed to `exact_match`.
- Benchmark aggregator is fixed to `mean`.

## 7. Data

- `data.path` MUST identify a readable JSONL file before execution.
- Each non-empty JSONL line MUST contain exactly `input` and `expected`.
- Dataset MUST contain at least one example.
- Rows MUST be streamed.
- `input` MUST strictly conform to compiled input schema.
- `expected` MUST strictly conform to compiled output schema.
- Dataset-row failure MUST abort the run.

## 8. Strict object conformance

For every object schema:

- missing key → invalid;
- extra key → invalid;
- wrong type → invalid.

No unknown fields are ignored.

## 9. Runtime value validation

- `string`: string.
- `enum`: string and member of declared enum.
- `int`: integer, not boolean.
- `float`: finite integer or floating numeric value, not boolean.
- `bool`: boolean.
- `date`: canonical `YYYY-MM-DD`.
- `time`: canonical `HH:MM:SS[.fraction]`.
- `datetime`: RFC 3339 with timezone.
- `image/audio/document`: existing readable regular-file path string.
- nested object: exact recursive mapping.

## 10. Artifact representation and benchmark workspace

Execution MUST have an allowed benchmark workspace root.

Benchmark-owned paths such as `data.path` and prompt paths MUST resolve relative to that workspace root.

Dataset artifact references MUST be relative path strings.

Engine MUST resolve artifact references against the dataset file directory, canonicalize the resulting real path, and verify that it remains inside the allowed benchmark workspace.

Path traversal and symlink escape outside the workspace MUST be rejected.

Adapter MUST receive resolved absolute local path strings.

A hosted orchestration/storage layer MAY use object storage or another backend, but required assets MUST be materialized into the benchmark workspace before core engine execution.

Adapter artifact outputs MUST be local path strings inside the allowed run output workspace and MUST identify existing readable files.

Filesystem paths are an Engine 1.0 runtime representation, not a semantic invariant of the `image`, `audio`, or `document` types.

## 11. Adapter protocol

Core semantic call:

```python
async invoke(input_object) -> output_object
```

Every AI-system MUST be invoked through this adapter protocol, including `ai-system.type: model`.

- adapter exception → `execution_error`;
- runtime timeout → `execution_error`;
- returned value failing output schema → `invalid_output`;
- valid output → `valid`.

Engine MUST validate input before invocation and output after invocation.

Credentials, timeout values, concurrency limits, retries, caching, and transport configuration MUST remain runtime policy outside benchmark YAML.

## 12. Exact match

After validation:

- string: exact equality;
- enum: exact equality;
- int: exact equality;
- float: numeric equality;
- bool: exact equality;
- date: parsed date equality;
- time: parsed time equality;
- datetime: same instant;
- artifacts: byte-for-byte content equality.

No hidden normalization.

## 13. Instance scoring

For leaf \(j\) of example \(i\):

$$
c_{ij}=\mathbf{1}[\hat y_{ij}=y_{ij}^*]
$$

Instance score:

$$
s_i=\frac{\sum_jw_jc_{ij}}{\sum_jw_j}
$$

## 14. Failure contribution

- valid → stored score = \(s_i\), contribution = \(s_i\);
- invalid_output → stored score = null, contribution = 0;
- execution_error → stored score = null, contribution = 0.

## 15. Benchmark score

For \(N\) dataset examples:

$$
B(\mathrm{AI})=\frac{1}{N}\sum_{i=1}^{N}q_i
$$

Every dataset example remains in the denominator.

## 16. Result schema

If execution is concurrent, result objects MUST retain dataset indices and the serialized `results` array MUST be emitted in dataset order.

Run result MUST contain:

```text
version
benchmark_score
summary.examples
summary.valid
summary.invalid_outputs
summary.execution_errors
results
```

Each example result MUST contain:

```text
index
status
prediction
field_scores
score
contribution
error
```

## 17. JSON IR

IR MUST contain:

```text
version
ontology_version
benchmark
program.input
program.output
scoring.evaluator = exact_match
scoring.dimensions[]
scoring.instance_aggregator = weighted_mean
scoring.benchmark_aggregator = mean
data.path
data.format = jsonl
ai-system
```

IR scoring dimension paths MUST be arrays of field names.

The engine MUST NOT read or reinterpret YAML after compilation.

## 18. Error object

Diagnostic errors SHOULD expose:

```json
{
  "phase": "compile|dataset|runtime",
  "code": "...",
  "path": ["..."],
  "message": "..."
}
```

Message wording is non-normative.

## 19. Hosted execution invariant

The core engine MAY execute on a hosted server or worker. Storage location is outside benchmark semantics.

Before execution, orchestration MUST ensure the benchmark YAML dependencies and dataset artifacts are available inside the engine's allowed workspace. Moving from local disk to shared/object storage MUST NOT require changing the semantic benchmark definition.

## 20. Non-goals

Engine 1.0 MUST NOT require:

```text
normalization layer
ontology constraint DSL
variable output collections
custom scoring evaluators
alternate aggregators
alternate data formats
remote artifact URIs
adapter DSL
```
