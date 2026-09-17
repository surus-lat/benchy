# Benchy Engine 1.0 — Build Agent Handoff

## Purpose

This document is the primary handoff for the agent implementing the new Benchy engine.

Read the accompanying `technical-paper-v10.md` for the semantic model. This document translates that model into an implementation architecture.

The engine must preserve these invariants:

1. YAML is the only authored semantic source.
2. One semantic concept has one accepted YAML syntax.
3. There is no normalization/repair layer.
4. Valid YAML compiles deterministically into JSON IR.
5. The engine consumes only JSON IR; it does not reinterpret YAML.
6. Every program uses fixed named input and output schemas.
7. Leaf output fields are scoring dimensions.
8. Every scoring dimension has exactly one explicit non-negative weight.
9. Exact match is the only field evaluator.
10. `weighted_mean` is the only instance aggregator.
11. Benchmark aggregation is arithmetic mean across all example contributions.
12. Benchy uses `AI-system` terminology for the benchmark taker.
13. The engine knows one runtime contract.
14. Adapters own integration-specific translation.
15. Dataset rows and AI-system outputs are validated strictly: missing, extra, or wrong-typed fields are invalid.
16. Invalid AI-system outputs and execution failures store `score = null` but contribute zero to the final aggregate.
17. Dataset errors abort the run; they are not AI-system failures.
18. Media/document dataset values are relative paths; the engine resolves them to absolute path strings before adapter invocation.

---

## 1. Recommended package boundaries

Do not over-modularize. The following boundaries are enough:

```text
benchy/
  parser
  compiler
  schema
  ontology
  ir
  data
  runtime
  scoring
  results
  errors
```

Responsibilities:

### parser

- parse YAML;
- reject duplicate mapping keys;
- reject aliases/anchors/merge keys/custom tags;
- return a raw parsed source object;
- no semantic repair.

### schema

- parse/validate program input/output schema grammar;
- represent semantic types;
- recursively validate named-field objects;
- enumerate leaf fields as path arrays;
- validate enum values;
- validate artifact paths at runtime.

### ontology

- load the shared ontology YAML;
- check ontology version;
- validate task/domain/language membership;
- implement Benchy task validators keyed by ontology version and task.

Do **not** implement a declarative ontology constraint DSL.

### compiler

- validate top-level Benchy definition;
- invoke schema/ontology/scoring checks;
- compile source semantics into canonical IR;
- never inspect dataset rows exhaustively;
- never repair source.

### ir

- define the typed machine-facing representation;
- serializable to JSON;
- contains only compiled/validated semantics required for execution.

### data

- open JSONL;
- stream examples;
- ignore blank lines;
- require at least one example;
- validate `input` and `expected` through compiled schemas;
- resolve artifact paths relative to dataset directory;
- abort run on invalid dataset records.

### runtime

- bind AI-system definition to an adapter;
- call adapter asynchronously;
- distinguish setup errors, execution errors, and returned invalid outputs;
- default sequential execution is acceptable.

### scoring

- exact-match typed equality;
- field score calculation;
- weighted mean;
- benchmark mean over contributions.

### results

- per-example result models;
- run summary;
- benchmark score;
- serialization.

### errors

- structured diagnostics `{phase, code, path, message}`;
- compiler and runtime errors use shared shape.

---

## 2. End-to-end flow

```text
load benchmark YAML
↓
parse strict YAML
↓
load ontology requested by ontology_version
↓
validate source semantics
↓
compile canonical JSON IR
↓
bind AI-system adapter
↓
open JSONL dataset
↓
for each example:
    validate + resolve input
    validate expected output
    invoke adapter
    if invocation failed:
        execution_error
        contribution = 0
        continue
    validate returned output
    if output invalid:
        invalid_output
        contribution = 0
        continue
    exact-match each scoring dimension
    weighted_mean → instance score
    contribution = instance score
↓
benchmark_score = mean(all contributions)
↓
emit run result
```

If dataset loading or validation fails, abort the run. Do not emit a normal benchmark score.

---

## 3. Strict source grammar

Top-level keys:

```text
version
ontology_version
benchmark
program
scoring
data
ai-system
```

Unknown keys fail compilation.

### benchmark

Single-language:

```yaml
benchmark:
  task: extract
  domain: finance
  language: es
```

Translation:

```yaml
benchmark:
  task: translate
  domain: general
  language:
    source: es
    target: en
```

No additional keys.

### program

```yaml
program:
  input:
    field: type
  output:
    field: type
```

Input/output non-empty.

A field value is one of:

1. scalar semantic type token;
2. enum declaration;
3. nested object.

Enum declaration:

```yaml
field:
  enum: [a, b, c]
```

Enum members must be distinct non-empty strings.

No arrays anywhere else in program schemas.

### scoring

```yaml
scoring:
  weights: ...
  aggregator: weighted_mean
```

`weights` mirrors the output schema to the leaves.

`aggregator` must equal `weighted_mean`.

### data

```yaml
data:
  path: ./data/file.jsonl
```

No additional semantic data options in 1.0.

### ai-system

External:

```yaml
ai-system:
  type: external
  id: name
```

Model:

```yaml
ai-system:
  type: model
  provider: provider-id
  model: model-id
  prompt: ./prompt.md
  parameters: {}
```

`parameters` is a mapping whose provider-specific meaning belongs to the provider adapter.

---

## 4. Program schema representation

Recommended internal schema nodes:

```text
ObjectSchema
  fields: ordered map<string, SchemaNode>

PrimitiveSchema
  type: string|int|float|bool|date|time|datetime|image|audio|document

EnumSchema
  values: list<string>
```

Do not create task-specific schema classes.

Leaf enumeration:

```text
output:
  supplier:
    name: string
    tax_id: string
  total: float
```

becomes:

```text
["supplier", "name"]
["supplier", "tax_id"]
["total"]
```

Use arrays, not dotted strings, in IR and errors.

---

## 5. Task validators

For ontology 1.0:

### extract

No task-specific structural rule.

### classify

- output leaf count = 1;
- only output leaf is enum.

### transcribe

- at least one input leaf has semantic type `audio`;
- output leaf count = 1;
- only output leaf is `string`.

### translate

- at least one input leaf has semantic type `string`;
- output leaf count = 1;
- only output leaf is `string`;
- benchmark language must be `{source, target}`;
- source and target must both exist in ontology language registry.

Compiler code may implement this directly:

```python
TASK_VALIDATORS_V1 = {
    "extract": validate_extract,
    "classify": validate_classify,
    "transcribe": validate_transcribe,
    "translate": validate_translate,
}
```

The registry does not contain this code/data structure.

---

## 6. Weight compiler

Compile the output leaf set first.

Compile the weight tree separately.

Reject:

```text
missing leaf weight
extra leaf weight
weight on intermediate object
negative weight
non-finite weight
sum(weights) == 0
```

IR scoring dimensions:

```json
[
  {
    "path": ["supplier", "name"],
    "weight": 1.0
  },
  {
    "path": ["total"],
    "weight": 5.0
  }
]
```

Preserve deterministic depth-first field order from source mappings.

---

## 7. Dataset loader

Format: JSONL.

Every non-empty line must parse into:

```json
{
  "input": {},
  "expected": {}
}
```

Exactly two keys.

Use line number for diagnostics and zero-based example index for results.

Stream records; do not load entire dataset.

For every record:

1. validate root structure;
2. validate input against compiled input schema;
3. validate expected against compiled output schema;
4. resolve artifact input/expected path values.

Dataset validation errors abort the run.

### Artifact paths

Dataset values for `image`, `audio`, `document` are relative path strings.

Resolve against the JSONL parent directory.

Return an absolute normalized path string to runtime/scoring.

Require an existing readable regular file.

The adapter receives that resolved path string and decides how to consume it.

---

## 8. Typed runtime validation

Validation is strict and recursive.

Rules:

### string

Python `str` or equivalent.

### enum

String and member of declared enum set.

### int

Integer excluding booleans.

### float

Finite numeric value; integer numeric values are allowed; booleans are not.

### bool

Boolean only.

### date

String in canonical `YYYY-MM-DD`; parse successfully.

### time

String in `HH:MM:SS[.fraction]`; parse successfully.

### datetime

RFC 3339 string including timezone offset or `Z`; parse successfully.

### image/audio/document

Path string resolving to existing readable regular file.

### object

Mapping with exact key set: no missing or extra keys.

---

## 9. Exact match

After both values pass schema validation:

```text
string: exact Unicode equality
enum: exact member equality
int: integer equality
float: numeric equality
bool: boolean equality
date: parsed date equality
time: parsed local time equality
datetime: instant equality after timezone parsing
image/audio/document: byte-for-byte file content equality
```

No hidden normalization.

For artifact equality, stream bytes or compare cryptographic hashes; do not load large files fully if avoidable.

---

## 10. Adapter boundary

Normative semantic protocol:

```python
class Adapter(Protocol):
    async def invoke(
        self,
        input_object: dict[str, object],
    ) -> dict[str, object]:
        ...
```

The adapter may be constructed with:

```text
program contract
AI-system definition
runtime/integration configuration
```

But only `invoke(input_object)` is used per example.

### Adapter owns

- native argument mapping;
- HTTP/SDK/provider calls;
- reading artifact path files if needed;
- provider-specific payloads;
- native output extraction;
- mapping native output into named-field output object.

### Engine owns

- validation before invocation;
- validation after invocation;
- scoring;
- aggregation;
- diagnostics.

If `invoke()` raises: `execution_error`.

If it returns a value but that value fails the output schema: `invalid_output`.

A missing adapter binding is a run setup error, not an example result.

---

## 11. Artifact representation decision

Use **resolved absolute filesystem path strings** at the runtime boundary for `image`, `audio`, and `document`.

Why:

- zero new host-language abstraction;
- trivially serializable/debuggable;
- adapters already own native conversion;
- avoids loading large blobs in the engine;
- a Python adapter can open it, an HTTP adapter can upload it, a provider adapter can decode it;
- the schema still carries the semantic type.

Do not introduce `Artifact`, `ImageValue`, `AudioValue`, etc. in engine 1.0.

This is a runtime representation choice, not a semantic-type change.

---

## 12. Status and error semantics

Per-example statuses:

```text
valid
invalid_output
execution_error
```

### valid

Output conforms to schema. Score can be anywhere in valid aggregator range, including 0.

### invalid_output

Adapter returned a value, but it does not satisfy output schema.

Stored score = null. Contribution = 0.

### execution_error

Adapter invocation failed before a usable output was returned.

Stored score = null. Contribution = 0.

Dataset errors are not statuses. They abort the run.

---

## 13. Result schema

Run:

```json
{
  "version": "1.0",
  "benchmark_score": 0.86,
  "summary": {
    "examples": 100,
    "valid": 96,
    "invalid_outputs": 2,
    "execution_errors": 2
  },
  "results": []
}
```

Valid example:

```json
{
  "index": 0,
  "status": "valid",
  "prediction": {
    "total": 100.0
  },
  "field_scores": [
    {
      "path": ["total"],
      "score": 0,
      "weight": 5
    }
  ],
  "score": 0.0,
  "contribution": 0.0,
  "error": null
}
```

Invalid output and execution-error examples use null score/field scores as specified in the technical paper appendix.

---

## 14. JSON IR contract

The engine receives only IR.

Recommended shape:

```json
{
  "version": "1.0",
  "ontology_version": "1.0",
  "benchmark": {},
  "program": {
    "input": {},
    "output": {}
  },
  "scoring": {
    "evaluator": "exact_match",
    "dimensions": [],
    "instance_aggregator": "weighted_mean",
    "benchmark_aggregator": "mean"
  },
  "data": {
    "path": "...",
    "format": "jsonl"
  },
  "ai-system": {}
}
```

IR schema nodes should be explicit objects, e.g.:

```json
{
  "type": "float"
}
```

and:

```json
{
  "type": "enum",
  "values": ["positive", "negative"]
}
```

Nested objects:

```json
{
  "type": "object",
  "fields": {
    "name": {"type": "string"}
  }
}
```

This removes source-syntax ambiguity before execution.

---

## 15. Compiler validation order

Use this order so diagnostics are stable:

```text
1. strict YAML parse
2. top-level keys
3. Benchy version
4. ontology file/version
5. benchmark task/domain/language membership
6. program grammar
7. task ↔ program validation
8. scoring structure
9. output leaf ↔ weight exact coverage
10. data configuration/path
11. AI-system semantic definition
12. emit JSON IR
```

Do not read all dataset rows during compilation.

---

## 16. Execution order

```text
1. load IR
2. resolve adapter binding
3. open dataset
4. stream next record
5. validate dataset input/expected
6. invoke adapter
7. classify invocation result
8. validate output if present
9. score valid output
10. append example result
11. repeat
12. mean(contributions)
13. emit run result
```

---

## 17. Implementation freedoms

The agent may choose:

```text
Python/Rust/etc.
dataclasses/Pydantic/serde/etc.
internal function/module names
async runtime library
hashing library for artifact equality
logging implementation
CLI structure
```

provided the observable semantics in this handoff are preserved.

Do not encode speculative features into the semantic model.

---

## 18. Explicit non-goals for engine 1.0

Do not implement:

```text
ontology constraint DSL
variable-length output collections
implicit weights
custom evaluators in benchmark YAML
aggregators other than weighted_mean
normalization/repair layer
adapter configuration DSL
automatic arbitrary endpoint inference
remote artifact URIs
alternate dataset formats
provider-specific logic inside engine core
```

Build extension points only where they are trivial; do not generalize prematurely.

---

## 19. Definition of done

The engine is ready when all of these work:

1. compile valid YAML to deterministic IR;
2. reject malformed/ambiguous YAML;
3. validate ontology coordinates;
4. validate the four task rules;
5. validate fixed schemas;
6. enforce strict weight coverage;
7. stream and validate JSONL;
8. resolve media paths;
9. run an external test adapter;
10. distinguish valid / invalid_output / execution_error;
11. exact-match typed values;
12. compute weighted mean instance scores;
13. compute benchmark mean including zero contributions for failures;
14. emit the documented result schema;
15. pass the conformance test matrix in the build plan.
