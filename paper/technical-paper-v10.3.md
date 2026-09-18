# Benchy — Technical Design v10.3

## 1. Overview

Benchy is a semantic language and execution engine for benchmarking AI programs.

A **program** defines a typed input/output contract. An **AI-system** is any AI-based implementation of that program: a single AI model; an AI-node such as a model with an optimized prompt and fixed task behavior; a composition of AI models or AI programs; AI components intermingled with explicit or deterministic code; or an agent/workflow composed from those elements.

A benchmark is separate from the AI-system taking it.

$$
\boxed{B=(P,S,D)}
$$

> Benchmark \(B\) is completely specified by program \(P\), scoring function \(S\), and dataset \(D\).

A benchmark run binds that benchmark to an AI-system:

$$
\boxed{R=(B,\mathrm{AI})}
$$

> Run \(R\) evaluates AI-system \(\mathrm{AI}\) on benchmark \(B\).

The design follows a small number of decisions:

- YAML is the canonical semantic definition authored by humans, agents, and the UI.
- Each semantic concept has one valid YAML syntax.
- Valid YAML is deterministically compiled into a canonical JSON intermediate representation (IR).
- Compilation changes representation, not meaning. It does not repair invalid definitions or inject hidden defaults.
- Types express semantic constraints, not storage representation.
- Programs use fixed schemas composed of named input and output fields.
- Variable-length output collections are outside the current program model.
- The output schema determines the fixed scoring dimensions.
- Task, domain, and language come from one shared SURUS ontology registry.
- Benchy defines one universal runtime contract from the declared program schema.
- External AI-systems adapt to that contract at the boundary.
- Integration-specific mechanics do not propagate into benchmark semantics or the engine.

The authoring and execution path is:

```text
Human / Agent / UI
        ↕
       YAML
        ↓
      parse
        ↓
     validate
        ↓
     compile
        ↓
     JSON IR
        ↓
      Engine
        │
        │ runtime contract
        ↓
      Adapter
        ↓
    AI-system
```

The **engine** is Benchy's execution machinery. It consumes the compiled JSON IR, reads exam examples, invokes the AI-system through an adapter, validates outputs, computes scores, and produces results.

The **runtime contract** is the interface rule the engine expects at its boundary:

$$
\boxed{
\text{named-field input object}
\rightarrow
\text{named-field output object}
}
$$

> Every Benchy-compliant AI-system is presented to the engine as something that consumes an object matching the program's named input fields and produces an object matching its named output fields.

An adapter translates between that contract and an AI-system's native interface.

---

## 2. Semantic object model

The program defines what must be done.

$$
\boxed{
\text{Program}
=
\text{Input Schema}
+
\text{Output Schema}
}
$$

> A program consists of one schema describing its inputs and another describing its outputs.

Conceptually:

$$
P:X\rightarrow Y
$$

> Program \(P\) maps values conforming to input schema \(X\) into values conforming to output schema \(Y\).

A schema combines field structure with semantic types:

$$
\boxed{
\text{Schema}
=
\text{Field Structure}
+
\text{Semantic Types}
}
$$

> A schema says which named fields exist, how they are nested, and what each field semantically means.

The semantic type vocabulary is:

```text
string
int
float
bool
enum
date
time
datetime
image
audio
document
```

| Type | Meaning |
|---|---|
| `string` | unconstrained text |
| `int` | integer |
| `float` | real-valued number |
| `bool` | true / false |
| `enum` | value from a closed categorical set |
| `date` | calendar date |
| `time` | time of day |
| `datetime` | date + time |
| `image` | one visual artifact treated as an image |
| `audio` | audio artifact |
| `document` | document artifact that may contain pages, text, images, and layout |

For an enum:

$$
Y=\{v_1,v_2,\ldots,v_k\}
$$

> An enum field may take exactly one value from the finite declared set.

Every input and output value is represented through one or more named fields.

```yaml
program:
  input:
    document: document
  output:
    total: float
```

A classification program can be:

```yaml
program:
  input:
    text: string
  output:
    sentiment:
      enum: [positive, neutral, negative]
```

A structured extraction program can be:

```yaml
program:
  input:
    image: image
  output:
    invoice_number: string
    date: date
    supplier: string
    subtotal: float
    total: float
```

Fixed nested structures are allowed:

```yaml
program:
  input:
    document: document
  output:
    supplier:
      name: string
      tax_id: string
    date: date
    total: float
```

A **leaf output field** is an output field whose value is a semantic type rather than another nested field structure.

The leaf output fields are the scoring dimensions.

$$
\boxed{
\text{fixed named output schema}
\Rightarrow
\text{fixed scoring dimensions}
\Rightarrow
\text{fixed field weights}
}
$$

> Once the output fields are fixed, Benchy knows exactly which dimensions can be scored and weighted.

---

## 3. Benchmark classification and the shared SURUS ontology

Benchmarks are classified with the shared SURUS task-oriented AI ontology:

$$
\boxed{
/\text{task}/\text{domain}/\text{language}/
}
$$

> The ontology identifies what operation is performed, the domain/distribution in which it is evaluated, and the linguistic context of the data.

```yaml
benchmark:
  task: extract
  domain: finance
  language: es
```

The three coordinates have different roles:

- **task** describes the operation performed;
- **domain** describes the world or distribution represented by the exam data;
- **language** describes the linguistic distribution represented by the exam data.

Tasks, domains, and languages live in one shared, versioned registry used by Benchy, DataHub, EvalsHub, and other SURUS systems.

A task defines a family of admissible programs.

Let:

- \(T\) = a task;
- \(\mathcal{P}_T\) = the family of programs admitted by task \(T\);
- \(P\) = a concrete program.

Then:

$$
T\rightarrow\mathcal{P}_T
$$

> Task \(T\) determines the family of programs considered valid instances of that task.

The compiler checks:

$$
\boxed{
P\in\mathcal{P}_T
}
$$

> Program \(P\) must belong to the family admitted by task \(T\).

The program schema alone cannot always determine the task.

$$
\text{string}\rightarrow\text{string}
$$

> The same input/output shape could represent translation, summarization, rewriting, question answering, or another operation.

Task is therefore explicit rather than inferred.

The ontology registry carries shared identifiers and descriptions. Benchy implements the finite task-to-program structural validation rules for the ontology versions it supports. The registry does not contain a separate constraint language.

Translation uses an ordered source/target language relation:

```yaml
benchmark:
  task: translate
  domain: general
  language:
    source: es
    target: en
```

which may be represented externally as:

```text
/translate/general/es-en
```

---

## 4. Scoring

The output schema determines the dimensions over which correctness can be evaluated.

A scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance.

$$
\boxed{
\text{Scoring Function}
=
\text{Field Weights}
+
\text{Aggregator}
}
$$

> Field weights express how important each output dimension is; the aggregator defines how those field scores become one instance score.

```yaml
scoring:
  weights:
    invoice_number: 1
    date: 1
    supplier: 1
    subtotal: 1
    total: 5
  aggregator: weighted_mean
```

Nested weights mirror the output schema:

```yaml
program:
  output:
    supplier:
      name: string
      tax_id: string
    total: float

scoring:
  weights:
    supplier:
      name: 1
      tax_id: 0
    total: 5
  aggregator: weighted_mean
```

Every leaf output field has exactly one explicit weight. There are no missing weights, extra weights, or weights on intermediate objects.

Weights are non-negative:

$$
w_j\geq0
$$

> The weight \(w_j\) of scoring dimension \(j\) may be zero or positive, but never negative.

A zero weight means the field is present and validated but does not affect the instance score.

For exam example \(i\) and scoring dimension \(j\), field correctness is:

$$
c_{ij}
=
\mathbf{1}
[
\hat y_{ij}=y_{ij}^*
]
$$

> Field correctness \(c_{ij}\) is 1 when the predicted semantic value exactly equals the expected semantic value and 0 otherwise.

Field correctness is currently produced by exact match. The evaluator is not exposed as a benchmark configuration option.

All field scores for example \(i\) form:

$$
\mathbf{c}_i
=
(c_{i1},c_{i2},\ldots,c_{in})
$$

> \(\mathbf{c}_i\) contains one correctness value for each scored output dimension of exam example \(i\).

The current aggregator is normalized weighted mean:

$$
s_i
=
\frac{\sum_j w_jc_{ij}}
{\sum_jw_j}
$$

> Instance score \(s_i\) is the weighted average of field correctness values for exam example \(i\).

Therefore:

$$
\sum_jw_j>0
$$

> At least one scoring dimension must have positive weight.

The weights encode **relative importance**. Multiplying every weight by the same positive constant does not change the instance score.

Normalization is a property of this aggregator, not a universal property of Benchy.

$$
\boxed{
\text{The aggregator defines the semantics and scale of the score.}
}
$$

> A normalized aggregator can produce a score in \([0,1]\); future aggregators may naturally use another scale.

---

## 5. Data

The dataset is the exam.

Each exam example \(i\) contains:

$$
(x_i,y_i^*)
$$

> \(x_i\) is the input for example \(i\), and \(y_i^*\) is its expected or ground-truth output.

The program schema constrains both sides:

$$
x_i\in X
$$

> Exam input \(x_i\) must conform to the program's input schema \(X\).

$$
y_i^*\in Y
$$

> Expected output \(y_i^*\) must conform to the program's output schema \(Y\).

Thus:

```text
Input Schema  ───→ exam inputs
Output Schema ───→ expected outputs
Output Schema ───→ scoring dimensions
```

The benchmark definition references the dataset:

```yaml
data:
  path: ./data/invoices.jsonl
```

The program schema defines the semantic structure of every example. The engine specification in Appendix A defines the concrete dataset encoding used by the current implementation.

---

## 6. AI-system

The AI-system is the benchmark taker.

$$
\mathrm{AI}:X\rightarrow Y
$$

> AI-system \(\mathrm{AI}\) implements the declared program contract: it consumes values from \(X\) and produces values intended to conform to \(Y\).

An AI-system may be:

```text
single AI model
AI-node: model + optimized prompt + fixed task behavior
composition of AI models or AI programs
AI components + explicit deterministic code
agent or workflow
```

For a directly specified model:

```yaml
ai-system:
  type: model
  provider: openai
  model: <model>
  prompt: ./prompts/invoice.md
  parameters:
    temperature: 0
```

These fields identify the AI-system under evaluation.

Credentials, SDK construction, HTTP mechanics, response extraction, and similar integration details are outside benchmark semantics.

An arbitrary external AI-system can instead be identified as:

```yaml
ai-system:
  type: external
  id: invoice-extractor-v7
```

The runtime environment binds that identifier to an adapter.

**Every AI-system is executed through the same adapter boundary.** A directly specified model is not a special execution path inside the engine; it is an AI-system for which the runtime may provide a reusable model-provider adapter.

The distinction is:

```text
AI-system definition
= what is being evaluated

adapter/runtime binding
= how this environment invokes it
```

---

## 7. Compilation and execution architecture

The canonical compilation path is:

$$
\boxed{
\text{YAML}
\rightarrow
\text{parse}
\rightarrow
\text{validate}
\rightarrow
\text{compile}
\rightarrow
\text{JSON IR}
\rightarrow
\text{engine}
}
$$

> Benchy parses the canonical source, validates its semantics, compiles it once into a machine-facing representation, and executes from that representation.

There is no normalization stage.

Let:

- \(Y_{\text{yaml}}\) = valid Benchy YAML;
- \(C\) = compiler;
- \(J\) = canonical JSON IR.

Then:

$$
\boxed{
J=C(Y_{\text{yaml}})
}
$$

> Compiler \(C\) deterministically produces JSON IR \(J\) from valid YAML \(Y_{\text{yaml}}\).

Compilation preserves meaning:

$$
\boxed{
\operatorname{Semantics}(J)
=
\operatorname{Semantics}(Y_{\text{yaml}})
}
$$

> JSON IR and YAML express the same benchmark/run semantics; the IR is the executable representation, not another semantic source.

The engine does not reinterpret YAML.

At runtime, the engine knows one AI-system interface:

$$
\boxed{
\text{named-field input object}
\rightarrow
\text{named-field output object}
}
$$

> External AI-system differences are translated at the adapter boundary instead of being spread through the engine.

---

## 8. Execution and results

For exam example \(i\):

$$
\hat y_i=\mathrm{AI}(x_i)
$$

> AI-system \(\mathrm{AI}\) receives input \(x_i\) and produces prediction \(\hat y_i\).

The output is validated against schema \(Y\) before scoring.

A valid but completely incorrect output can have:

```yaml
status: valid
score: 0
```

An output that does not satisfy the declared schema is structurally different:

```yaml
status: invalid_output
score: null
```

An invocation failure is different again:

```yaml
status: execution_error
score: null
```

This distinction preserves the difference between a valid zero score and a failure to produce a valid program output.

For final aggregation, define each example's contribution \(q_i\):

$$
q_i=
\begin{cases}
s_i, & \text{if the AI-system produced a valid output} \\
0, & \text{if the output was invalid or execution failed}
\end{cases}
$$

> Invalid outputs and execution failures retain `null` as their stored instance score, but contribute zero to the benchmark aggregate.

For \(N\) exam examples:

$$
\boxed{
B(\mathrm{AI})
=
\frac{1}{N}
\sum_{i=1}^{N}q_i
}
$$

> The benchmark score is the arithmetic mean across all example contributions, so failed examples remain in the denominator.

---

## 9. Canonical YAML

```yaml
version: "1.0"
ontology_version: "1.0"

benchmark:
  task: extract
  domain: finance
  language: es

program:
  input:
    image: image
  output:
    invoice_number: string
    date: date
    supplier: string
    subtotal: float
    total: float

scoring:
  weights:
    invoice_number: 1
    date: 1
    supplier: 1
    subtotal: 1
    total: 5
  aggregator: weighted_mean

data:
  path: ./data/invoices.jsonl

ai-system:
  type: external
  id: invoice-extractor-v7
```

`version` pins the Benchy semantic specification.

`ontology_version` pins the shared SURUS task/domain/language registry used to interpret the benchmark classification. The benchmark does not contain an ontology file path; during compilation, the compiler resolves the requested registry version from Benchy’s installed/configured ontology registry store. The resulting JSON IR contains the validated classification semantics required by the engine, so execution does not need to reinterpret the ontology.

---

## 10. Scope and future extensions

The current language deliberately does not expose:

```text
variable-length output collections
custom field evaluators
aggregators other than weighted_mean
adapter configuration inside benchmark YAML
implicit field weights
anonymous root scalar inputs or outputs
```

Because field correctness is fixed to exact match (A.7), Engine 1.0 cannot meaningfully score tasks whose outputs are judged by similarity rather than equality. Transcription is the clearest case: a transcript that differs from the reference by one word scores identically to one that is entirely wrong. `transcribe` is therefore not part of the ontology 1.0 task vocabulary, and a benchmark declaring it fails compilation under A.4 rather than producing scores that cannot be interpreted. Appendix E describes the field-evaluator extension that would admit it.

The `audio` semantic type is unaffected and remains a valid input type; only the `transcribe` task is withdrawn.

These are extension points rather than undefined behavior.

The present architecture is intended to let those capabilities be introduced without changing the meaning of existing benchmark definitions.

---

# Appendix A — Engine 1.0 Engineering Contract

This appendix is normative for the first implementation. The main paper defines semantics; this appendix fixes the engineering representation required to build the engine.

## A.1 YAML grammar and strictness

The top-level YAML object contains exactly:

```text
version
ontology_version
benchmark
program
scoring
data
ai-system
```

Unknown top-level keys are errors.

Mappings must not contain duplicate keys.

YAML anchors, aliases, merge keys, and custom tags are rejected.

The program grammar is:

```yaml
program:
  input:
    <field-name>: <type-or-nested-object>
  output:
    <field-name>: <type-or-nested-object>
```

Rules:

- `input` and `output` are mappings;
- each contains at least one named field;
- field names are non-empty strings;
- all fields are required;
- `null` is not a valid field value;
- lists/arrays are not supported as field values;
- nested objects are allowed;
- scalar type declarations must be one of the supported semantic type names;
- enum is declared only as:

```yaml
sentiment:
  enum: [positive, neutral, negative]
```

Enum values are non-empty, distinct strings.

Unknown type declarations are errors.

## A.2 Strict object validation

Program schemas are closed.

For dataset records and AI-system outputs:

```text
missing required field → invalid
extra field            → invalid
wrong semantic type    → invalid
```

Nested objects are validated recursively.

## A.3 Weight coverage

The weight tree must mirror the output schema down to every leaf output field.

There must be:

```text
exactly one explicit weight per leaf output field
no missing weights
no extra weights
no weights on intermediate objects
```

Weights are finite numbers satisfying \(w_j\ge0\), and their total must be positive.

## A.4 Task-to-program validation

The shared ontology registry contains identifiers and descriptions only.

Benchy implements the structural validators for the task vocabulary of the ontology version it supports.

For ontology version `1.0`:

```text
extract
  no task-specific structural constraint beyond normal program rules

classify
  exactly one leaf output field
  that leaf field is enum

translate
  input contains at least one string leaf field
  exactly one leaf output field
  that leaf field is string
  benchmark.language is {source, target}
```

If an ontology task is not supported by the loaded Benchy specification, compilation fails rather than silently skipping task validation.

## A.5 Dataset encoding

The engine uses JSONL.

Each non-empty line is one exam example:

```json
{
  "input": {
    "image": "assets/invoice-001.png"
  },
  "expected": {
    "total": 121.0
  }
}
```

The row contains exactly `input` and `expected`.

Blank lines may be ignored.

The dataset must contain at least one example.

Dataset examples are streamed. They are not embedded into JSON IR.

Each example is validated at execution time against the already-compiled schemas in the IR.

An invalid dataset record is a benchmark/data error and aborts the run. It is not scored as an AI-system failure.

## A.6 JSON representation of semantic values

| Semantic type | Engine 1.0 JSON/runtime representation |
|---|---|
| `string` | string |
| `enum` | string |
| `int` | integer |
| `float` | finite JSON number; integers are valid real values |
| `bool` | boolean |
| `date` | canonical string `YYYY-MM-DD` |
| `time` | canonical string `HH:MM:SS[.fraction]` |
| `datetime` | RFC 3339 string with timezone |
| nested object | JSON object |
| `image` | filesystem path string |
| `audio` | filesystem path string |
| `document` | filesystem path string |

`bool` is not accepted as `int` or `float`.

### Artifact path and workspace rule

In a dataset, `image`, `audio`, and `document` are encoded as relative path strings.

Engine 1.0 executes inside a **benchmark workspace**: a server/local directory containing the benchmark files required by the engine. The storage/orchestration layer may obtain those files from an upload, object storage, or another backend. Assets may be materialized eagerly before the run or lazily before the corresponding example is validated/executed. The engine only requires that an artifact has been materialized into its accessible workspace before that artifact is resolved, validated, or passed to an adapter.

Path bases are explicit:

```text
benchmark-level paths such as data.path and prompt paths
→ resolved relative to the benchmark workspace root

artifact paths inside JSONL
→ resolved relative to the JSONL file's directory
```

The engine:

1. resolves and canonicalizes the path;
2. verifies that the resolved path remains inside the allowed benchmark workspace;
3. rejects path traversal or symlink resolution outside that workspace;
4. verifies that the path exists and is a regular readable file;
5. passes the resolved absolute path string through the runtime contract.

The adapter decides whether to open the file, upload it, decode it, or convert it to another native representation.

For artifact outputs, an adapter returns a local path string to a produced file inside the run's allowed output workspace. The engine validates that path before treating the output as schema-valid.

Filesystem paths are therefore the **Engine 1.0 runtime representation**, not a permanent Benchy semantic invariant. The semantic type remains `image`, `audio`, or `document` regardless of whether a future hosted runtime materializes bytes from local storage, shared storage, or an object store.

## A.7 Exact-match equality

Exact match compares schema-valid semantic values.

| Type | Equality |
|---|---|
| `string` | exact Unicode string equality |
| `enum` | exact declared-member equality |
| `int` | integer equality |
| `float` | finite numeric equality |
| `bool` | boolean equality |
| `date` | parsed calendar-date equality |
| `time` | parsed time-of-day equality |
| `datetime` | parsed instant equality |
| `image` | byte-for-byte file equality |
| `audio` | byte-for-byte file equality |
| `document` | byte-for-byte file equality |

There is no trimming, case folding, Unicode normalization, numeric tolerance, or semantic similarity.

## A.8 Compile-time versus execution-time validation

Compile-time validation covers:

```text
YAML syntax
specification version
ontology version
task/domain/language registry membership
task ↔ program compatibility
program schema grammar
weight coverage and validity
data configuration
AI-system semantic definition
```

Execution-time validation covers:

```text
each dataset input against compiled input schema
each expected value against compiled output schema
each AI-system output against compiled output schema
```

The engine uses schemas already present in the JSON IR. It never reinterprets source YAML.

## A.9 Canonical JSON IR

The IR contains the validated executable semantics needed by the engine, including:

```text
versions
benchmark classification
typed input/output schemas
resolved scoring dimensions
field evaluator
instance aggregator
benchmark aggregator
data configuration
AI-system semantic definition
```

Nested output fields become scoring paths represented as string arrays, for example:

```json
{
  "path": ["supplier", "name"],
  "weight": 1
}
```

Paths are an IR representation detail. Authors do not write them.

The IR includes:

```json
{
  "scoring": {
    "evaluator": "exact_match",
    "instance_aggregator": "weighted_mean",
    "benchmark_aggregator": "mean"
  },
  "data": {
    "format": "jsonl"
  }
}
```

These fields make fixed specification semantics explicit to the engine; they do not introduce new benchmark meaning.

## A.10 Adapter protocol

The engine binds one adapter before execution.

Conceptually:

```python
class Adapter:
    async def invoke(
        self,
        input_object: dict[str, object],
    ) -> dict[str, object]:
        ...
```

The adapter instance may be created with the compiled program contract and AI-system definition.

The engine owns:

```text
dataset loading
input validation
output validation
scoring
aggregation
result/error recording
```

The adapter owns:

```text
mapping Benchy inputs to the native AI-system interface
invoking the AI-system
mapping native outputs back to the Benchy named-field object
```

Adapter exceptions become execution errors.

If the adapter returns an object that violates the output schema, the result is `invalid_output`.

Credentials, endpoint mappings, response paths, Python imports, SDK construction, and other integration mechanics are runtime configuration outside benchmark YAML.

The default engine may execute sequentially. Concurrency is an execution optimization and must not change benchmark semantics.

If examples are executed concurrently, each result retains its dataset index and the serialized `results` array is emitted in dataset order.

Timeouts, concurrency limits, credentials, retries, caching, and similar execution controls are runtime policy rather than benchmark semantics. A runtime timeout becomes an `execution_error`; changing runtime policy must not change the benchmark definition. Non-secret effective runtime settings that can affect observed results should be recorded as run metadata for reproducibility. Secrets such as credentials must never be serialized into benchmark definitions, JSON IR, or result metadata.

## A.11 AI-system execution setup

Every AI-system requires an adapter binding before execution.

For:

```yaml
ai-system:
  type: external
  id: invoice-extractor-v7
```

the runtime binds that identifier to an implementation-specific adapter.

For:

```yaml
ai-system:
  type: model
  provider: ...
  model: ...
```

the runtime may select a reusable provider adapter.

The engine has no separate `model` execution branch. Both forms are invoked through the same adapter protocol.

A missing adapter binding is a run setup error and aborts before evaluating examples. Provider integrations remain outside the core engine.

## A.12 Instance statuses

Every evaluated example has one of:

```text
valid
invalid_output
execution_error
```

`valid` means the AI-system produced a schema-valid output, regardless of score.

`invalid_output` means the adapter returned an output value but it failed the compiled output schema.

`execution_error` means invocation failed before a Benchy output object was successfully produced.

Examples include adapter exceptions, HTTP/provider failures, and runtime timeouts.

## A.13 Scoring failed examples

Stored instance score:

```text
valid            → numeric s_i
invalid_output   → null
execution_error  → null
```

Aggregation contribution:

$$
q_i=
\begin{cases}
s_i, & \text{valid} \\
0, & \text{invalid_output or execution_error}
\end{cases}
$$

> `null` preserves the diagnostic distinction, while zero contribution prevents failed examples from disappearing from the aggregate.

## A.14 Benchmark result schema

A run result contains:

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

A valid example result:

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

An invalid output:

```json
{
  "index": 0,
  "status": "invalid_output",
  "prediction": {
    "debug": "foo"
  },
  "field_scores": null,
  "score": null,
  "contribution": 0.0,
  "error": {
    "code": "missing_field",
    "path": ["total"],
    "message": "required output field is missing"
  }
}
```

An execution error:

```json
{
  "index": 0,
  "status": "execution_error",
  "prediction": null,
  "field_scores": null,
  "score": null,
  "contribution": 0.0,
  "error": {
    "code": "adapter_error",
    "path": null,
    "message": "..."
  }
}
```

## A.15 Dataset failures

Dataset failures are run-level benchmark errors rather than example scores.

Examples:

```text
data file missing
malformed JSONL
row lacks input/expected
input does not conform to input schema
expected output does not conform to output schema
artifact path does not exist
empty dataset
```

The run aborts and does not produce a benchmark score.

## A.16 Error model

Compiler and runtime diagnostics should use:

```text
phase
code
path
message
```

`path` is an array of semantic/object keys when applicable.

Representative codes:

```text
invalid_yaml
duplicate_key
unknown_key
unsupported_version
unsupported_ontology_version
unknown_task
unknown_domain
unknown_language
task_program_mismatch
invalid_schema
missing_weight
extra_weight
invalid_weight
data_not_found
invalid_dataset_record
artifact_not_found
missing_field
extra_field
wrong_type
invalid_enum
invalid_value
adapter_error
timeout
provider_error
```

The exact message text is not part of benchmark semantics.

---

# Appendix B — Shared SURUS Ontology Registry

The ontology is one shared YAML file containing:

```text
tasks
domains
languages
```

Example:

```yaml
version: "1.0"

tasks:
  extract:
    description: Extract named information from an input.
  classify:
    description: Assign one declared categorical label.
  translate:
    description: Transform text from a source language to a target language.

domains:
  general: {}
  finance: {}
  healthcare: {}
  legal: {}
  retail: {}

languages:
  es: {}
  pt: {}
  en: {}
```

Benchy code, not the registry, implements the structural task validators for each supported ontology version.

This avoids prematurely introducing a task-constraint DSL while preserving:

$$
P\in\mathcal{P}_T
$$

> The declared program must satisfy the structural semantics associated with its task.

---

# Appendix C — Artifact Runtime Representation

Two designs were considered for `image`, `audio`, and `document`.

### Typed artifact object

The engine could convert a dataset path into an internal object such as:

```text
Artifact(type=image, path=...)
```

This gives the runtime value an explicit host-language wrapper, but introduces a new abstraction that every adapter and language binding must understand.

### Path string

The dataset may keep the artifact as a path string and let the adapter decide how to consume it.

This is simpler and fits the adapter boundary: one adapter may open the file, another may upload it over HTTP, and another may decode it into a provider-specific object.

The first implementation therefore uses **resolved local filesystem path strings** at the runtime boundary. In hosted deployments, an orchestration/storage layer materializes uploaded or object-stored assets into the engine's benchmark workspace either eagerly or on demand, as long as each artifact is locally available before the engine validates or executes the corresponding example.

The semantic meaning still comes from the program schema:

```yaml
image: image
```

not from the host-language type of the value.

A future runtime representation may replace path strings with a richer artifact handle without changing the benchmark's semantic type system.

---

# Appendix D — Compact Architecture

```text
               shared SURUS ontology
                        │
                        ▼
Human / Agent / UI ↔ canonical YAML
                        │
                        ▼
                      parse
                        │
                        ▼
                    validate
                        │
                        ▼
                     compile
                        │
                        ▼
                 canonical JSON IR
                        │
                        ▼
                      Engine
                        │
                 runtime contract
                        │
                        ▼
                     Adapter
                        │
                        ▼
                   AI-system
```

The central rule is:

> **Benchmark meaning is defined once in the canonical YAML, compiled once into JSON IR, and executed through one runtime contract.**

---

# Appendix E — Field evaluators (extension, not Engine 1.0)

Engine 1.0 fixes field correctness to exact match (A.7). That is sufficient wherever a field is either the declared value or not — extraction and classification — and insufficient wherever the output is judged by similarity. Transcription is the motivating case, and the reason `transcribe` is absent from the ontology 1.0 task vocabulary.

The extension is a per-field evaluator declared in the scoring section, defaulting to `exact_match` so that every existing benchmark keeps its current meaning:

```yaml
scoring:
  weights:
    transcription: 1
  evaluators:
    transcription: wer
  aggregator: weighted_mean
```

An evaluator maps a predicted and an expected field value to a field score in \([0,1]\), occupying the slot A.7 currently fills with a fixed equality test.

$$
\boxed{
\text{evaluator}:(\hat y_{ij},\,y_{ij}^*)\rightarrow c_{ij}\in[0,1]
}
$$

> An evaluator produces the field correctness value \(c_{ij}\) directly, rather than the indicator that exact match produces.

The instance-score and aggregation semantics of §4 are unchanged: evaluators produce \(c_{ij}\), and the weighted mean consumes it exactly as before. Exact match remains the case where the evaluator's range is \(\{0,1\}\).

## Error-rate metrics must be inverted and bounded

Word Error Rate and Character Error Rate are error rates: lower is better, and they are not bounded above — WER exceeds 1.0 when a prediction contains more insertions than the reference has words. A scoring dimension must be higher-is-better and confined to \([0,1]\), so:

$$
\boxed{
c_{ij}=\operatorname{clamp}\left(1-r_{ij},\,0,\,1\right)
}
$$

> Field score \(c_{ij}\) is one minus the raw error rate \(r_{ij}\), clamped to \([0,1]\).

The raw, uninverted rate should be preserved alongside the score in the result artifact, since the raw rate is the number practitioners report and compare.

Preserving the raw numerator and denominator rather than only the ratio also keeps both corpus statistics available. The mean of per-example scores is a macro-average; the error rate customarily reported in the literature is a micro-average over total edits and total reference length. They are different numbers, and only one is comparable to published results.

## Candidate evaluators

`exact_match` (the Engine 1.0 default), `wer`, `cer`.

Whether normalization — case folding, punctuation stripping, number expansion — belongs inside an evaluator or is declared separately is deliberately left open here. It is not a small effect: measured over stored ASR predictions in this repository, per-example word accuracy differs by roughly 9 to 15 points between the raw and normalized forms, because an unnormalized comparison charges a model for emitting correct capitalization and punctuation against a reference that carries neither. The decision changes reported scores substantially and deserves its own treatment.

## Restoring `transcribe`

Introducing evaluators would restore `transcribe` to the ontology task vocabulary, with the A.4 structural constraints already drafted for it:

```text
transcribe
  input contains at least one audio leaf field
  exactly one leaf output field
  that leaf field is string
```

The `audio` semantic type is already present in Engine 1.0 and requires no change.
