# Benchy B1 — Technical Design

## 1. Objective

Benchy is a universal benchmark-definition and execution engine for AI programs.

The unit of evaluation is an **AI system implementing a program**, not a model.

A system may be a model, model + prompt, fine-tuned model, workflow, agent, composition of models, deterministic pipeline, or arbitrary external endpoint.

Benchy only requires that the implementation satisfy the declared program contract.

$$
M:X\rightarrow Y
$$

```text
named input record
      ↓
   AI system
      ↓
named output record
```

Benchy separates two semantic objects:

$$
\boxed{B=(P,S,D)}
$$

where:

- \(B\) = benchmark,
- \(P\) = program,
- \(S\) = scoring function,
- \(D\) = exam data.

A benchmark run adds the system:

$$
\boxed{R=(B,M)}
$$

where \(M\) is the AI system taking the benchmark.

One YAML document contains both so the complete run definition is self-contained.

---

# 2. Design Principles

## 2.1 YAML is the canonical semantic source

Humans edit YAML.  
Agents edit YAML.  
The UI edits YAML.  
Version control stores YAML.

> **The YAML is the single canonical semantic definition. Everything else is derived machinery.**

The execution path is:

```text
YAML
  ↓
parse
  ↓
validate
  ↓
compile
  ↓
canonical JSON IR
  ↓
engine
```

There is no normalization layer.

Benchy defines one valid YAML syntax for each semantic concept. Invalid or incomplete YAML is rejected instead of repaired, normalized, or silently completed.

> **Compile ≠ repair.**

Compilation may change representation. It may not change meaning.

## 2.2 Explicit is better than implicit

Canonical YAML should expose benchmark-specific semantics explicitly.

Bad:

```yaml
scoring:
  aggregator: weighted_mean
```

Better:

```yaml
scoring:
  weights:
    invoice_number: 1
    date: 1
    total: 1
  aggregator: weighted_mean
```

The UI may help the user choose defaults, but the serialized YAML writes them explicitly.

Explicitness should not create redundancy.

## 2.3 The YAML should be self-sufficient

A reader should be able to inspect the YAML and determine:

1. what benchmark is being defined;
2. what program is evaluated;
3. how it is scored;
4. what exam data is used;
5. what system takes the benchmark.

Universal language semantics such as `float`, `enum`, or the definition of a registered task belong to the versioned Benchy/SURUS specification.

## 2.4 Nothing about the runtime should leak into benchmark semantics unless necessary

```text
Authoring semantics           Execution mechanics

Human / Agent / UI
        ↕
       YAML
        ↓
      compiler
        ↓
     JSON IR
        ↓
   Benchy engine
        ↓
 universal runtime boundary
        ↓
      adapter
        ↓
 external implementation
```

Python classes, Rust structs, HTTP payload conventions, provider SDK objects, and vendor response formats do not define benchmark semantics.

## 2.5 Expressivity through a very small vocabulary

> **Types express semantic constraints, not storage representation.**

Avoid task-specific classes such as `InvoiceInput`, `InvoiceOutput`, or `TranscriptionResult`.

Prefer universal field structures and semantic types.

## 2.6 Avoid complexity through oversimplification

Simplification must not remove information required to understand or reproduce the benchmark.

Examples:

- omitted equal weights are too implicit;
- classification as arbitrary `string` loses the closed label set;
- anonymous root scalar inputs/outputs lose their semantic names;
- variable-length output collections make scoring dimensions variable.

## 2.7 B1 invariant: fixed named schemas

Every B1 program has:

- one or more **named input fields**;
- one or more **named output fields**;
- a **fixed output schema**.

Therefore:

$$
\boxed{
\text{fixed named output schema}
\Rightarrow
\text{fixed scoring dimensions}
\Rightarrow
\text{fixed weights}
\Rightarrow
\text{simple scoring}
}
$$

Anonymous scalar roots are unsupported.

Variable-length output collections are unsupported.

---

# 3. Benchmark Classification

SURUS uses the task-oriented AI ontology:

$$
\boxed{
/\text{task}/\text{domain}/\text{language}/
}
$$

A benchmark declares its coordinates explicitly:

```yaml
benchmark:
  task: extract
  domain: finance
  language: es
```

- **task** — operation performed;
- **domain** — world/distribution represented by the exam;
- **language** — linguistic distribution represented by the exam.

`task`, `domain`, and `language` are controlled identifiers.

---

# 4. Task Registry

A task is explicit but not free-form.

Let:

- \(T\) = a task;
- \(\mathcal{P}_T\) = the set of programs admitted by task \(T\);
- \(P\) = one concrete program.

Then:

$$
T\rightarrow\mathcal{P}_T
$$

and the compiler checks:

$$
\boxed{
P\in\mathcal{P}_T
}
$$

Example task registry:

```yaml
tasks:
  extract:
    input: any
    output: structured

  classify:
    input: any
    output: enum

  transcribe:
    input: audio
    output: string

  translate:
    input: string
    output: string
```

The registry is canonical and versioned outside individual benchmark definitions and shared by Benchy, DataHub, EvalsHub, and related SURUS systems.

Domain and language are also validated against controlled registries.

---

# 5. Four Structural Pillars

The user-facing flow is:

```text
1. Program
2. Scoring Function
3. Data
4. System
```

The first three define the benchmark:

$$
B=(P,S,D)
$$

The fourth defines the benchmark taker:

$$
R=(B,M)
$$

The YAML contains all four because it defines a complete reproducible run.

---

# 6. Pillar 1 — Program

The program defines **what must be done**.

$$
\boxed{
\text{Program}
=
\text{Input Schema}
+
\text{Output Schema}
}
$$

$$
P:X\rightarrow Y
$$

A schema is:

$$
\boxed{
\text{Schema}
=
\text{Field Structure}
+
\text{Semantic Types}
}
$$

## 6.1 B1 semantic types

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

## 6.2 Every field is named

Transcription:

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

Conceptually:

$$
P:\text{audio}\rightarrow\text{string}
$$

but Benchy names the semantic dimensions `audio` and `transcription`.

Classification:

```yaml
program:
  input:
    text: string
  output:
    sentiment:
      enum: [positive, neutral, negative]
```

Extraction:

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

Fixed hierarchy is allowed:

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

The hierarchy may be nested, but the set of output scoring dimensions is fixed.

---

# 7. Pillar 2 — Scoring Function

> **For structured outputs, a scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance.**

B1 defines:

$$
\boxed{
\text{Scoring Function}
=
\text{Field Weights}
+
\text{Aggregator}
}
$$

Example:

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

Weights define importance, not correctness.

For nested outputs, weights mirror the output structure:

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
      tax_id: 1
    total: 5
  aggregator: weighted_mean
```

Internally the compiler may derive canonical scoring dimensions such as:

```text
supplier.name
supplier.tax_id
total
```

but users do not need to author dotted paths.

Weights satisfy:

$$
w_j\geq0
$$

A weight of `0` means the field is part of the program output but does not affect the score.

Negative weights are invalid.

For weighted mean:

$$
\sum_jw_j>0
$$

Scoring is constrained by the output schema:

```text
Output Schema ───→ Scoring Validation
```

A weight may only refer to a declared output dimension.

---

# 8. Pillar 3 — Data

The data is the exam.

Each example \(i\) is:

$$
(x_i,y_i^*)
$$

where:

- \(i\) = exam-example index;
- \(x_i\) = input for example \(i\);
- \(y_i^*\) = expected/ground-truth output.

The input schema constrains the semantic shape and types of \(x_i\):

$$
x_i\in X
$$

The output schema constrains the semantic shape and types of \(y_i^*\):

$$
y_i^*\in Y
$$

```text
Input Schema  ───→ exam input shape
Output Schema ───→ expected-output shape
Output Schema ───→ scoring dimensions
```

YAML only references the dataset:

```yaml
data:
  path: ./data/invoices.jsonl
```

B1 should define one canonical physical dataset format separately from the program schema.

Synthetic data must satisfy the same program contract.

---

# 9. Pillar 4 — System

The system is the benchmark taker.

$$
M:X\rightarrow Y
$$

The user may configure a direct model:

```yaml
system:
  type: model
  provider: openai
  model: <model>
  prompt: ./prompts/invoice.md
  parameters:
    temperature: 0
```

or an arbitrary external implementation:

```yaml
system:
  type: endpoint
  endpoint: https://example.com/extract
```

`provider`, `model`, `prompt`, and inference parameters are intentionally runtime-facing and belong under `system`.

They do not change program, scoring, or data semantics.

---

# 10. Canonical Compilation Model

YAML is the only authored semantic source.

The compiler pipeline is:

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

There is no normalization step.

## 10.1 Validation

Validation rejects invalid or incomplete definitions.

It checks at least:

```text
syntax
  ↓
benchmark identifiers
  ↓
task ↔ program compatibility
  ↓
program ↔ scoring compatibility
  ↓
program ↔ data compatibility
  ↓
system configuration
```

Formally:

$$
P\in\mathcal{P}_T
$$

$$
S\in\mathcal{S}_P
$$

$$
D\in\mathcal{D}_P
$$

where:

- \(\mathcal{P}_T\) = programs admitted by task \(T\);
- \(\mathcal{S}_P\) = scoring functions valid for program \(P\);
- \(\mathcal{D}_P\) = datasets conforming to program \(P\).

## 10.2 Compilation

Let:

- \(Y\) = valid Benchy YAML;
- \(C\) = compiler;
- \(J\) = canonical JSON IR.

Then:

$$
\boxed{
J=C(Y)
}
$$

and:

$$
\boxed{
\operatorname{Semantics}(J)
=
\operatorname{Semantics}(Y)
}
$$

The compiler may:

- resolve the task definition from the registry;
- validate semantic types;
- derive canonical field/scoring addresses;
- produce a machine-facing representation.

The compiler may not:

- repair invalid YAML;
- infer omitted benchmark-specific semantics;
- insert hidden defaults;
- reinterpret ambiguous source.

The JSON IR is **derived executable state**, not a second source of truth.

## 10.3 Example JSON IR

From:

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string

scoring:
  weights:
    transcription: 1
  aggregator: weighted_mean
```

the compiler may produce an executable representation such as:

```json
{
  "program": {
    "input": {
      "audio": {"type": "audio"}
    },
    "output": {
      "transcription": {"type": "string"}
    }
  },
  "scoring": {
    "dimensions": [
      {
        "path": ["transcription"],
        "weight": 1
      }
    ],
    "aggregator": "weighted_mean"
  }
}
```

This structure is illustrative. The IR schema is machine-facing and may evolve as long as compilation is deterministic and semantics-preserving.

---

# 11. Universal Runtime Boundary

Benchy exposes one canonical semantic runtime contract.

$$
\boxed{
\text{named inputs}
\rightarrow
\text{named outputs}
}
$$

The contract is derived directly from the declared program schema.

For:

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

the Benchy boundary receives:

```json
{
  "audio": "<audio-value>"
}
```

and expects:

```json
{
  "transcription": "hello world"
}
```

This gives the invariant:

$$
\boxed{
\text{Every program executed by Benchy consumes and produces values conforming to its declared program schema.}
}
$$

## 11.1 Adapters

External implementations adapt to this boundary.

A native function may be:

```python
def transcribe(audio) -> str:
    ...
```

Its adapter performs:

```text
Benchy input
{"audio": ...}
      ↓
adapter
      ↓
native input
audio
      ↓
implementation
      ↓
native output
"hello world"
      ↓
adapter
      ↓
Benchy output
{"transcription": "hello world"}
```

The Benchy core never needs to understand the native calling convention.

## 11.2 Hub-and-spoke

If \(I_k\) is external implementation \(k\), \(A_k\) its adapter, and \(B_R\) the Benchy runtime boundary:

$$
I_k\leftrightarrow A_k\leftrightarrow B_R
$$

for:

$$
k=1,\ldots,n
$$

Each new implementation requires one adapter.

Integration complexity remains localized instead of propagating through engine internals.

## 11.3 Semantic interface vs transport

The universal boundary does not require one transport.

An implementation may run:

```text
in-process
HTTP
SDK
model provider
container
future executor
```

Those are transport mechanics.

The semantic contract remains:

$$
\boxed{
\text{named input record}
\rightarrow
\text{named output record}
}
$$

---

# 12. Benchmark Execution

For exam example \(i\):

$$
(x_i,y_i^*)
$$

the system produces:

$$
\hat y_i=M(x_i)
$$

where \(\hat y_i\) is the predicted output for example \(i\).

The universal runtime boundary validates \(\hat y_i\) against the declared output schema before scoring.

## 12.1 Field correctness

Let:

- \(j\) = output-field/scoring-dimension index;
- \(c_{ij}\) = correctness of field \(j\) on example \(i\).

B1 uses exact match:

$$
c_{ij}
=
\mathbf{1}
[
\hat y_{ij}=y_{ij}^*
]
$$

All field scores for example \(i\) form the correctness vector:

$$
\mathbf{c}_i
=
(c_{i1},c_{i2},\ldots,c_{in})
$$

Field scores and correctness vectors are runtime outputs, not user-authored benchmark configuration.

Exact match means equality of **schema-valid semantic values**, not equality of arbitrary serialized bytes.

## 12.2 Instance score

Let:

- \(w=(w_1,\ldots,w_n)\) = fixed field-weight vector;
- \(A\) = selected aggregator;
- \(s_i\) = score for exam example \(i\).

Then:

$$
\boxed{
s_i=A(\mathbf{c}_i,w)
}
$$

For weighted mean:

$$
A(\mathbf{c}_i,w)
=
\frac{\sum_jw_jc_{ij}}
{\sum_jw_j}
$$

## 12.3 Benchmark score

Let \(N\) be the number of exam examples.

B1 currently defines the final benchmark score as arithmetic mean across instance scores:

$$
\boxed{
B(M)
=
\frac{1}{N}
\sum_{i=1}^{N}s_i
}
$$

This is a B1 protocol rule unless later made configurable.

---

# 13. Canonical B1 YAML

```yaml
version: b1

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

system:
  type: model
  provider: openai
  model: <model>
  prompt: ./prompts/invoice.md
  parameters:
    temperature: 0
```

The UI exposes:

```text
Benchmark metadata
Task / Domain / Language

1. Program
2. Scoring Function
3. Data
4. System
```

Changing the UI changes YAML.  
Changing YAML changes the run definition.  
The compiler deterministically produces the JSON IR consumed by the engine.

---

# 14. B1 Open Decisions / Limitations

## 14.1 Field evaluators

B1 supports exact match only.

Future versions may support:

```text
normalized exact match
numeric tolerance
set match
semantic similarity
LLM-as-judge
task-specific evaluators
```

These extend field correctness without changing the benchmark/program model.

## 14.2 Variable-length outputs

Variable-length collections are unsupported in B1.

They introduce variable evaluation dimensions and require semantics for:

```text
matching
ordering
missing elements
extra elements
weights
aggregation
```

## 14.3 Aggregator semantics

`weighted_mean` is fully defined.

Other aggregators such as:

```text
sum
median
min
max
```

must not be considered supported until Benchy defines exactly how each interprets field weights and what output scale it produces.

## 14.4 Invalid runtime outputs

B1 must define the behavior when a runtime output does not satisfy the output schema.

Candidate policy:

```text
invalid output
    ↓
record validation error
    ↓
instance score = 0
    ↓
continue benchmark
```

This remains to be finalized.

## 14.5 Dataset physical format

The program schema determines the semantic shape of each example.

A separate B1 data-format rule must determine how examples and file-valued fields are physically encoded.

## 14.6 Multilingual ontology

Single-language tasks fit:

```text
/extract/finance/es
```

Translation needs an ordered source/target language relation.

A candidate representation is:

```yaml
language:
  source: es
  target: en
```

while rendering an ontology coordinate such as:

```text
/translate/general/es-en
```

This remains ontology design work.

---

# Appendix A — SURUS Task-Oriented AI Ontology

SURUS uses one shared ontology across Benchy, DataHub, EvalsHub, and related task-oriented AI systems:

$$
\boxed{
/\text{task}/\text{domain}/\text{language}/
}
$$

## A.1 Task

Task describes the operation performed.

Examples:

```text
extract
classify
transcribe
translate
summarize
```

Task is explicit and comes from a controlled registry.

The task does not equal the program schema.

The same schema may represent different tasks.

For example:

$$
\text{string}\rightarrow\text{string}
$$

could describe translation, summarization, rewriting, or question answering.

Therefore task \(T\) defines a family of valid programs \(\mathcal{P}_T\), and Benchy validates:

$$
P\in\mathcal{P}_T
$$

## A.2 Domain

Domain describes the world/distribution represented by the benchmark data.

Examples:

```text
general
finance
healthcare
legal
retail
```

The compiler verifies that the declared domain is a valid identifier in the domain registry.

Domain primarily characterizes the data distribution; it does not define program structure.

## A.3 Language

Language describes the linguistic distribution represented by the benchmark data.

Examples:

```text
es
pt
en
```

The compiler verifies that the declared language identifier is registered.

The compiler does not generally prove that arbitrary content is linguistically or semantically faithful to that metadata.

## A.4 Relationship across SURUS systems

The same ontology coordinates identify compatible artifacts across SURUS:

```text
DataHub
  /extract/finance/es
      datasets

EvalsHub
  /extract/finance/es
      evaluations

Benchy
  /extract/finance/es
      benchmark definitions + runs
```

The ontology is shared semantic infrastructure.

Its purpose is interoperability: the same task, domain, and language identifiers mean the same thing across the SURUS ecosystem.
