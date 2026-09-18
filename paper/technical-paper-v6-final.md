# Benchy B1 — Technical Design v6

## 1. Objective

Benchy is a universal benchmark-definition and execution engine for AI programs.

The benchmark taker is an **AI-system**, not necessarily a single model.

An **AI-system** is any AI-based program implementation. It may be:

- a single AI model;
- an **AI-node**, such as a model with an optimized prompt and fixed behavior;
- a composition of multiple AI models or AI programs;
- AI components intermingled with explicit or deterministic code;
- an agent or workflow built from any combination of the above.

<<<dont love this list, should rewrite it to single ai model, ai node, workflow, agent, or any composition of these. >>>

The defining property is that it is an AI-based implementation of the declared program, but might be a single ai program or a composition of programs

Benchy separates the benchmark from the AI-system taking it.

Let:

- \(P\) = program,
- \(S\) = scoring function,
- \(D\) = exam data,
- \(B\) = benchmark,
- \(M\) = AI-system,
- \(R\) = benchmark run.

Then:

$$
\boxed{B=(P,S,D)}
$$

and:

$$
\boxed{R=(B,M)}
$$

The same benchmark can therefore evaluate many AI-systems:

$$
B(M_1),\;B(M_2),\;\ldots,\;B(M_n)
$$

The long-term objective is a small semantic language for defining AI-program benchmarks that humans and agents can author directly, while remaining independent of programming language, framework, provider, and transport.

---

# 2. Design Principles

## 2.1 YAML is the canonical semantic source

Humans edit YAML.  
Agents edit YAML.  
The UI edits YAML.  
Version control stores YAML.

> **The YAML is the canonical Benchy definition. Everything downstream is derived machinery.**

The authoring/execution path is:

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
Benchy core
```

There is no normalization layer.

Benchy defines **one valid YAML syntax per semantic concept**. Invalid or incomplete source is rejected instead of repaired, normalized, or silently completed.

> **Compile ≠ repair.**

## 2.2 Explicit is better than implicit

Benchmark-specific semantics should be visible in the YAML.

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

The UI may help the user choose defaults. The serialized definition makes those choices explicit.

Explicitness should not create redundancy.

## 2.3 The YAML should be self-sufficient

A reader should be able to determine:

1. what benchmark is being defined;
2. what program is evaluated;
3. how it is scored;
4. what exam data is used;
5. what AI-AI-system is taking it.

Universal language semantics such as `float`, `enum`, and registered task meanings belong to the versioned Benchy/SURUS specification rather than being redefined in every file.

## 2.4 Nothing about runtime mechanics should leak into benchmark semantics unless necessary

The benchmark definition describes **what** is evaluated.

Adapters and transports describe **how** a concrete implementation is invoked.

```text
Human / Agent / UI
        ↕
       YAML
        ↓
     compiler
        ↓
     JSON IR
        ↓
   Benchy core
        ↓
 Benchy contract
        ↓
      adapter
        ↓
external AI-system implementation
```

Python signatures, endpoint request mappings, response JSON paths, SDK objects, credentials, and transport-specific behavior terminate at the adapter boundary.

## 2.5 Expressivity through a very small vocabulary

> **Types express semantic constraints, not storage representation.**

Avoid task-specific classes such as:

```text
InvoiceInput
InvoiceOutput
TranscriptionResult
```

Prefer universal field structures and semantic types.

## 2.6 Avoid complexity through oversimplification

Simplification must not remove information required to understand or reproduce the benchmark.

Examples:

- omitted equal weights are too implicit;
- classification as arbitrary `string` loses the closed label set;
- anonymous scalar roots lose semantic names;
- variable-length output collections make evaluation dimensions variable.

## 2.7 Terminology invariant: AI-system

Benchy uses **AI-system** for the benchmark taker.

It does not use `system` as shorthand in the semantic model.

An AI-system is any AI-based implementation of a program, including:

```text
single AI model
AI-node = model + optimized prompt + fixed task behavior
composition of AI models / AI programs
AI components + explicit deterministic code
agent or workflow
```

This keeps the abstraction broader than `model` while still requiring the implementation to be AI-based.

## 2.8 B1 invariant: fixed named schemas

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
\text{fixed field weights}
\Rightarrow
\text{simple scoring}
}
$$

Anonymous root scalar inputs and outputs are unsupported.

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

All three values come from controlled registries.

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

and Benchy validates:

$$
\boxed{
P\in\mathcal{P}_T
}
$$

Example registry:

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

The registry is versioned outside individual benchmark definitions and shared by Benchy, DataHub, EvalsHub, and related SURUS systems.

Domain and language identifiers are validated against their own registries.

---

# 5. Four User-Facing Steps

The Benchy UI presents four steps:

```text
1. Program
2. Scoring Function
3. Data
4. AI-System
```

Semantically:

```text
Program + Scoring + Data = Benchmark
Benchmark + AI-System    = Run
```

The first three define:

$$
B=(P,S,D)
$$

The fourth binds the AI-AI-system taking the benchmark:

$$
R=(B,M)
$$

The distinction matters because the same benchmark can be reused across AI-systems.

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

Conceptually:

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

The conceptual task remains:

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

The hierarchy may be nested, but the set of output scoring dimensions remains fixed.

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

The compiler may derive canonical machine-facing addresses such as:

```text
supplier.name
supplier.tax_id
total
```

but the author does not need to write path syntax.

Weights satisfy:

$$
w_j\geq0
$$

where \(w_j\) is the weight of scoring dimension \(j\).

A zero weight means the field remains part of the program output but does not affect the score.

Negative weights are invalid.

For weighted mean:

$$
\sum_j w_j>0
$$

Scoring is constrained specifically by the output schema:

```text
Output Schema ───→ Scoring Validation
```

A scoring weight may only refer to a declared output dimension.

---

# 8. Pillar 3 — Data

The data is the exam.

Each exam example \(i\) contains:

$$
(x_i,y_i^*)
$$

where:

- \(i\) = exam-example index;
- \(x_i\) = input for example \(i\);
- \(y_i^*\) = expected/ground-truth output.

The input schema constrains the shape and semantic types of \(x_i\):

$$
x_i\in X
$$

The output schema constrains the shape and semantic types of \(y_i^*\):

$$
y_i^*\in Y
$$

```text
Input Schema  ───→ exam input
Output Schema ───→ expected output
Output Schema ───→ scoring dimensions
```

The YAML references the dataset:

```yaml
data:
  path: ./data/invoices.jsonl
```

The program schema determines the **semantic shape** of each example.

A separate B1 data-format specification determines its **physical encoding**.

Synthetic data generation must obey the same program contract.

---

# 9. Pillar 4 — AI-System

The AI-system is the benchmark taker.

$$
M:X\rightarrow Y
$$

The AI-system definition identifies **what AI-based implementation is being evaluated**.

For a direct model, a semantic AI-AI-system specification may include:

```yaml
ai-system:
  type: model
  provider: openai
  model: <model>
  prompt: ./prompts/invoice.md
  parameters:
    temperature: 0
```

These values define the AI-AI-system under test.

They do **not** define adapter mechanics such as credentials, SDK object construction, HTTP request mappings, or response extraction.

For an arbitrary external AI-system implementation, the semantic definition should identify the AI-system without embedding transport mechanics:

```yaml
ai-system:
  type: external
  id: invoice-extractor-v7
```

The runtime environment binds that AI-system identifier to an adapter.

This preserves both goals:

- the run definition says **what AI-system is being benchmarked**;
- integration mechanics remain outside benchmark semantics.

---

# 10. Canonical Compilation Model

The canonical path is:

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
\text{Benchy core}
}
$$

There is no normalization step.

## 10.1 Validation

Validation rejects invalid or incomplete source.

It checks:

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
AI-system semantic validity
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

with the invariant:

$$
\boxed{
\operatorname{Semantics}(J)
=
\operatorname{Semantics}(Y)
}
$$

The compiler may:

- resolve registered semantic definitions;
- validate types;
- derive canonical field/scoring addresses;
- create a typed machine-facing executable representation.

The compiler may not:

- repair invalid source;
- infer missing benchmark-specific semantics;
- inject hidden defaults;
- silently reinterpret source.

The JSON IR is **derived executable state**, not a second source of truth.

## 10.3 Illustrative JSON IR

Input YAML:

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

Possible compiled representation:

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

The exact IR schema is machine-facing. It may evolve as long as compilation remains deterministic and semantics-preserving.

---

# 11. Universal Runtime Contract

> **Benchy core knows one interface. Everything else adapts to it at the boundary.**

The universal element is the **contract**, not a universal adapter.

The contract is derived from the program's named input and output schemas.

$$
\boxed{
\text{named-field input object}
\rightarrow
\text{named-field output object}
}
$$

For:

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

the Benchy boundary consumes:

```json
{
  "audio": "<audio-value>"
}
```

and receives:

```json
{
  "transcription": "hello world"
}
```

A Benchy-compliant program therefore satisfies:

$$
\boxed{
\text{input value conforms to }X
\quad\land\quad
\text{output value conforms to }Y
}
$$

---

# 12. Adapter Boundary

An adapter is the architectural primitive that translates between:

1. an external AI-system implementation's native interface;
2. the Benchy runtime contract.

$$
\boxed{
\text{external interface}
\leftrightarrow
\text{adapter}
\leftrightarrow
\text{Benchy contract}
}
$$

Suppose an external Python function is:

```python
def transcribe(audio) -> str:
    ...
```

The adapter performs:

```text
Benchy input
{"audio": ...}
      ↓
adapter
      ↓
native call
transcribe(audio)
      ↓
native output
"hello world"
      ↓
adapter
      ↓
Benchy output
{"transcription": "hello world"}
```

The Benchy core does not need to know that the native implementation used:

```text
a positional argument
a raw-string return
different field names
HTTP
an SDK object
a provider-specific response
```

Those differences terminate at the boundary.

## 12.1 One contract, many adapters

Let:

- \(I_k\) = external AI-system implementation \(k\);
- \(A_k\) = adapter for implementation \(k\);
- \(B_R\) = Benchy runtime contract.

Then:

$$
I_k\leftrightarrow A_k\leftrightarrow B_R
\qquad
k=1,\ldots,n
$$

Conceptually:

```text
Implementation A ─ Adapter A ─┐
Implementation B ─ Adapter B ─┤
Implementation C ─ Adapter C ─┼── Benchy contract ── Core
Implementation D ─ Adapter D ─┤
Implementation N ─ Adapter N ─┘
```

This is a hub-and-spoke topology rather than a many-to-many mapping problem inside the engine.

## 12.2 Generic adapters are conveniences, not semantics

Benchy may later provide reusable:

```text
HTTP adapters
Python-function adapters
model-provider adapters
SDK helpers
agent skills
```

These are engineering conveniences built on top of the adapter primitive.

B1 does not need an adapter taxonomy or adapter DSL.

## 12.3 Adapter configuration is outside benchmark semantics

The canonical semantic definition should not contain integration mappings such as:

```text
endpoint request field mappings
response JSON paths
Python import locations
credentials
SDK construction
transport-specific configuration
```

Those belong to the runtime/integration layer.

This preserves:

$$
\boxed{
\text{benchmark semantics}
\neq
\text{integration mechanics}
}
$$

---

# 13. Execution

For exam example \(i\):

$$
(x_i,y_i^*)
$$

the AI-system produces:

$$
\hat y_i=M(x_i)
$$

where \(\hat y_i\) is the predicted output.

The adapter first maps native output into the Benchy named-field output object.

The runtime boundary validates that object against the output schema \(Y\).

Only schema-valid values proceed to field scoring.

## 13.1 Field correctness

Let:

- \(j\) = output scoring-dimension index;
- \(c_{ij}\) = correctness of field \(j\) on exam example \(i\).

B1 uses exact match:

$$
c_{ij}
=
\mathbf{1}
[
\hat y_{ij}=y_{ij}^*
]
$$

All field scores for example \(i\) form:

$$
\mathbf{c}_i
=
(c_{i1},c_{i2},\ldots,c_{in})
$$

Exact match means equality of **schema-valid semantic values**, not arbitrary serialized bytes.

Field scores and correctness vectors are runtime outputs, not benchmark-definition inputs.

## 13.2 Instance score

Let:

- \(w=(w_1,\ldots,w_n)\) = fixed weight vector;
- \(A\) = scoring aggregator;
- \(s_i\) = score for exam example \(i\).

Then:

$$
\boxed{
s_i=A(\mathbf{c}_i,w)
}
$$

B1 currently defines `weighted_mean`:

$$
A(\mathbf{c}_i,w)
=
\frac{\sum_j w_jc_{ij}}
{\sum_jw_j}
$$

## 13.3 Benchmark score

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

# 14. Canonical B1 YAML

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

ai-system:
  type: model
  provider: openai
  model: <model>
  prompt: ./prompts/invoice.md
  parameters:
    temperature: 0
```

The `ai-system` section identifies the benchmark taker.

Runtime adapter mechanics remain outside the canonical semantic definition.

The UI exposes:

```text
Benchmark metadata
Task / Domain / Language

1. Program
2. Scoring Function
3. Data
4. AI-System
```

Changing the UI changes YAML.  
Changing YAML changes the semantic run definition.  
Compilation deterministically produces the JSON IR consumed by Benchy's core.

---

# 15. Semantic Dependency DAG

The core dependency structure is:

```text
                         Task
                          ↓
                       Program
                    /           \
                   ↓             ↓
            Input Schema     Output Schema
                 ↓             /        \
                 ↓            ↓          ↓
           Exam inputs   Expected data  Scoring
```

More precisely:

$$
\text{Input Schema}
\rightarrow
\text{valid }x_i
$$

$$
\text{Output Schema}
\rightarrow
\text{valid }y_i^*
$$

$$
\text{Output Schema}
\rightarrow
\text{valid scoring dimensions}
$$

The program schema also defines the universal runtime contract:

$$
\boxed{
\text{Program Schema}
\rightarrow
\text{Benchy Runtime Contract}
}
$$

---

# 16. B1 Open Decisions / Limitations

## 16.1 Field evaluators

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

These extend field correctness without changing the core program model.

## 16.2 Variable-length outputs

Variable-length output collections are unsupported in B1.

They introduce variable evaluation dimensions and require separate semantics for:

```text
matching
ordering
missing elements
extra elements
weights
aggregation
```

## 16.3 Aggregators

`weighted_mean` is fully defined.

Other aggregators such as:

```text
weighted_sum
weighted_median
min
max
```

should not be considered supported until their treatment of weights and score scale is explicitly defined.

## 16.4 Invalid runtime outputs

B1 still needs a final policy for a Benchy-contract output that fails output-schema validation.

Candidate:

```text
invalid output
    ↓
record validation error
    ↓
instance score = 0
    ↓
continue benchmark
```

## 16.5 Dataset physical format

The program schema defines the semantic shape of each example.

A separate B1 data-format specification must define the physical representation of:

```text
examples
image/audio/document references
expected output objects
```

## 16.6 Multilingual ontology

Single-language tasks fit:

```text
/extract/finance/es
```

Translation requires an ordered source/target language relation.

Candidate:

```yaml
language:
  source: es
  target: en
```

with an ontology coordinate such as:

```text
/translate/general/es-en
```

This remains ontology design work.

## 16.7 Version coupling

`version: b1` should pin the Benchy semantic specification.

If the SURUS ontology evolves independently, a separate ontology version may later become necessary. Do not introduce one until independent evolution requires it.

---

# Appendix A — SURUS Task-Oriented AI Ontology

SURUS uses a shared ontology across Benchy, DataHub, EvalsHub, and related systems:

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

Task is explicit and controlled.

The task does not equal the program schema.

The same program signature may represent different operations.

For example:

$$
\text{string}\rightarrow\text{string}
$$

could represent:

```text
translation
summarization
rewriting
question answering
```

Therefore task \(T\) defines a family of admissible programs:

$$
T\rightarrow\mathcal{P}_T
$$

and Benchy validates:

$$
P\in\mathcal{P}_T
$$

## A.2 Domain

Domain describes the world/distribution represented by benchmark data.

Examples:

```text
general
finance
healthcare
legal
retail
```

The compiler verifies that the declared domain is a valid registry identifier.

Domain primarily classifies the data distribution; it does not define program structure.

## A.3 Language

Language describes the linguistic distribution represented by benchmark data.

Examples:

```text
es
pt
en
```

The compiler verifies that the declared identifier exists in the language registry.

It does not generally prove that arbitrary content is linguistically faithful to that metadata.

## A.4 Shared SURUS semantics

The same ontology coordinates identify compatible artifacts:

```text
DataHub
  /extract/finance/es
      datasets

EvalsHub
  /extract/finance/es
      evaluations

Benchy
  /extract/finance/es
      benchmarks + runs
```

The ontology is shared semantic infrastructure.

Its purpose is interoperability: the same task, domain, and language identifiers mean the same thing across the SURUS ecosystem.

---

# Appendix B — Compact Architecture

Authoring:

$$
\boxed{
\text{Human / Agent / UI}
\leftrightarrow
\text{YAML}
}
$$

Compilation:

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
}
$$

Execution:

$$
\boxed{
\text{JSON IR}
\rightarrow
\text{Benchy core}
\rightarrow
\text{one Benchy contract}
\rightarrow
\text{adapter}
\rightarrow
\text{external AI-system implementation}
}
$$

Core runtime rule:

> **Benchy core knows one interface. External differences are resolved at the boundary.**

Core semantic rule:

> **A small, explicit, self-sufficient semantic language defines the benchmark; compiled representations and runtime adapters do not introduce new benchmark meaning.**
