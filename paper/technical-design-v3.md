# Benchy B1 — Technical Design

## 1. Objective

Benchy is a universal benchmark-definition and execution engine for AI programs.

The unit of evaluation is an **AI system implementing a program**, not a model.

A system may be:
- a model,
- a model + prompt,
- a fine-tuned model,
- a workflow,
- an agent,
- multiple models,
- an endpoint encapsulating arbitrary logic.

Benchy only requires:

\[
M:X\rightarrow Y
\]

```python
prediction = system(input)
```

The benchmark itself is defined in YAML.

> **The YAML is the benchmark definition. Everything else is machinery.**

Humans edit YAML.  
Agents edit YAML.  
The UI edits YAML.  
Version control stores YAML.  
Python, Rust, hosted services, or other runtimes consume the same YAML.

---

# 2. Design Principles

## 2.1 Explicit is better than implicit

Canonical YAML should contain all benchmark-specific information needed to understand the benchmark.

Bad:

```yaml
scoring:
  aggregator: mean
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

The UI may generate defaults, but the serialized YAML should make them explicit.

## 2.2 The YAML should be self-sufficient

A reader should be able to inspect the YAML and answer:

1. What benchmark is this?
2. What program is being evaluated?
3. How is it scored?
4. What is the exam?
5. What system is taking it?

No knowledge of Benchy internals should be required.

Universal language semantics such as `float`, `enum`, or `classify` belong to the versioned Benchy/SURUS specification and do not need to be redefined in every YAML file.

## 2.3 Nothing about the runtime should leak into benchmark semantics unless necessary

```text
Benchy YAML
    ↓
Benchy Engine / Compiler
    ↓
Runtime / Provider / Endpoint / Workflow
```

The semantic layer must not depend on Python classes, Rust structs, SDK objects, or provider-specific execution details.

## 2.4 Expressivity through a very small vocabulary

> **Types express semantic constraints, not storage representation.**

A `date` may serialize as a string, but semantically it is still a date.

Avoid task-specific classes such as:

```text
InvoiceInput
InvoiceOutput
TranscriptionResult
```

Prefer universal schemas composed from a small vocabulary.

## 2.5 Avoid complexity through oversimplification

Simplification must not remove necessary information.

Examples:
- hiding equal field weights makes the benchmark less explicit,
- defining classification output as `string` loses the closed set of valid labels,
- supporting variable output dimensions in B1 would complicate scoring semantics.

## 2.6 B1 invariant: every program has a fixed output schema

\[
\boxed{
\text{fixed output schema}
\Rightarrow
\text{fixed evaluation dimensions}
\Rightarrow
\text{fixed field weights}
\Rightarrow
\text{simple scoring}
}
\]

Variable-length output collections are outside B1 because they complicate:
- weights,
- exact-match semantics,
- matching predicted and expected items,
- aggregation,
- UI representation.

---

# 3. Benchmark Classification

SURUS uses the task-oriented AI ontology:

\[
\boxed{
/\text{task}/\text{domain}/\text{language}/
}
\]

A benchmark declares its coordinates explicitly:

```yaml
benchmark:
  task: extract
  domain: finance
  language: es
```

Meaning:
- **task** — what operation is performed,
- **domain** — what world/distribution it is performed on,
- **language** — what linguistic distribution it is performed on.

Example:

```text
/extract/finance/es
/classify/healthcare/pt
/transcribe/general/es
```

The benchmark YAML contains values from the ontology. It does not define the ontology itself.

---

# 4. Task Registry

`task` is explicit but not free-form.

A task is a **controlled semantic class of programs**.

\[
\boxed{
\text{Task} = \text{family of valid Programs}
}
\]

Examples:

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

This registry is canonical and versioned outside individual benchmark files.

It is shared across:

```text
Benchy
DataHub
EvalsHub
```

Each task \(T\) defines a set of admissible programs:

\[
\mathcal{P}_T
\]

The Benchy compiler checks:

\[
\boxed{
P \in \mathcal{P}_T
}
\]

meaning:

> the declared program is a valid member of the declared task class.

Example:

```yaml
benchmark:
  task: classify

program:
  input: image
  output:
    enum: [cat, dog]
```

Valid.

```yaml
benchmark:
  task: classify

program:
  input: image
  output:
    caption: string
```

Invalid.

---

# 5. The Four Structural Pillars

The user-facing Benchy flow is:

```text
1. Program
2. Scoring Function
3. Data
4. System
```

Canonical YAML:

```yaml
benchmark: ...

program: ...

scoring: ...

data: ...

system: ...
```

`benchmark` contains classification metadata. It is not a fifth pillar.  
The four pillars define what is evaluated, how it is scored, which exam is used, and which system takes it.

---

# 6. Pillar 1 — Program

The program defines **what must be done**.

\[
\boxed{
\text{Program}
=
\text{Input Schema}
+
\text{Output Schema}
}
\]

\[
P:X\rightarrow Y
\]

```yaml
program:
  input: ...
  output: ...
```

## 6.1 Schema

\[
\boxed{
\text{Schema}
=
\text{Field Structure}
+
\text{Semantic Types}
}
\]

### B1 semantic types

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

\[
Y=\{v_1,v_2,\ldots,v_k\}
\]

```yaml
output:
  enum: [positive, neutral, negative]
```

## 6.2 Program examples

### Transcription

| Math | YAML |
|---|---|
| \(P:\text{audio}\rightarrow\text{string}\) | `input: audio`, `output: string` |

```yaml
program:
  input: audio
  output: string
```

### Classification

| Math | YAML |
|---|---|
| \(P:X\rightarrow\{v_1,\ldots,v_k\}\) | enum output |

```yaml
program:
  input: string
  output:
    enum: [positive, neutral, negative]
```

### Extraction

| Math | YAML |
|---|---|
| \(P:\text{image}\rightarrow Y\) | fixed structured output |

```yaml
program:
  input: image
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
  input: document
  output:
    supplier:
      name: string
      tax_id: string
    date: date
    total: float
```

The output dimensions remain fixed.

---

# 7. Pillar 2 — Scoring Function

The scoring function defines **what good performance means**.

> **For structured outputs, a scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance.**

B1 exposes:

\[
\boxed{
\text{Scoring Function}
=
\text{Field Weights}
+
\text{Aggregator}
}
\]

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

## 7.1 Field weights

Weights define **importance**, not correctness.

\[
w=(w_1,\ldots,w_n)
\]

```yaml
weights:
  invoice_number: 1
  date: 1
  supplier: 1
  subtotal: 1
  total: 5
```

All weights are explicit in canonical YAML.

## 7.2 Aggregator

The aggregator maps runtime field correctness into one instance score.

\[
A(c,w)\rightarrow s
\]

Weighted mean:

\[
\boxed{
A(c,w)=
\frac{\sum_i w_i c_i}
{\sum_i w_i}
}
\]

```python
score = weighted_mean(field_scores, weights)
```

Possible aggregators:

```text
weighted_mean
sum
min
max
median
```

For binary field scores:

\[
\min(1,1,0)=0
\]

so `min` implements all-or-nothing correctness.

## 7.3 Scoring is constrained by the program schema

A scoring function may only reference fields defined in the output schema.

\[
\boxed{
\text{Program Schema}
\rightarrow
\text{Scoring Validation}
}
\]

Valid:

```yaml
program:
  output:
    supplier: string
    total: float

scoring:
  weights:
    supplier: 1
    total: 5
```

Invalid:

```yaml
scoring:
  weights:
    supplier: 1
    tax: 5
```

if `tax` is absent from the output schema.

---

# 8. Pillar 3 — Data

The data is the **exam**.

Each example is:

\[
(x_i,y_i^*)
\]

where:
- \(x_i\) = exam question,
- \(y_i^*\) = known correct answer.

Dataset:

\[
D=\{(x_i,y_i^*)\}_{i=1}^{N}
\]

The benchmark only references its location:

```yaml
data:
  path: ./data/invoices.jsonl
```

The program schema constrains every example:

\[
x_i\in X
\]

\[
y_i^*\in Y
\]

Therefore:

\[
\boxed{
\text{Program Schema}
\rightarrow
\text{Data Validation}
}
\]

The semantic dependency DAG is:

```text
                     Task
                      ↓
                Program Schema
                /            \
               ↓              ↓
       Data Validation   Scoring Validation
```

The scoring function does not define the data shape.

Both data and scoring are independently constrained by the program schema.

## 8.1 Synthetic data

Synthetic generation must produce benchmark-compliant examples.

\[
G(P,\text{context})\rightarrow D
\]

subject to:

\[
D\models P
\]

The program schema therefore defines the validity contract for both human-authored and synthetic benchmark data.

---

# 9. Pillar 4 — System

The system is the **exam taker**.

\[
M:X\rightarrow Y
\]

Benchy is AI-system-agnostic.

## 9.1 Model mode

Common user case:

```yaml
system:
  type: model
  provider: openai
  model: <model>
  prompt: ./prompts/invoice.md
  parameters:
    temperature: 0
```

Core concepts:

```text
provider
model
prompt
inference parameters
```

## 9.2 Endpoint mode

Escape hatch for arbitrary AI systems:

```yaml
system:
  type: endpoint
  endpoint: https://example.com/extract
```

The endpoint may encapsulate:

```text
models
prompts
retrieval
agents
deterministic code
validators
retries
workflows
```

From Benchy's perspective:

\[
M:X\rightarrow Y
\]

in both cases.

---

# 10. Compiler / Validation Layer

Before execution, the Benchy engine compiles and validates the YAML.

```text
Parse YAML
    ↓
Validate benchmark classification
    ↓
Validate task ↔ program
    ↓
Validate program ↔ scoring
    ↓
Validate program ↔ data
    ↓
Compile executable benchmark
```

Formally:

\[
P \in \mathcal{P}_T
\]

\[
S \in \mathcal{S}_P
\]

\[
D \in \mathcal{D}_P
\]

where:
- \(\mathcal{P}_T\) = programs admitted by task \(T\),
- \(\mathcal{S}_P\) = scoring functions valid for program \(P\),
- \(\mathcal{D}_P\) = datasets conforming to program \(P\).

The compiler is responsible for enforcing these semantic constraints.

---

# 11. Benchmark Execution Layer

The user does not define field scores or correctness vectors.

Benchy produces them during execution.

For each example:

\[
\hat y_i=M(x_i)
\]

```python
prediction = system(example.input)
expected = example.expected
```

## 11.1 Field correctness

B1 uses exact match.

For output field \(j\):

\[
c_j=
\mathbf 1[
\hat y_j=y_j^*
]
\]

```python
field_score = int(predicted_field == expected_field)
```

All field scores form:

\[
c=(c_1,\ldots,c_n)
\]

the **correctness vector**.

Example:

```text
expected   = [A123, 2026-09-07, 500.0]
prediction = [A123, 2026-09-07, 450.0]

field_scores = [1, 1, 0]
```

A field score is one element \(c_j\).

The correctness vector is the collection of all field scores.

These are runtime outputs.

## 11.2 Instance score

\[
s_i=A(c_i,w)
\]

Example:

\[
c=(1,1,0)
\]

\[
w=(1,1,5)
\]

\[
s=
\frac{1(1)+1(1)+5(0)}
{1+1+5}
=
\frac{2}{7}
\]

```python
score = weighted_mean(
    field_scores=[1, 1, 0],
    weights=[1, 1, 5],
)
```

## 11.3 Benchmark score

For instance scores:

\[
s_1,\ldots,s_N
\]

B1 uses:

\[
\boxed{
B(M)=
\frac{1}{N}
\sum_{i=1}^{N}s_i
}
\]

```python
benchmark_score = mean(instance_scores)
```

Complete runtime:

```text
input
  ↓
system
  ↓
prediction
  ↓
compare with expected
  ↓
field scores / correctness vector
  ↓
weights + aggregator
  ↓
instance score
  ↓
aggregate exam
  ↓
benchmark score
```

---

# 12. Canonical B1 YAML

```yaml
benchmark:
  task: extract
  domain: finance
  language: es

program:
  input: image
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

The UI exposes this as:

```text
Benchmark metadata
Task / Domain / Language

1. Program
2. Scoring Function
3. Data
4. System
```

Changing the UI changes the YAML.

Changing the YAML changes the benchmark.

---

# 13. B1 Limitations / Open Work

## 13.1 Field correctness

B1 supports exact match only:

\[
c_j=
\mathbf 1[
\hat y_j=y_j^*
]
\]

Suitable for:

```text
structured extraction
classification
IDs
booleans
exact categorical outputs
```

Insufficient for:

```text
question answering
summarization
translation
semantic equivalence
free-form generation
```

Future versions need generalized field evaluators:

\[
C_j(\hat y_j,y_j^*)\rightarrow[0,1]
\]

Possible extensions:

```text
normalized exact match
numeric tolerance
set match
semantic similarity
LLM-as-judge
task-specific evaluators
```

These should extend scoring without changing the core Benchy model.

## 13.2 Variable-length outputs

B1 does not support variable-length output collections.

\[
\text{variable output dimensions}
\Rightarrow
\text{variable evaluation dimensions}
\]

This introduces separate problems around:

```text
weights
matching
ordering
missing elements
extra elements
aggregation
UI representation
```

B1 preserves:

\[
\boxed{
\text{Every program has a fixed output schema.}
}
\]

---

# 14. Core Model

\[
\boxed{
\text{Schema}
=
\text{Field Structure}
+
\text{Semantic Types}
}
\]

\[
\boxed{
\text{Program}
=
\text{Input Schema}
+
\text{Output Schema}
}
\]

\[
\boxed{
\text{Scoring Function}
=
\text{Field Weights}
+
\text{Aggregator}
}
\]

\[
\boxed{
\text{Data}
=
\text{Exam Questions}
+
\text{Known Correct Answers}
}
\]

\[
\boxed{
\text{System}
=
\text{Implementation Taking the Exam}
}
\]

Task constrains the valid program space:

\[
\boxed{
\text{Task}
\rightarrow
\text{Program}
}
\]

The program constrains both scoring and data:

```text
                     Task
                      ↓
                   Program
                  /       \
                 ↓         ↓
              Data       Scoring
```

Execution:

\[
\boxed{
x_i
\xrightarrow{M}
\hat y_i
\xrightarrow{\text{compare with }y_i^*}
c_i
\xrightarrow{S}
s_i
}
\]

then:

\[
\boxed{
(s_1,\ldots,s_N)
\rightarrow
B(M)
}
\]

Benchy's design objective is:

> **A small, explicit, self-sufficient semantic language for defining and running universal AI-program benchmarks.**

---

# Appendix A — SURUS Task-Oriented AI Ontology

SURUS uses one shared ontology across Benchy, DataHub, EvalsHub, and related task-oriented AI systems:

\[
\boxed{
/\text{task}/\text{domain}/\text{language}/
}
\]

Example:

```text
/extract/finance/es
/classify/healthcare/pt
/transcribe/general/es
```

The ontology provides stable semantic coordinates for AI programs, datasets, and evaluations across the SURUS ecosystem.

## A.1 Task

**Task** describes the operation being performed.

Examples:

```text
extract
classify
transcribe
translate
summarize
```

Task is explicit and comes from a controlled registry.

It is not inferred only from the input/output schema because the same schema can represent different tasks.

For example:

\[
\text{string}\rightarrow\text{string}
\]

could represent:

```text
translate
summarize
rewrite
question-answer
```

Therefore:

\[
\boxed{
\text{Task} \neq \text{Program Schema}
}
\]

Instead, a task defines a family of valid programs:

\[
\boxed{
T \rightarrow \mathcal{P}_T
}
\]

and a benchmark program must satisfy:

\[
\boxed{
P \in \mathcal{P}_T
}
\]

Example:

```yaml
benchmark:
  task: classify

program:
  input: image
  output:
    enum: [cat, dog]
```

A `classify` task may admit any valid input schema while requiring an enum output.

```text
classify:
  input: any
  output: enum
```

An `extract` task may admit different input modalities while requiring a fixed structured output.

```text
extract:
  input: any
  output: structured
```

The task registry is canonical, versioned, and shared across SURUS systems.

## A.2 Domain

**Domain** describes the world or distribution represented by the benchmark data.

Examples:

```text
general
finance
healthcare
legal
retail
```

Domain does not define the program structure.

It characterizes the distribution on which the program is evaluated.

\[
\boxed{
\text{Domain} \rightarrow \text{Data Distribution}
}
\]

Example:

```yaml
benchmark:
  task: extract
  domain: finance
```

The same extraction program shape may be benchmarked on different domains.

## A.3 Language

**Language** describes the linguistic distribution represented by the benchmark data.

Examples:

```text
es
pt
en
```

Like domain, language characterizes the exam distribution rather than the program structure.

\[
\boxed{
\text{Language} \rightarrow \text{Data Distribution}
}
\]

The same task and program may therefore have separate benchmarks:

```text
/extract/finance/es
/extract/finance/pt
```

## A.4 Relationship to Benchy

The ontology and Benchy have distinct roles.

```text
SURUS Task-Oriented AI Ontology
          ↓
 task / domain / language
          ↓
      Benchy YAML
          ↓
    Benchy Compiler
          ↓
 Executable Benchmark
```

Within Benchy:

```text
                     Task
                      ↓
                   Program
                  /       \
                 ↓         ↓
              Data       Scoring
```

- `task` constrains the valid program space,
- `program` structurally constrains data,
- `program` structurally constrains scoring,
- `domain` and `language` classify the data distribution.

## A.5 Relationship across SURUS systems

The same coordinates should identify compatible artifacts across SURUS:

```text
DataHub
  /extract/finance/es
      datasets

EvalsHub
  /extract/finance/es
      evaluations / benchmarks

Benchy
  /extract/finance/es
      benchmark definitions + executions
```

This allows a benchmark created in Benchy to be published into EvalsHub and its compatible datasets to be discovered or published through DataHub without redefining task semantics.

The ontology is therefore shared semantic infrastructure:

\[
\boxed{
\text{Task-Oriented Ontology}
\rightarrow
\text{Programs}
+
\text{Data}
+
\text{Evaluations}
}
\]

Its purpose is interoperability: the same task, domain, and language identifiers mean the same thing across the SURUS ecosystem.
