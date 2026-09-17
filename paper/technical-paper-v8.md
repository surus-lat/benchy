# Benchy B1 — Technical Design v8

## 1. Objective

Benchy is a universal benchmark-definition and execution engine for AI programs.

The benchmark taker is an **AI-system**, not necessarily a single model.

An **AI-system** is any AI-based program implementation. It may be:

- a single AI model;
- an **AI-node**, such as a model with an optimized prompt and fixed behavior;
- a composition of multiple AI models or AI programs;
- AI components intermingled with explicit or deterministic code;
- an agent or workflow built from any combination of the above.

alternative

- single AI model
- AI-node = model + optimized prompt + fixed task behavior
- composition of AI models / AI programs
- AI components + explicit deterministic code
- agent or workflow

The defining property is that it is an AI-based implementation of the declared program.

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

**Code equivalent**

```python
benchmark = Benchmark(
    program=P,
    scoring=S,
    data=D,
)
```

**Speak the math**

> “A benchmark `B` is completely specified by three things: its program `P`, its scoring function `S`, and its dataset `D`.”

and:

$$
\boxed{R=(B,M)}
$$

**Code equivalent**

```python
run = Run(
    benchmark=B,
    ai_system=M,
)
```

**Speak the math**

> “A benchmark run `R` consists of benchmark `B` being taken by AI-system `M`.”

The same benchmark can therefore evaluate many AI-systems:

$$
B(M_1),\;B(M_2),\;\ldots,\;B(M_n)
$$

**Code equivalent**

```python
scores = [
    B.evaluate(ai_system)
    for ai_system in [M1, M2, ..., Mn]
]
```

**Speak the math**

> “The same benchmark `B` can be applied to many different AI-systems—`M₁`, `M₂`, through `Mₙ`—so their performance can be compared on the same exam.”

The long-term objective is a small semantic language for defining AI-program benchmarks that humans and agents can author directly, while remaining independent of programming language, framework, provider, and transport.

Throughout this paper, formal equations are paired with **Python-like pseudocode**. The code is explanatory rather than a commitment to Benchy's final implementation API.

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

The UI may help the user choose defaults. <<< remove this last sentence, not useful here.>>> The serialized definition makes those choices explicit.

Explicitness should not create redundancy.

## 2.3 The YAML should be self-sufficient

A reader should be able to determine:

1. what benchmark is being defined;
2. what program is evaluated;
3. how it is scored;
4. what exam data is used;
5. what AI-system is taking it.

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

**Code equivalent**

```python
scoring_dimensions = program.output_schema.named_fields()

assert scoring_dimensions.is_fixed()

weights = scoring.weights_for(scoring_dimensions)
```

**Speak the math**

> “If the output fields are fixed and named, Benchy knows exactly what can be scored. Because those scoring dimensions are fixed, each can have a fixed weight, which keeps scoring simple.”

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

**Code equivalent**

```python
ontology_path = f"/{task}/{domain}/{language}/"
```

**Speak the math**

> “Every benchmark is classified along three semantic coordinates: what task it performs, what domain its data belongs to, and what language context it uses.”

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

All three values come from controlled registries. <<< not actually that they come from, but that they are limited by, or tested against, at compile time, to the registries>>>

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

**Code equivalent**

```python
valid_programs = task_registry[T].valid_programs
```

**Speak the math**

> “A task `T` defines a family of programs that are valid for that task. We call that family `𝒫_T`.”

and Benchy validates:

$$
\boxed{
P\in\mathcal{P}_T
}
$$

**Code equivalent**

```python
assert P in task_registry[T].valid_programs
```

**Speak the math**

> “The concrete program `P` must be one of the programs allowed by the declared task `T`.”

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

**Code equivalent**

```python
benchmark = Benchmark(
    program=P,
    scoring=S,
    data=D,
)
```

**Speak the math**

> “Benchmark `B` is made from program `P`, scoring function `S`, and dataset `D`.”

The fourth binds the AI-system taking the benchmark:

$$
R=(B,M)
$$

**Code equivalent**

```python
run = Run(
    benchmark=B,
    ai_system=M,
)
```

**Speak the math**

> “Run `R` means evaluating AI-system `M` on benchmark `B`.”

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

**Code equivalent**

```python
program = Program(
    input_schema=X,
    output_schema=Y,
)
```

**Speak the math**

> “To define a program, Benchy needs to know the structure and semantic types of what goes in and what must come out.”

Conceptually:

$$
P:X\rightarrow Y
$$

**Code equivalent**

```python
y = P(x)

assert X.validate(x)
assert Y.validate(y)
```

**Speak the math**

> “Program `P` maps values conforming to input schema `X` into values conforming to output schema `Y`.”

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

**Code equivalent**

```python
schema = Schema(
    field_structure=structure,
    semantic_types=types,
)
```

**Speak the math**

> “A schema says which fields exist and how they are organized, plus the semantic type of each field.”

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

**Code equivalent**

```python
Y = Enum({
    v1,
    v2,
    ...,
    vk,
})
```

**Speak the math**

> “An enum output schema `Y` allows exactly a finite set of values: `v₁`, `v₂`, through `vₖ`.”

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

**Code equivalent**

```python
transcription = P(audio)
```

**Speak the math**

> “Conceptually, this transcription program takes audio as input and produces text as output, even though Benchy gives both values explicit field names in the YAML.”

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

**Code equivalent**

```python
scoring = ScoringFunction(
    weights=weights,
    aggregator=aggregator,
)
```

**Speak the math**

> “The scoring function needs two ingredients: how important each output field is, and the rule used to combine field-level correctness into one score.”

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

<<< from "The compiler..." to "... path syntax." should be removed, does not seem to add anything here >>>

Weights satisfy:

<<< include that any w should be equal or greater than 0>>>

$$
w_j\geq0
$$

**Code equivalent**

```python
assert all(
    weight >= 0
    for weight in weights.values()
)
```

**Speak the math**

> “The weight `w_j` of output/scoring dimension `j` can be zero or positive, but never negative.”

where \(w_j\) is the weight of scoring dimension \(j\).

A zero weight means the field remains part of the program output but does not affect the score.

Negative weights are invalid.

For weighted mean:

$$
\sum_j w_j>0
$$

**Code equivalent**

```python
assert sum(weights.values()) > 0
```

**Speak the math**

> “For a weighted mean, the total weight across all scoring dimensions must be greater than zero, otherwise the mean would require division by zero.”

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

**Code equivalent**

```python
example_i = Example(
    input=x_i,
    expected=y_i_star,
)
```

**Speak the math**

> “One exam example `i` contains an input `x_i` and its expected or ground-truth output `y_i*`.”

where:

- \(i\) = exam-example index;
- \(x_i\) = input for example \(i\);
- \(y_i^*\) = expected/ground-truth output.

The input schema constrains the shape and semantic types of \(x_i\):

$$
x_i\in X
$$

**Code equivalent**

```python
assert X.validate(x_i)
```

**Speak the math**

> “The input value `x_i` for example `i` must conform to the program’s input schema `X`.”

The output schema constrains the shape and semantic types of \(y_i^*\):

$$
y_i^*\in Y
$$

**Code equivalent**

```python
assert Y.validate(y_i_star)
```

**Speak the math**

> “The expected output `y_i*` for example `i` must conform to the program’s output schema `Y`.”

```text
Input Schema  ───→ exam input
Output Schema ───→ expected output
Output Schema ───→ scoring dimensions
```
<<< here seems a bit out of context. i believe you want to state that the things on the left constrain the things on the right>>>
link

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

**Code equivalent**

```python
prediction = M(input_value)

assert X.validate(input_value)
assert Y.validate(prediction)
```

**Speak the math**

> “AI-system `M` must implement the same contract as the program: it receives valid inputs from `X` and must produce outputs in `Y`.”

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

**Code equivalent**

```python
parsed = parse(yaml_source)
validate(parsed)
json_ir = compile(parsed)
result = benchy_core.run(json_ir)
```

**Speak the math**

> “Benchy reads the YAML, parses its syntax, validates its semantics, compiles it into the canonical JSON intermediate representation, and only then gives that representation to the Benchy core.”

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

**Code equivalent**

```python
assert P in task_registry[T].valid_programs
```

**Speak the math**

> “Program `P` must belong to the set of programs permitted by task `T`.”

$$
S\in\mathcal{S}_P
$$

**Code equivalent**

```python
assert S in valid_scoring_functions(P)
```

**Speak the math**

> “Scoring function `S` must belong to the set of scoring functions that are valid for program `P`.”

$$
D\in\mathcal{D}_P
$$

**Code equivalent**

```python
assert D in valid_datasets(P)
```

**Speak the math**

> “Dataset `D` must belong to the set of datasets whose examples conform to program `P`.”

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

**Code equivalent**

```python
json_ir = compiler.compile(valid_yaml)
```

**Speak the math**

> “Compiler `C` takes valid Benchy YAML `Y` and deterministically produces canonical JSON IR `J`.”

with the invariant:

$$
\boxed{
\operatorname{Semantics}(J)
=
\operatorname{Semantics}(Y)
}
$$

**Code equivalent**

```python
assert semantics(json_ir) == semantics(yaml_source)
```

**Speak the math**

> “The compiled JSON IR `J` must mean exactly the same thing as the source YAML `Y`. Compilation can change representation, but not benchmark meaning.”

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

**Code equivalent**

```python
output_object = ai_system(input_object)

# both are named-field objects
```

**Speak the math**

> “At Benchy’s runtime boundary, every AI-system receives one object with the declared named input fields and returns one object with the declared named output fields.”

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

**Code equivalent**

```python
assert X.validate(input_object)
assert Y.validate(output_object)
```

**Speak the math**

> “A Benchy-compliant execution requires both conditions to hold: the runtime input must satisfy input schema `X`, and the runtime output must satisfy output schema `Y`. 

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

**Code equivalent**

```python
native_input = adapter.to_external(benchy_input)
native_output = external_ai_system(native_input)
benchy_output = adapter.to_benchy(native_output)
```

**Speak the math**

> “An adapter translates in both directions between an AI-system’s native interface and Benchy’s one canonical runtime contract.”

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

**Code equivalent**

```python
for implementation, adapter in integrations:
    adapter.bind(
        implementation=implementation,
        contract=benchy_runtime_contract,
    )
```

**Speak the math**

> “For each external AI-system implementation `I_k`, there is an adapter `A_k` connecting it to the same Benchy runtime contract `B_R`. The index `k` simply counts implementations from 1 through `n`.”

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

**Code equivalent**

```python
benchmark_definition = yaml_semantics
integration_mechanics = adapter_configuration

assert benchmark_definition is not integration_mechanics
```

**Speak the math**

> “What the benchmark means is not the same thing as how Benchy technically calls an AI-system. Endpoint mappings, SDK details, and transport belong to integration mechanics, not benchmark semantics.”

---

# 13. Execution

For exam example \(i\):

$$
(x_i,y_i^*)
$$

**Code equivalent**

```python
example_i = Example(
    input=x_i,
    expected=y_i_star,
)
```

**Speak the math**

> “One exam example `i` contains an input `x_i` and its expected or ground-truth output `y_i*`.”

the AI-system produces:

$$
\hat y_i=M(x_i)
$$

**Code equivalent**

```python
y_hat_i = M(x_i)
```

**Speak the math**

> “For exam example `i`, AI-system `M` receives input `x_i` and produces prediction `ŷ_i`.”

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

**Code equivalent**

```python
c_ij = int(
    predicted_value == expected_value
)
```

**Speak the math**

> “`c_ij` is the correctness score for output field `j` on exam example `i`. It equals 1 when the predicted value `ŷ_ij` exactly equals the expected value `y*_ij`, and 0 otherwise. The bold `1[condition]` is an indicator function.”

All field scores for example \(i\) form:

$$
\mathbf{c}_i
=
(c_{i1},c_{i2},\ldots,c_{in})
$$

**Code equivalent**

```python
c_i = [
    c_i1,
    c_i2,
    ...,
    c_in,
]
```

**Speak the math**

> “`c_i` is the full correctness vector for exam example `i`: one field-level correctness value for each of its `n` scoring dimensions.”

Exact match means equality of **schema-valid semantic values**, not arbitrary serialized bytes.

Field scores and correctness vectors are runtime outputs, not benchmark-definition inputs.

## 13.2 Instance score

Let:

- $(w=(w_1,\ldots,w_n)$) = fixed weight vector;
- $(A)$ = scoring aggregator;
- $(s_i)$ = score for exam example $(i)$.

Then:

$$
\boxed{
s_i=A(\mathbf{c}_i,w)
}
$$

**Code equivalent**

```python
s_i = aggregator(
    field_scores=c_i,
    weights=w,
)
```

**Speak the math**

> “The instance score $s_i$ for exam example `i` is produced by aggregator `A`, using that example’s correctness vector `c_i` and the fixed field-weight vector `w`.”

B1 currently defines `weighted_mean`:

$$
A(\mathbf{c}_i,w)
=
\frac{\sum_j w_jc_{ij}}
{\sum_jw_j}
$$

**Code equivalent**

```python
def weighted_mean(field_scores, weights):
    numerator = sum(
        weight * score
        for weight, score in zip(weights, field_scores)
    )
    return numerator / sum(weights)
```

**Speak the math**

> “For weighted mean, multiply each field’s correctness `c_ij` by its weight `w_j`, add those weighted correctness values, and divide by the total weight.”

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

**Code equivalent**

```python
benchmark_score = (
    sum(instance_scores)
    / len(instance_scores)
)
```

**Speak the math**

> “The final score of AI-system `M` on benchmark `B` is the arithmetic mean of its instance scores `s_i` across all `N` exam examples.”

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
wh
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

**Code equivalent**

```python
assert program.input_schema.validate(x_i)
```

**Speak the math**

> “The input schema determines which runtime/data values are valid inputs `x_i`.”

$$
\text{Output Schema}
\rightarrow
\text{valid }y_i^*
$$

**Code equivalent**

```python
assert program.output_schema.validate(y_i_star)
```

**Speak the math**

> “The output schema determines which values are valid expected outputs `y_i*` in the dataset.”

$$
\text{Output Schema}
\rightarrow
\text{valid scoring dimensions}
$$

**Code equivalent**

```python
scoring_dimensions = (
    program.output_schema.named_scoring_dimensions()
)
```

**Speak the math**

> “The output schema also determines which named output fields may appear as scoring dimensions.”

The program schema also defines the universal runtime contract:

$$
\boxed{
\text{Program Schema}
\rightarrow
\text{Benchy Runtime Contract}
}
$$

**Code equivalent**

```python
runtime_contract = RuntimeContract(
    input_schema=program.input_schema,
    output_schema=program.output_schema,
)
```

**Speak the math**

> “The declared input and output schemas are not just documentation: together they directly determine the one runtime interface that Benchy exposes to AI-systems.”

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

**Code equivalent**

```python
ontology_path = f"/{task}/{domain}/{language}/"
```

**Speak the math**

> “Every benchmark is classified along three semantic coordinates: what task it performs, what domain its data belongs to, and what language context it uses.”

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

**Code equivalent**

```python
output_text = P(input_text)

# This shape alone does not reveal whether P is
# translation, summarization, rewriting, etc.
```

**Speak the math**

> “A program that maps text to text cannot be classified by its schema alone: the same shape could represent translation, summarization, rewriting, question answering, or another task.”

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

**Code equivalent**

```python
valid_programs = task_registry[T].valid_programs
```

**Speak the math**

> “A task `T` defines a family of programs that are valid for that task. We call that family `𝒫_T`.”

and Benchy validates:

$$
P\in\mathcal{P}_T
$$

**Code equivalent**

```python
assert P in task_registry[T].valid_programs
```

**Speak the math**

> “Program `P` must belong to the set of programs permitted by task `T`.”

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

**Code equivalent**

```python
yaml_source = author.edit(yaml_source)
yaml_source = agent.edit(yaml_source)
yaml_source = ui.edit(yaml_source)
```

**Speak the math**

> “Humans, agents, and the UI all author and edit the same canonical YAML representation.”

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

**Code equivalent**

```python
parsed = parse(yaml_source)
validate(parsed)
json_ir = compile(parsed)
```

**Speak the math**

> “The YAML passes through one semantic compilation pipeline and becomes the canonical JSON intermediate representation used for execution.”

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

**Code equivalent**

```python
benchmark = benchy_core.load(json_ir)

benchy_output = adapter.run(
    contract=benchmark.runtime_contract,
    ai_system=external_ai_system,
    input=benchy_input,
)
```

**Speak the math**

> “Execution starts from the compiled JSON IR. The Benchy core speaks only one runtime contract; an adapter then translates that contract to the native interface of the external AI-system.”

Core runtime rule:

> **Benchy core knows one interface. External differences are resolved at the boundary.**

Core semantic rule:

> **A small, explicit, self-sufficient semantic language defines the benchmark; compiled representations and runtime adapters do not introduce new benchmark meaning.**
