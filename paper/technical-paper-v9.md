# Benchy — Technical Design v9

## 1. Overview and design decisions

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

The design follows these decisions:

- YAML is the canonical semantic definition authored by humans, agents, and the UI.
- Each semantic concept has one valid YAML syntax.
- The YAML is parsed, validated, and deterministically compiled into a canonical JSON intermediate representation (IR).
- Compilation changes representation, not meaning; it does not repair invalid definitions or inject hidden defaults.
- Types express semantic constraints, not storage representation.
- Programs use fixed schemas composed of named input and output fields.
- Variable-length output collections are outside the current program model.
- The output schema determines the fixed scoring dimensions.
- Task, domain, and language come from one shared SURUS ontology registry.
- Benchy defines one universal runtime contract from the declared program schema.
- External AI-systems adapt to that contract at the boundary; integration-specific mechanics do not propagate into the engine.
- The benchmark definition remains independent of programming language, framework, provider, and transport.

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
        │  runtime contract
        ↓
      Adapter
        ↓
    AI-system
```

The **engine** is Benchy's execution machinery: it consumes the compiled JSON IR, iterates through exam examples, validates runtime values, invokes the AI-system, computes field correctness and scores, and produces results.

The **runtime contract** is not another engine component. It is the interface rule the engine expects at its boundary:

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

## 2. Program

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

> The field may take exactly one value from the finite set \(v_1,\ldots,v_k\).

Every input and output value is represented by one or more named fields.

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

The semantic names matter. `audio` identifies what the input value represents; `transcription` identifies what the output value represents.

A classification program might be:

```yaml
program:
  input:
    text: string
  output:
    sentiment:
      enum: [positive, neutral, negative]
```

A structured extraction program might be:

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

The set of named output fields is fixed. Therefore the set of scoring dimensions is also fixed.

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

## 3. Benchmark classification and the SURUS ontology

Benchmarks are classified using the shared SURUS task-oriented AI ontology:

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

All three values come from one shared, versioned ontology registry used by Benchy, DataHub, EvalsHub, and other SURUS systems.

### Task and program

A task defines a family of admissible programs.

Let:

- \(T\) = a task;
- \(\mathcal{P}_T\) = the set of programs admitted by task \(T\);
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

> Program \(P\) must belong to the family of programs admitted by task \(T\).

The program schema alone cannot always determine the task. For example:

$$
\text{string}\rightarrow\text{string}
$$

> The same input/output shape could represent translation, summarization, rewriting, question answering, or another operation.

Task must therefore be explicit.

### One shared registry

The ontology should be maintained as one canonical file rather than separate task, domain, and language registries.

A minimal shape is:

```yaml
version: "1.0"

tasks:
  extract:
    description: extract named information from an input
  classify:
    description: assign one or more declared categorical labels
  transcribe:
    description: convert audio speech into text
  translate:
    description: transform text from a source language to a target language

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

Task entries additionally carry the structural constraints needed for task-to-program validation. The exact registry syntax for those constraints should be defined in the ontology specification rather than overloaded onto Benchy's semantic type vocabulary.

In particular, terms such as `structured` or `fixed_named_fields` would be **registry-level structural predicates**, not semantic types like `string`, `float`, or `document`.

For translation, language is naturally an ordered source/target relation:

```yaml
benchmark:
  task: translate
  domain: general
  language:
    source: es
    target: en
```

A corresponding ontology coordinate can be rendered as:

```text
/translate/general/es-en
```

---

## 4. Scoring

The output schema determines the dimensions over which correctness can be evaluated.

For structured outputs:

> **A scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance.**

The scoring function contains field weights and an aggregator:

$$
\boxed{
\text{Scoring Function}
=
\text{Field Weights}
+
\text{Aggregator}
}
$$

> Field weights express how important each output dimension is; the aggregator defines how those field-level scores become one instance score.

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
      tax_id: 1
    total: 5
  aggregator: weighted_mean
```

Weights are non-negative:

$$
w_j\geq0
$$

> The weight \(w_j\) of scoring dimension \(j\) may be zero or positive, but never negative.

A zero weight means the field is present and validated but does not affect the score.

### Field correctness

For exam example \(i\) and output field \(j\), let \(c_{ij}\) be field correctness.

The current evaluator is exact match:

$$
c_{ij}
=
\mathbf{1}
[
\hat y_{ij}=y_{ij}^*
]
$$

> Field score \(c_{ij}\) is 1 when the predicted value exactly equals the expected value and 0 otherwise.

Exact match is applied to schema-valid semantic values rather than arbitrary serialized bytes.

Conceptually, field correctness is the primitive produced by a field evaluator. In the current design the evaluator is fixed to exact match; it is not yet a user-configurable module.

Future evaluators could map a predicted and expected field value to a score in \([0,1]\), for example through numeric tolerance, normalized string matching, semantic similarity, or an LLM judge.

### Instance score and normalization

All field scores for example \(i\) form the correctness vector:

$$
\mathbf{c}_i
=
(c_{i1},c_{i2},\ldots,c_{in})
$$

> \(\mathbf{c}_i\) contains one correctness value for every scored output dimension of exam example \(i\).

The current aggregator is the normalized weighted mean:

$$
s_i
=
\frac{\sum_j w_jc_{ij}}
{\sum_jw_j}
$$

> Instance score \(s_i\) is the weighted average of field correctness values for exam example \(i\).

The weights therefore encode **relative importance**. Multiplying every weight by the same constant does not change the score.

For example, weights \((2,1,1)\) and \((20,10,10)\) express the same relative priorities under weighted mean.

For weighted mean:

$$
\sum_j w_j>0
$$

> At least one scoring dimension must have positive weight.

Normalization is a property of this aggregator, not a universal property of Benchy.

A future weighted-sum aggregator could instead use:

$$
s_i=\sum_j w_jc_{ij}
$$

> Under a weighted sum, weights act as point values and the score is not normalized.

The general principle is:

$$
\boxed{
\text{The aggregator defines the semantics and scale of the score.}
}
$$

> A normalized aggregator may produce scores in \([0,1]\); another aggregator may naturally use a different scale.

---

## 5. Data

The dataset is the exam.

Each exam example \(i\) contains:

$$
(x_i,y_i^*)
$$

> \(x_i\) is the input for exam example \(i\), and \(y_i^*\) is its expected or ground-truth output.

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

The benchmark definition may reference the dataset:

```yaml
data:
  path: ./data/invoices.jsonl
```

The program schema defines the **semantic structure** of every dataset example: which named values exist and which semantic types they must satisfy.

This paper does **not** define a separate canonical on-disk dataset serialization. JSONL, Parquet, file references, asset stores, and similar encodings concern how examples are stored and transported rather than what they mean. If Benchy later standardizes one dataset serialization, that can be specified separately without changing the program/data semantics defined here.

---

## 6. AI-system

The AI-system is the benchmark taker.

$$
\mathrm{AI}:X\rightarrow Y
$$

> AI-system \(\mathrm{AI}\) implements the declared program contract: it consumes valid input values from \(X\) and produces output values intended to conform to \(Y\).

An AI-system may be:

```text
single AI model
AI-node: model + optimized prompt + fixed task behavior
composition of AI models or AI programs
AI components + explicit deterministic code
agent or workflow
```

The YAML identifies the AI-system being evaluated.

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

For an externally implemented AI-system:

```yaml
ai-system:
  type: external
  id: invoice-extractor-v7
```

Here `id` answers:

> **Which AI-system is being benchmarked?**

It does not answer:

> **How does this machine call that AI-system?**

That second question belongs to the runtime adapter.

For example, the runtime environment may bind `invoice-extractor-v7` to an HTTP endpoint, Python function, container, or another executable interface. Endpoint URLs, authentication, request-field mappings, response JSON paths, Python import locations, and SDK construction are **integration mechanics**.

Keeping those outside the benchmark YAML means the YAML is self-contained **semantically**: it identifies the benchmark and AI-system under evaluation without encoding environment-specific invocation details.

---

## 7. Compiler and canonical JSON IR

The compilation pipeline is:

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

> Benchy parses the source YAML, validates its semantics, and deterministically compiles it into the JSON intermediate representation consumed by the engine.

There is no normalization step. There is one valid YAML syntax for each semantic concept.

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

> Compiler \(C\) takes valid YAML \(Y_{\text{yaml}}\) and produces JSON IR \(J\).

Compilation preserves meaning:

$$
\boxed{
\operatorname{Semantics}(J)
=
\operatorname{Semantics}(Y_{\text{yaml}})
}
$$

> The JSON IR means exactly the same thing as the YAML source; it is a machine-facing executable representation, not another source of benchmark semantics.

Validation checks:

```text
YAML syntax
  ↓
specification version
  ↓
task / domain / language registry membership
  ↓
task ↔ program compatibility
  ↓
dataset examples ↔ input/output schema compatibility
  ↓
scoring weights ↔ output schema compatibility
  ↓
AI-system definition validity
```

There is no separate notion of a “benchmark identifier” in this validation sequence unless Benchy later introduces an explicit benchmark ID.

The compiler may:

- resolve ontology registry entries;
- validate semantic types and structural constraints;
- verify scoring dimensions against the output schema;
- compile the explicit source definition into deterministic JSON IR.

It does not need a separate semantic concept called a “canonical scoring address.” An implementation may internally represent a nested field such as `supplier.name` as a path, but that is an IR representation detail rather than benchmark semantics.

---

## 8. Engine, runtime contract, and adapters

The **engine** consumes the JSON IR and performs benchmark execution.

Its responsibilities include:

```text
load compiled benchmark
iterate through dataset examples
validate inputs
invoke the AI-system
validate outputs
compute field correctness
aggregate instance scores
compute benchmark result
emit results and errors
```

The engine expects every AI-system through one runtime contract derived from the program schema.

For:

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

the contract is conceptually:

```json
{
  "audio": "<audio-value>"
}
```

to:

```json
{
  "transcription": "hello world"
}
```

An external AI-system may expose a different native interface:

```python
def transcribe(audio) -> str:
    ...
```

The adapter translates:

```text
Benchy input object
        ↓
      adapter
        ↓
native AI-system input
        ↓
     AI-system
        ↓
native AI-system output
        ↓
      adapter
        ↓
Benchy output object
```

The architectural rule is:

$$
\boxed{
\text{one runtime contract}
\leftrightarrow
\text{many adapters}
\leftrightarrow
\text{many AI-systems}
}
$$

> The engine understands one interface. Differences between AI-systems are localized in adapters rather than spread throughout Benchy.

Reusable HTTP, Python, provider, or SDK adapters may exist as engineering conveniences, but they do not change benchmark semantics.

---

## 9. Execution and results

For exam example \(i\), the AI-system produces:

$$
\hat y_i=\mathrm{AI}(x_i)
$$

> AI-system \(\mathrm{AI}\) receives exam input \(x_i\) and produces prediction \(\hat y_i\).

The adapter maps that native result into the named output object expected by Benchy.

The output is then validated against schema \(Y\).

### Valid output

If the output conforms to the schema, Benchy computes field correctness and then the instance score \(s_i\).

### Invalid output

An output may be invalid because it:

```text
omits a required field
contains a value of the wrong semantic type
returns an enum value outside the declared domain
cannot be mapped into the declared named output object
otherwise fails output-schema validation
```

An invalid output should remain distinguishable from a valid output whose score happens to be zero.

The result for that exam example therefore records:

```yaml
status: invalid_output
score: null
error: <validation error>
```

A valid but completely incorrect result can instead be:

```yaml
status: valid
score: 0
```

For benchmark-level aggregation, invalid outputs must not be silently omitted because doing so would artificially improve the final score.

Define aggregation contribution \(q_i\) as:

$$
q_i=
\begin{cases}
s_i, & \text{if example }i\text{ produced a valid output} \\
0,   & \text{if example }i\text{ produced an invalid output}
\end{cases}
$$

> The stored instance score remains `null` for an invalid output, but its contribution to the final benchmark score is zero.

For \(N\) exam examples:

$$
\boxed{
B(\mathrm{AI})
=
\frac{1}{N}
\sum_{i=1}^{N}q_i
}
$$

> The benchmark score of AI-system \(\mathrm{AI}\) is the arithmetic mean of all \(N\) aggregation contributions, so invalid outputs remain visible without disappearing from the denominator.

---

## 10. Canonical YAML example

```yaml
version: "1.0"

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

`version: "1.0"` identifies the Benchy semantic specification version.

If the shared SURUS ontology registry later evolves independently from the Benchy specification, an independent ontology version can be introduced then. Until that separation is necessary, one specification version is sufficient.

---

## 11. Dependency structure

The semantic dependencies are:

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

The program schema also defines the runtime contract:

$$
\boxed{
\text{Program Schema}
\rightarrow
\text{Runtime Contract}
}
$$

> The same named input/output schemas that define the program also define the canonical interface through which Benchy executes AI-systems.

---

## 12. Scope and future extensions

### Richer field evaluators

Field correctness is currently exact match.

The conceptual evaluator boundary is:

$$
E_j(\hat y_{ij},y_{ij}^*)\rightarrow c_{ij}
$$

> Field evaluator \(E_j\) compares the predicted and expected value for output dimension \(j\) and produces its field correctness score.

Today \(E_j\) is fixed to exact match. The evaluator is not exposed as a user-configurable module.

Future evaluators may support:

```text
normalized exact match
numeric tolerance
set match
semantic similarity
LLM-as-judge
task-specific evaluation
```

### Variable-length output collections

Outputs are fixed named schemas.

Variable-length collections would require additional semantics for:

```text
matching
ordering
missing elements
extra elements
weights
aggregation
```

They should be introduced deliberately when required rather than hidden inside the current model.

### Additional aggregators

The current scoring semantics use normalized weighted mean, where weights express relative importance.

Future aggregators may define other score scales, including raw weighted sums. The architecture should not impose a universal \([0,1]\) range on all future Benchy scores.

---

# Appendix A — Shared SURUS Task-Oriented AI Ontology

The SURUS ontology provides shared semantic coordinates for Benchy, DataHub, EvalsHub, and related systems.

Its canonical coordinate is:

$$
\boxed{
/\text{task}/\text{domain}/\text{language}/
}
$$

> The coordinate locates an artifact by operation, domain/distribution, and linguistic context.

The ontology is maintained in one shared registry containing:

```text
tasks
domains
languages
```

Task entries additionally define the structural constraints used to determine whether a program belongs to that task's admissible program family.

The registry serves several purposes:

- prevents free-form incompatible task names;
- gives domain and language stable identifiers;
- lets Benchy validate benchmark classification at compile time;
- lets DataHub and EvalsHub organize compatible artifacts under the same coordinates;
- preserves consistent semantics across SURUS projects.

Example coordinates:

```text
/extract/finance/es
/extract/finance/pt
/classify/healthcare/es
/transcribe/general/es
/translate/general/es-en
```

Task constrains the program:

$$
P\in\mathcal{P}_T
$$

> Program \(P\) must conform to the structural and semantic constraints associated with task \(T\).

Domain and language primarily classify the distribution represented by the data.

The compiler validates that all declared task, domain, and language values exist in the shared registry. It does not generally prove from arbitrary dataset content that the declared domain or language metadata is factually correct.

---

# Appendix B — Compact architecture

```text
                    SURUS ontology registry
                             │
                             ▼
Human / Agent / UI  ↔  canonical YAML
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
                     [runtime contract]
                             │
                             ▼
                           Adapter
                             │
                             ▼
                         AI-system
```

The core semantic rule is:

> **The benchmark is defined in a small, explicit semantic language. Compiled representations, adapters, and transport mechanics may change how Benchy executes that definition, but they do not introduce new benchmark meaning.**
