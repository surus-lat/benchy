# Benchy Architecture Handoff

## Canonical JSON IR and a Universal Runtime Boundary

### Purpose

This note exports the architectural decisions needed to update the full Benchy technical paper.

It focuses on two coupled choices:

1. how the canonical YAML benchmark definition is compiled into a JSON intermediate representation (IR);
2. why Benchy should standardize one runtime contract at its boundary instead of carrying many external calling conventions inside the engine.

---

## 1. Decisions to carry into the paper

- **YAML is the single canonical semantic definition of a benchmark.**
- Each semantic concept has **one valid YAML syntax**.
- There is **no normalization layer** and no alternate equivalent syntax to reconcile.
- The authoring/execution path is:

$$
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
$$

- Validation rejects invalid or incomplete definitions.
- Compilation does **not** repair the benchmark, infer missing semantics, or inject hidden defaults.
- Humans, agents, and the UI read and write YAML.
- The UI belongs to the authoring layer; it does not need to operate on the internal IR.
- The JSON IR is a deterministic executable representation of the YAML, not a second source of truth and not a place where new semantics are introduced.
- The engine consumes the JSON IR rather than independently deriving benchmark semantics from YAML.
- At execution time, Benchy should expose **one universal runtime contract at its boundary**.
- External program implementations adapt to Benchy; the core engine does not adapt itself to every external interface.

---

# 2. Canonical semantic source and JSON IR

## YAML is the canonical source

The YAML **is the benchmark**.

It contains the complete semantic definition that humans, agents, version control, and the UI operate on.

The pipeline is:

```text
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
engine
```

## No normalization

Normalization would exist to accept multiple syntactic forms expressing the same semantic concept and collapse them into one internal form.

For example, allowing both:

```yaml
transcription: string
```

and:

```yaml
transcription:
  type: string
```

would require some normalization step if both were meant to represent the same thing.

Benchy should instead define **exactly one valid syntax**.

If the YAML does not conform, validation fails and the human or agent fixes the source.

This follows the principle:

> The best part is no part.

If Benchy does not need multiple equivalent syntaxes, it does not need a mechanism for reconciling them.

Therefore:

$$
\text{parse}
\rightarrow
\text{validate}
\rightarrow
\text{compile}
$$

is preferable to:

$$
\text{parse}
\rightarrow
\text{validate}
\rightarrow
\text{normalize}
\rightarrow
\text{compile}
$$

## Compiler invariant

> **Compile ≠ repair.**

A compiler may change representation, but it must not change meaning, invent defaults, fill missing fields, or silently reinterpret invalid source.

A precise model is:

$$
C :
\text{Valid Benchy YAML}
\rightarrow
\text{Canonical JSON IR}
$$

with:

$$
\boxed{
\operatorname{Semantics}(C(Y))
=
\operatorname{Semantics}(Y)
}
$$

The same valid benchmark definition must deterministically produce the same executable representation.

The JSON IR is therefore **derived state**: useful for execution, but never independently authored or treated as another benchmark definition.

---

# 3. Why JSON?

JSON is **not** chosen because YAML is impossible or difficult for code to parse.

After YAML has been parsed, either format can already be represented in code as ordinary maps, arrays, strings, numbers, and so on.

For example:

```yaml
program:
  output:
    transcription: string
```

and:

```json
{
  "program": {
    "output": {
      "transcription": "string"
    }
  }
}
```

are similarly easy to access programmatically once parsed.

The reason to use JSON as the IR is that it is deliberately:

- simple;
- unambiguous;
- ubiquitous across languages;
- easy to serialize and deserialize;
- easy to pass between processes and components;
- free from YAML-specific authoring syntax;
- appropriate as a stable machine-facing interchange representation.

But the architectural value is **not merely YAML → JSON**.

A mechanical conversion such as:

$$
\text{YAML}
\rightarrow
\text{equivalent JSON}
$$

does not solve much by itself.

The important transformation is:

$$
\boxed{
\text{semantic source}
\rightarrow
\text{validated, typed executable model}
}
$$

JSON is simply the representation chosen for that executable model.

---

# 4. Why compile at all?

Without a compiler boundary, different parts of the engine could independently derive meaning from the source.

For example, one component might independently determine:

- which fields are outputs;
- which fields correspond to scoring dimensions;
- whether every weight corresponds to an output;
- whether the dataset satisfies the program schema;
- whether a particular scoring definition is legal.

Another component might repeat the same reasoning separately.

That creates the possibility of semantic drift.

The undesirable architecture is therefore:

```text
                    YAML
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      component    component   component
          │            │          │
      interprets   interprets  interprets
      semantics    semantics   semantics
```

Changing YAML to JSON does not fix this:

```text
                    JSON
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      component    component   component
          │            │          │
      interprets   interprets  interprets
      semantics    semantics   semantics
```

The problem is not YAML versus JSON.

The problem is **multiple semantic interpreters**.

The preferred architecture is:

```text
YAML
 │
 ▼
one semantic compiler
 │
 ▼
validated executable JSON IR
 │
 ▼
Engine
```

Therefore:

$$
\text{YAML semantics}
\overset{\text{compiler}}{\longrightarrow}
\boxed{\text{one validated executable representation}}
$$

and:

> **After compilation, engine components do not independently derive benchmark semantics from the source definition.**

---

# 5. Authoring layer versus execution layer

The UI belongs on the **authoring side**.

Humans, agents, and the UI should operate on the YAML directly because YAML is the canonical benchmark definition.

The architecture is therefore:

```text
Human / Agent / UI
        ↕
       YAML
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
     JSON IR
        │
        ▼
      Engine
        │
        ▼
      Results
```

This gives two clean interfaces:

$$
\boxed{
\text{YAML = interface for authors}
}
$$

$$
\boxed{
\text{JSON IR = interface for execution}
}
$$

The UI does not need to consume the IR.

It assists humans and agents in creating and editing the canonical YAML.

The paper also should **not prematurely commit** to internal modules such as a `runner`, `scorer`, or `dataset loader`.

Those may eventually be useful implementation components, but they are not necessary architectural commitments at this stage.

The important invariant is simply:

> The engine consumes the compiled benchmark representation rather than reinterpreting the source YAML.

---

# 6. One universal runtime boundary

A benchmark eventually has to execute some concrete ai program implementation.

An ai **program implementation** means the actual executable system being evaluated, for example:

- a Python function;
- an HTTP API;
- an LLM workflow;
- a local model;
- a model-provider API;
- a composed pipeline;
- another executable system.

These implementations may naturally expose very different calling conventions.

For example:

```python
def transcribe(audio) -> str:
    ...
```

or:

```python
def classify(text, language="es") -> str:
    ...
```

or:

```text
POST /extract
```

or an LLM pipeline taking a message structure and returning a provider-specific response.

Benchy has two broad architectural choices.

---

## Option A — one universal Benchy runtime boundary

Every external implementation adapts once to a single Benchy runtime contract.

Conceptually:

```text
                    Benchy runtime contract
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
      Adapter A1        Adapter A2        Adapter A3
          │                 │                 │
    Python function      HTTP API        LLM workflow
```

If \(B\) is the Benchy runtime contract and \(I_1,\dots,I_n\) are external implementations:

$$
I_k
\leftrightarrow
A_k
\leftrightarrow
B
\qquad
k=1,\dots,n
$$

Each external implementation only needs to answer:

> How do I map my interface to and from the Benchy interface?

The core engine sees only \(B\).

---

## Option B — arbitrary runtime interfaces inside the engine

The alternative is for Benchy itself to understand all the different external shapes.

Then the engine starts accumulating logic such as:

```text
Engine
 ├── Python positional arguments
 ├── Python named arguments
 ├── raw-string outputs
 ├── HTTP request bodies
 ├── HTTP response extraction
 ├── provider-specific LLM messages
 ├── provider-specific output formats
 └── ...
```

Now Benchy must answer questions such as:

- Is this input positional or named?
- Is this output a raw string or an object?
- Which response key corresponds to `transcription`?
- How should multiple inputs be passed?
- How do I extract the meaningful output from this provider response?
- How does this endpoint differ from this Python function?

This pushes integration-specific knowledge into the core engine.

---

# 7. One-to-many versus many-to-many

Both designs require some form of adaptation.

The important question is:

$$
\boxed{\text{Where does adaptation live?}}
$$

With a universal Benchy boundary:

$$
\text{external implementation}
\rightarrow
\boxed{\text{Benchy interface}}
$$

Each implementation gets one adapter.

This is a **hub-and-spoke** topology.

The complexity grows approximately with the number of integrations:

$$
O(n)
$$

Conceptually:

```text
Implementation 1 ─┐
Implementation 2 ─┤
Implementation 3 ─┼──► Benchy contract
Implementation 4 ─┤
Implementation n ─┘
```

By contrast, if multiple parts of the engine understand multiple external conventions, the architecture drifts toward a mesh:

```text
                    Implementation 1
                  ↗       ↑       ↖
               engine paths / mappings
                  ↘       ↓       ↙
                    Implementation 2
                       ...
```

Different internal components may need knowledge about different implementation-specific shapes.

That approaches a **many-to-many mapping problem**.

So the chosen design principle is:

> **Standardize aggressively at the Benchy boundary.**

External implementations conform through wrappers/adapters.

The Benchy core remains ignorant of external calling conventions.

---

# 8. Why the universal boundary is simpler

A universal runtime interface gives Benchy:

- one calling convention inside the core engine;
- one validation model for runtime inputs;
- one validation model for runtime outputs;
- a direct relationship between program schemas and runtime values;
- fewer integration-specific branches in core logic;
- simpler engine reasoning;
- localized adapters;
- easier addition of new execution backends.

Adding a new backend becomes:

$$
\boxed{
\text{new implementation}
+
\text{one adapter}
}
$$

rather than:

$$
\boxed{
\text{new implementation}
+
\text{changes across engine internals}
}
$$

The engine does not need to know whether the external implementation naturally uses:

- positional arguments;
- raw strings;
- HTTP payloads;
- SDK objects;
- vendor-specific responses;
- some other representation.

That complexity terminates at the boundary.

---

# 9. Relationship to the program schema

The semantic program schema already gives Benchy a natural universal interface.

For example:

```yaml
program:
  input:
    audio: audio
  output:
    transcription: string
```

describes a program with:

- one named input field, `audio`, of type `audio`;
- one named output field, `transcription`, of type `string`.

A clean runtime design is therefore for Benchy's canonical boundary to use the same named-record shape.

For example:

```json
{
  "audio": "<audio-value>"
}
```

produces:

```json
{
  "transcription": "hello world"
}
```

The runtime value now conforms directly to the semantic schema.

This creates a strong invariant:

$$
\boxed{
\text{Every program executed by Benchy consumes and produces values conforming to its declared program schema.}
}
$$

If an external implementation instead naturally behaves like:

```python
def transcribe(audio) -> str:
    ...
```

and returns:

```text
"hello world"
```

its adapter performs:

```text
Benchy input
{"audio": ...}

     ↓ adapter

implementation input
audio

     ↓ implementation

implementation output
"hello world"

     ↓ adapter

Benchy output
{"transcription": "hello world"}
```

That mapping happens at the boundary.

The engine therefore does not require a second generic mapping mechanism for arbitrary interface shapes.

---

# 10. Semantic interface versus transport

The paper can commit to one canonical runtime boundary **without over-specifying transport**.

The program implementation may still be invoked:

- in-process;
- over HTTP;
- through an SDK;
- through a model provider;
- inside a container;
- through some future executor.

Those are execution details.

The architectural invariant concerns the contract visible to Benchy's core:

$$
\boxed{
\text{named inputs}
\rightarrow
\text{named outputs}
}
$$

not how an adapter ultimately invokes the external system.

This preserves a clean separation:

$$
\text{benchmark semantics}
\neq
\text{transport mechanics}
$$

---

# 11. One source of truth, not two

It is better not to describe the YAML and IR as two independent sources of truth.

There is only one:

$$
\boxed{
\text{YAML = canonical benchmark definition}
}
$$

The JSON IR is:

$$
\boxed{
\text{the canonical executable representation of that definition}
}
$$

Formally:

$$
J=C(Y)
$$

where:

- \(Y\) is valid Benchy YAML;
- \(C\) is the compiler;
- \(J\) is the JSON IR.

The JSON IR does not independently define anything.

Therefore:

$$
\boxed{
\operatorname{Semantics}(J)
=
\operatorname{Semantics}(Y)
}
$$

The compiler may change representation.

It may not change meaning.

---

# 12. Changes the technical paper should make

The next version of the paper should incorporate the following changes:

1. Remove `normalize` from the compiler pipeline.

   Use:

   ```text
   YAML -> parse -> validate -> compile -> JSON IR -> engine
   ```

2. State explicitly that Benchy supports **one valid YAML syntax per semantic concept**.

3. State that invalid or incomplete source is rejected rather than normalized or repaired.

4. Describe YAML as the **single canonical semantic definition** of the benchmark.

5. Describe JSON IR as the **deterministic canonical executable representation** derived from that YAML.

6. Make clear that JSON is not valuable merely because keys are easier to access. Its value comes from being a simple, stable machine-facing representation behind a single compiler boundary.

7. Put the UI on the authoring side:

   ```text
   Human / Agent / UI <-> YAML
   ```

8. Do not require the UI to consume the JSON IR.

9. Avoid prematurely committing to engine internals such as `runner`, `scorer`, or `dataset loader` unless independently justified later.

10. State that after compilation the engine does not independently derive benchmark semantics from the source YAML.

11. Add the runtime-boundary decision:

    > Benchy defines one canonical calling convention at its boundary.

12. External implementations map to Benchy through adapters/wrappers.

13. Do not let integration-specific calling conventions propagate through the core engine.

14. Explain the architecture as **hub-and-spoke instead of mesh**, or equivalently as one canonical interface serving many implementations rather than many engine components mapping against many runtime conventions.

15. Tie the universal runtime interface directly to the declared fixed, named input and output schemas.

---

# 13. Compact synthesis for the technical paper

> **Benchy uses YAML as the single canonical semantic definition of a benchmark. The language permits one valid syntax for each concept and relies on strict validation rather than normalization or hidden defaults. Valid YAML is deterministically compiled into a canonical JSON intermediate representation that introduces no new semantics and is consumed by the execution engine. At runtime, Benchy standardizes one program interface at its boundary, derived from the declared fixed named input and output schemas. External implementations adapt once to that interface; integration-specific calling conventions do not propagate into the core engine. This yields a hub-and-spoke architecture rather than a many-to-many mapping problem, keeping both benchmark semantics and execution logic small and consistent.**

In short:

$$
\boxed{
\text{YAML}
\rightarrow
\text{validate once}
\rightarrow
\text{compile once}
\rightarrow
\text{JSON IR}
\rightarrow
\text{one runtime contract}
\rightarrow
\text{many implementations}
}
$$

rather than:

$$
\boxed{
\text{many implementations}
\times
\text{many internal interpretation paths}
}
$$