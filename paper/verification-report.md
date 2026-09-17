# Benchy B1 — Independent Verification Report

Reviewed artifacts:
- `technical-design-v3.md`
- `important-stuff.md`

Review method: five independent passes with different failure criteria:
1. formal logic / ontology,
2. scoring mathematics,
3. YAML / schema / compiler implementability,
4. design-principle consistency,
5. adversarial edge cases.

## Overall verdict

The core architecture is coherent and strong.

The following are internally consistent and should be preserved:

- `Schema = Field Structure + Semantic Types`
- `Program = Input Schema + Output Schema`
- task is explicit and controlled, not free-form
- task constrains the valid program space
- program constrains data
- output schema constrains scoring
- B1 fixed-output-schema invariant
- field weights represent importance, not correctness
- field scores / correctness vectors are runtime outputs
- YAML is the canonical semantic representation
- runtime implementation details should not leak into benchmark semantics
- variable-length collections are deliberately outside B1
- B1 field correctness is exact match

The current document is **not yet implementation-complete**. The remaining issues are concentrated at a few boundary cases rather than in the central model.

---

# Reviewer A — Formal Logic / Ontology

## A1. `Task -> Program -> {Data, Scoring}` is directionally correct

Recommended precise reading:

$$
T \rightarrow \mathcal{P}_T
$$

$$
P \in \mathcal{P}_T
$$

A task defines a set of admissible programs; the concrete program must be a member of that set.

This is cleaner than `P models T` for the paper.

## A2. Data and scoring are not constrained in exactly the same way

The whole program constrains data:

$$
x_i \in X,\qquad y_i^* \in Y
$$

The **output schema specifically** constrains scoring, because weights refer to output evaluation dimensions.

More precise DAG:

```text
                     Task
                      ↓
                   Program
                  /       \
                 ↓         ↓
              Data     Output Schema
                            ↓
                         Scoring
```

Or, at the schema level:

```text
Input Schema  ───→ data.input
Output Schema ───→ data.expected
Output Schema ───→ scoring
```

## A3. `benchmark` vs `system` is still conceptually unresolved

Earlier design reasoning distinguishes:

$$
B=(P,S,D)
$$

from the system taking the benchmark:

$$
M:X\rightarrow Y
$$

and an evaluation run:

$$
R=(B,M)
$$

The current YAML includes `system`, which is useful for the four-step UI, but the paper currently sometimes calls the whole YAML "the benchmark".

This creates a real semantic ambiguity:

- Is `system` part of the benchmark?
- Or is it the implementation being evaluated by the benchmark?

This should be resolved explicitly before freezing B1.

A clean option is:

```text
Benchmark = Program + Scoring + Data
Run       = Benchmark + System
```

while still allowing one YAML document to contain both.

## A4. The ontology needs controlled values for domain and language too

The document correctly makes `task` controlled.

If `/task/domain/language/` is truly an ontology shared across SURUS systems, `domain` and `language` should also come from canonical identifiers, even if only `task` imposes structural constraints on the program.

Compiler validation can verify identifier validity.

It generally cannot verify that a dataset is *actually* finance or Spanish; that is metadata/curation semantics rather than structural validation.

## A5. Translation exposes a language-ontology edge case

A single coordinate:

```text
/translate/general/es
```

does not, by itself, encode translation direction.

Translation normally involves at least source and target language.

This does not invalidate the ontology, but the semantics of `language` for multilingual tasks must be defined explicitly or marked as open work.

---

# Reviewer B — Scoring Mathematics

## B1. Weighted mean is correct

For correctness vector:

$$
c=(1,1,0)
$$

and weights:

$$
w=(1,1,5)
$$

the document computes:

$$
\frac{1+1+0}{7}=\frac{2}{7}
$$

Correct.

## B2. Scalar outputs are not yet represented by the scoring YAML

Examples such as:

```yaml
program:
  input: audio
  output: string
```

and:

```yaml
program:
  input: string
  output:
    enum: [positive, neutral, negative]
```

have one scoring dimension, but the current `weights:` syntax assumes named output fields.

The design needs one explicit rule for scalar outputs.

Possible solutions include:
- make `weights` mirror the output schema, allowing `weights: 1` for scalar output;
- introduce a reserved root output name;
- require even one-value outputs to have a named field.

Do not leave this implicit.

## B3. Nested fixed outputs need an explicit weight mapping rule

Example:

```yaml
output:
  supplier:
    name: string
    tax_id: string
  total: float
```

What are the scoring dimensions?

The cleanest candidate is that weights mirror the output structure:

```yaml
weights:
  supplier:
    name: 1
    tax_id: 1
  total: 5
```

This preserves:
- explicitness,
- fixed dimensions,
- structural correspondence,
- no extra path notation.

This is a design recommendation, not yet an agreed B1 rule.

## B4. `min`, `max`, `median`, and `sum` are underspecified with weights

The document defines:

$$
A(c,w)\rightarrow s
$$

but then lists aggregators that do not have an obvious common interpretation of `w`.

For example:
- does `min` ignore weights?
- is `max` applied to correctness or weighted correctness?
- is `sum` normalized?
- what is a weighted median here?

Therefore B1 should not advertise aggregator names until their exact semantics are defined.

Safest B1:
- define `weighted_mean` completely;
- keep `aggregator` as the abstraction;
- list other aggregators as future extensions.

## B5. Weight validity needs constraints

If weights represent relative importance, B1 should define at minimum:

$$
w_j \ge 0
$$

and for weighted mean:

$$
\sum_j w_j > 0
$$

Whether `0` is allowed should be explicit:
- allow it to mean "present in output but does not affect score", or
- require strictly positive weights.

Negative weights should not be allowed.

## B6. There are two aggregation levels

The current design has:

1. field scores -> instance score,
2. instance scores -> benchmark score.

The document fixes level 2 to arithmetic mean:

$$
B(M)=\frac{1}{N}\sum_i s_i
$$

This is mathematically fine, but it is another scoring policy.

B1 should state clearly whether:
- this mean is a fixed invariant of the B1 protocol, or
- it belongs in configurable scoring semantics.

Otherwise it is a hidden assumption.

## B7. Indexing should distinguish examples from fields

Prefer:

$$
c_{ij}
$$

for field $j$ on example $i$, and:

$$
\mathbf c_i=(c_{i1},\ldots,c_{in})
$$

Then:

$$
s_i=A(\mathbf c_i,w)
$$

This removes the current ambiguity where `c_i` can look like either one field score or one example's vector.

---

# Reviewer C — YAML / Types / Compiler

## C1. Syntax verification passed

All 23 YAML blocks in `technical-design-v3.md` parse successfully as YAML.

All 6 Python code blocks parse successfully as Python.

Markdown code fences and display-math delimiters are balanced.

## C2. Exact match should operate on semantic typed values, not raw serialization

The paper says:

> Types express semantic constraints, not storage representation.

Therefore B1 exact match should not accidentally become raw-string comparison for every type.

Example:

```text
date("2026-09-07")
```

should be compared as a `date`, even if its storage representation is textual.

Recommended execution order:

```text
system output
    ↓
parse / validate against output schema
    ↓
typed predicted value
    ↓
exact equality with typed expected value
```

Then exact match means equality in the declared semantic type.

This is especially important for:
- `date`,
- `time`,
- `datetime`,
- `int`,
- `float`,
- `bool`,
- `enum`.

Canonical serialization for these types should eventually be specified.

## C3. Invalid system output behavior is unspecified

If the system:
- omits a required field,
- returns a wrong type,
- returns malformed structured output,
- adds undeclared fields,

the engine needs a deterministic B1 policy.

This must be defined before implementation.

## C4. Dataset physical format is underspecified

The YAML correctly contains only:

```yaml
data:
  path: ./data/invoices.jsonl
```

But the engine still needs a canonical dataset record contract.

Minimal example:

```json
{"input": "...", "expected": {...}}
```

The dataset format can live in a separate data-format specification, but B1 needs to reference it.

## C5. Version pinning is required for reproducibility

The document says the task registry / ontology is versioned, but the canonical YAML does not pin a version.

If the meaning of `classify` or a semantic type changes later, the same YAML could compile differently.

A durable benchmark should identify at least:
- Benchy specification version,
- SURUS ontology/task-registry version.

The exact YAML syntax is still a design choice.

---

# Reviewer D — Design Principles / Taste

## D1. The strongest design principles are consistently present

The current design aligns with the principles captured in `important-stuff.md`:

- explicit is better than implicit,
- explicit should not mean redundant,
- YAML should be self-sufficient,
- users should know as little Benchy machinery as possible,
- runtime details should not leak into benchmark semantics,
- types express semantic constraints,
- expressivity comes from a small vocabulary,
- fixed output schema is a deliberate B1 invariant.

## D2. The task ontology is duplicated too much in the main body and appendix

The main document currently has:
- Benchmark Classification,
- Task Registry,
- a full ontology appendix.

For a 2–4 page high-signal document, the main body should probably contain only the minimum needed to understand Benchy:

```text
benchmark.task/domain/language
task constrains program
compiler validates compatibility
```

The fuller rationale and shared SURUS ontology belong in Appendix A.

## D3. Do not expose compiler-internal abstractions unless they earn their place

Terms like:
- leaf,
- decomposition,
- mapping,
- sequence,
- collection semantics

were correctly removed from the core model unless necessary.

That should remain a hard design filter.

---

# Reviewer E — Adversarial Edge Cases

## E1. Fixed output schema does not automatically define scoring dimensions unless the scalar/nested rule is fixed

The invariant is correct, but the compiler still needs an exact mapping:

$$
Y \rightarrow \text{scoring dimensions}
$$

for:
- scalar $Y$,
- flat structured $Y$,
- nested fixed structured $Y$.

This is the biggest remaining scoring-schema gap.

## E2. Classification is structurally clean only if enum equality is typed

The program:

```yaml
output:
  enum: [positive, neutral, negative]
```

is excellent.

But the compiler should validate that both expected and predicted values are members of that enum before exact comparison.

## E3. Domain/language are not structurally provable from data

A compiler can verify:

```yaml
language: es
```

is a valid ontology identifier.

It cannot generally prove that an arbitrary dataset is actually Spanish.

The paper should distinguish:
- structural validation,
- semantic metadata / curation guarantees.

## E4. System configuration is intentionally runtime-facing

`provider/model/prompt/parameters` are runtime details, but they live under `system`, not inside program/scoring/data semantics.

The paper should state this explicitly so the runtime-leak principle is not read as forbidding system configuration.

---

# Recommended Fix Order

## Blocking before calling B1 implementation-complete

1. Define benchmark vs run (`B=(P,S,D)` vs system $M$).
2. Define scoring dimensions for scalar and nested fixed outputs.
3. Define exact semantics for supported B1 aggregator(s).
4. Define invalid system-output behavior.
5. Pin spec / ontology versions.

## Strongly recommended

6. Define exact match as typed semantic equality.
7. Define weight validity (`w >= 0`, nonzero total).
8. Define/reference canonical dataset record format.
9. Clarify domain/language registry validation.
10. Mark multilingual/translation language semantics as open if not already defined by SURUS.

## Editorial / taste

11. Move most ontology detail to the appendix.
12. Use `c_ij` / `c_i` vector notation consistently.
13. Replace ambiguous "Program Schema -> Scoring" wording with "Output Schema -> Scoring" where precision matters.

---

# Consensus

No reviewer found a flaw in the central Benchy thesis.

The strongest core remains:

$$
\boxed{
\text{Schema}
=
\text{Field Structure}
+
\text{Semantic Types}
}
$$

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
\boxed{
\text{fixed output schema}
\Rightarrow
\text{fixed evaluation dimensions}
\Rightarrow
\text{fixed weights}
\Rightarrow
\text{simple scoring}
}
$$

and:

> **For structured outputs, a scoring function specifies the relative importance of output fields and the rule by which field-level correctness is aggregated into program-level performance.**

The remaining work is mostly to make the boundary cases as explicit and self-sufficient as the core design already is.
