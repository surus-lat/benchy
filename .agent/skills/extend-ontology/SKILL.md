---
name: extend-ontology
description: Add a task to the SURUS ontology — the registry entry, the structural validator, and the four documents that must agree. Use when a benchmark needs an operation extract/classify/translate cannot express.
---
# Extend the ontology

The ontology is `/task/domain/language/`, shared across SURUS systems. A task defines a
*family of admissible programs*, and the compiler enforces `P ∈ P_T` — so adding one is a
semantic commitment, not a config change.

## Before you add a task

**Can exact match score it?** Field correctness is exact match, and that is the whole
constraint. It works for closed-vocabulary outputs — an `enum`, a `float` total, a `date`.
It does not work for open-vocabulary generated text: a transcript wrong by one word scores
the same as one that is entirely wrong.

This is why `transcribe` is *not* in ontology 1.0. It was withdrawn precisely because
declaring it let benchmarks compile and produce uninterpretable scores. If your task's
output is free text, adding it will produce a benchmark that cannot rank systems. Read
paper Appendix E first — the field-evaluator extension is the honest fix, not a new task.

`translate` is in 1.0 and has the same structural weakness. Know that going in.

## The four places that must agree

`tests/test_doc_agreement.py` enforces this, and it will fail loudly if you miss one.

1. **`benchy/ontologies/<version>.yaml`** — the registry entry: an id and a description.
   Identifiers only; no constraint language lives here, deliberately.

2. **`benchy/ontology.py`** — the validator pair in `_VALIDATORS`. Each task has a
   language rule and a program rule:

   ```python
   "classify": (_one_language, _classify_program),
   ```

   The program rule is where `P ∈ P_T` is enforced — `classify` requires exactly one
   output leaf and that it is an enum. Write the rule that makes an inadmissible program
   fail at compile time rather than score nonsense.

3. **`paper/technical-paper-v10.3.md`** — Appendix A.4 (the validator list) and Appendix B
   (the registry). Both, in the same order.

4. **`paper/benchy-engine-spec-v1.2.md`** §5, and the handoff's validator table.

## A new ontology version

Adding `1.1` means a new registry YAML and a new `_VALIDATORS["1.1"]` entry. The compiler
does not change — that separation is the point, so a benchmark pinning
`ontology_version: "1.0"` keeps its exact meaning forever.

## Do not

- Put constraints in the registry YAML. The paper rules this out explicitly: a constraint
  DSL is a language to invent, maintain and version, where three small functions say it
  more precisely and can be read.
- Add a task "for completeness". Every task is a promise that a benchmark declaring it
  will be scored meaningfully.
