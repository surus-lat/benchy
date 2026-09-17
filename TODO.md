# Benchy — TODO

State as of 2026-09-17. Engine 1.0 is complete: 303 tests, conformance 31/31,
PyYAML the only dependency, verified from a clean clone. Branch `REF/benchy-v1.0`,
unmerged.

Everything below is what is *not* done. Each item says who it needs.

---

## 1. Provider validation — **DONE for Together, 2026-09-17**

Together AI is the main backend. `examples/together/` runs live and scores 1.0.

- [x] First live run — `Qwen/Qwen3.8-2.4T-A95B`, 3/3 valid, benchmark score 1.0
- [x] Strict structured outputs confirmed: `response_format.json_schema` with
      `strict: true` and `additionalProperties: false` is honoured exactly
- [x] `json_object` fallback confirmed **unnecessary** — correctly not built
- [x] Real 4xx/5xx body shape: Cloudflare HTML-ish, `provider_error` truncation at
      500 chars is adequate
- [x] **Found only by running live:** Together's WAF 403s urllib's default
      `User-Agent` (Cloudflare code 1010). Fixed with an explicit `benchy/1.0` header
      and pinned by `test_requests_do_not_go_out_as_python_urllib`.

Still open for this adapter:

- [ ] Confirm an `image` input round-trips to a vision model (needs a vision-capable
      model on Together; the invoice example is text-only)
- [ ] Exercise a reasoning model with a deliberately small `max_tokens` against the
      live endpoint, to confirm the `finish_reason: length` diagnostic fires in the
      wild as it does against the stand-in
- [ ] AWS Bedrock as the secondary backend — **needs credentials from you**. Bedrock
      is *not* OpenAI-compatible (SigV4 auth, different request shape), so unlike
      Together it cannot be a one-line entry in `_ENDPOINTS`. Decide then whether it
      justifies a second adapter or is better reached through a proxy that speaks
      OpenAI.

---

## 2. Rewrite the agent skills — **mine, on your go-ahead**

`.agent/` was deleted: 21 skills, 13 of which described the architecture now in
`.attic/`. You chose to write fresh later rather than port them.

Most of the old set no longer has anything to describe. `add-provider`,
`define-task`, `define-scoring`, `configure-model` all collapse to "edit the YAML",
and `run-benchmark` is now two commands. A realistic new set is perhaps:

- [ ] `author-benchmark` — turn a business problem into a `benchmark.yaml`
- [ ] `run-benchmark` — compile, run, read the result
- [ ] `write-adapter` — expose an AI-system through `invoke(dict) -> dict`
- [ ] `interpret-result` — what `valid` / `invalid_output` / `execution_error` and a
      weighted score actually tell you
- [ ] Re-add the skills section to `CLAUDE.md` once they exist

---

## 3. Downstream: `benchy-agent` — **needs you or repo access**

`github.com/surus-lat/benchy-agent`, file `src/spec/ontology.ts`.

Its task dropdown ships `summarize`, which is in no registry version, and
`transcribe`, which was withdrawn from ontology 1.0. A spec authored there can
therefore produce YAML this engine rejects.

- [ ] Remove `summarize` and `transcribe`; leave `extract`, `classify`, `translate`

---

## 4. Deferred engineering — **no action until something forces it**

Each was left out on purpose. The reason matters more than the item.

- [ ] **Concurrency.** Sequential execution is conformant; spec §16 makes concurrency
      an optimization, and requires results keep dataset indices and serialize in
      dataset order. Add when a real run is measurably too slow, not before.
- [ ] **Field evaluators (`wer` / `cer`) and the return of `transcribe`.** Design is
      written in paper Appendix E. Blocked on one decision below.
- [ ] **A second provider adapter.** Only if something genuinely cannot be reached
      through an OpenAI-compatible endpoint.
- [ ] **`audio` / `document` inputs and artifact outputs** in the provider adapter.
      Currently rejected at setup with a clear message rather than failing per example.

---

## 5. Open decision: normalization for `wer` / `cer` — **yours, when §4 comes up**

Paper Appendix E deliberately leaves this open. It is not a small effect: measured
across the ASR predictions that were stored in `outputs/`, per-example word accuracy
differs by **9 to 15 points** between raw and normalized comparison, because an
unnormalized comparison charges a model for emitting correct capitalization and
punctuation against a reference carrying neither.

- [ ] Decide: do `word` / `char` comparisons normalize (NFKC + casefold + strip
      punctuation), or compare raw?

Related, and easy to get wrong: mean-of-per-example is a **macro**-average, while the
WER published in the literature is a **micro**-average over total edits and total
reference length. They are different numbers. Appendix E records that the raw
numerator and denominator must be preserved, not just the ratio, so both stay
available.

---

## 6. Repo hygiene — **your call**

Not referenced by the engine or any current document. Several look like personal
notes rather than project files, so they were left alone.

- [ ] `misc/`, `proto/`, `reference/`, `logs/`, `search/`
- [ ] `.notes/`, `.plans/`, `.workshop/`, `.search/`
- [ ] Root: `ai-notes.md`, `IDEAS.md`, `HANDOFF.md`, `seed.md`,
      `canonical-json-ir.md.md`, `Untitled.base`, `Untitled 1.base`, `env.example`,
      `uv.lock`
- [ ] `.attic/` itself — deletable whenever you stop wanting the old tree browsable;
      git history keeps it either way

---

## 7. Landing — **yours**

- [ ] Merge `REF/benchy-v1.0` into `main` (30 commits ahead; 11 are this rebuild).
      You chose to keep working on the branch for now.

---

## Standing rules for anything added here

From the rebuild, in priority order:

1. Cleanest expression of the paper. The code should read as the spec, not as a
   framework that happens to implement it.
2. **Fewest parts. "The best part is no part."** A module, class, protocol or
   abstraction layer must be *forced* by the spec. `docs/engine-v1/PLAN.md` lists the
   parts that were rejected and why — read it before adding one.
3. Fewest lines, never at the cost of clarity.

And the lesson this rebuild paid for four times: **tests check the code; only
installing and cloning check the artifact.** A green suite hid a missing
`package-data` entry, a dependency list pulling vendor packages into a PyYAML-only
engine, an example dataset eaten by `.gitignore`, and a CI job linting a deleted
directory. Before claiming done, install into a fresh venv and run from a fresh clone.
