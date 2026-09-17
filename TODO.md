# Benchy — TODO

State as of 2026-09-17. Engine 1.0 is complete: conformance 31/31, PyYAML the only
core dependency, verified from a clean clone with and without the providers extra. Branch `REF/benchy-v1.0`,
unmerged.

Everything below is what is *not* done. Each item says who it needs.

---

## 1. Providers — **Together live; Bedrock built, needs credentials**

The adapter's transport is now SURUS's `llm-client` (`pip install -e '.[providers]'`).
Together is the main backend and runs live: `examples/together/` scores 1.0.

Done, each verified against the real endpoint:

- [x] Strict structured outputs honoured exactly; no `json_object` fallback needed
- [x] Together's WAF 403s `Python-urllib` but accepts `llm-client`'s httpx agent
- [x] A null `max_tokens` is accepted, so `llm-client`'s 2000 default is never injected
- [x] A 429 is retried rather than scored; a rejected schema is never retried without it
- [x] Parameters beyond `temperature` / `max_tokens` refused at setup — through
      `llm-client` they would be silently ignored (proven with `stop` on Together)
- [x] Bedrock endpoint shape confirmed: `bedrock-runtime.<region>/openai/v1` answers an
      invalid bearer token with an OpenAI-shaped 401

Open:

- [ ] **Bedrock live run — needs from you:** a Bedrock **API key** (not IAM access
      keys) as `AWS_BEARER_TOKEN_BEDROCK`, an `AWS_REGION`, and which model(s).
- [ ] **Decide on Claude via Bedrock.** Bedrock's Chat Completions endpoint serves
      OpenAI, Qwen, Mistral, Google, NVIDIA, xAI and others — but **not** Claude
      (0 of 17), Nova (0 of 13) or Llama (0 of 12). Claude on Bedrock needs the
      Anthropic Messages API or Converse, i.e. a new request shape in `llm-client`.
      Only worth building if Claude is why Bedrock is wanted.
- [ ] `image` input round-trip against a vision model (the invoice example is text-only)

---

## 1b. `llm-client` upstream fixes — **ready on a local branch, needs your OK to push**

Found while wiring benchy to it; both proven live on Together. Branch
`benchy/finish-reason-and-extra-body` in a local clone, two independent commits,
23 tests passing (19 original + 4 new). **Not pushed** — it is a shared repo, and the
second commit changes behaviour for its other consumers.

- [ ] **Expose `finish_reason`.** Today a reply cut off by its token budget returns
      partial JSON and `llm-client` drops the reason, so benchy scores it as the
      system's own malformed answer (*"expected an object, got str"*). With the fix it
      becomes *"hit its token limit… raise max_tokens"* — verified live, before and
      after. Contract change: the result dict gains a key.
- [ ] **Merge `extra_body` into the request instead of nesting it.** Nested, servers
      ignore it without an error, so `seed` / `top_p` / `stop` never reached the model.
      **Behaviour change for the internal backend:** parameters it currently passes this way
      will start taking effect. Needs review by whoever owns those calls.
- [ ] Once merged: widen benchy's accepted parameters beyond `temperature` / `max_tokens`
      for chat-completions endpoints. (`OpenAIProfile` drops `extra_body` by design, so
      the `openai` provider stays restricted.)

---

## 1c. CI does not exercise providers — **needs a secret**

`.github/workflows/ci.yml` installs `.[dev]`, so the 42 provider tests skip there:
`llm-client` is private. Installing the extra in CI needs a GitHub token with read
access to `surus-lat/llm-client`, added as a repository secret.

- [ ] Add the token and install `.[dev,providers]` in CI

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
- [ ] **A second request shape.** Only if something genuinely cannot be reached
      through OpenAI chat completions — Claude on Bedrock is the live candidate (§1).
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
