# Benchy — TODO

State as of 2026-09-17. Engine 1.0 is complete: conformance 31/31, PyYAML the only
core dependency, verified from a clean clone with and without the providers extra. Branch `REF/benchy-v1.0`,
unmerged.

Everything below is what is *not* done. Each item says who it needs.

---

## 1. Providers — **Together and Bedrock both live, including Claude**

Transport is SURUS's `llm-client` (`pip install -e '.[providers]'`). Three examples run
the *same* benchmark against different AI-systems: `examples/invoices` (offline stand-in),
`examples/together`, `examples/bedrock-claude`.

- [x] Together live: `Qwen/Qwen3.8-2.4T-A95B` scores 3/3
- [x] **Bedrock + Claude live:** `us.anthropic.claude-haiku-4-5-20251001-v1:0` scores 3/3
      with nested objects, dates and floats all schema-conformant
- [x] Strict structured outputs, 429 retry, no-fallback and no-dropped-parameter
      guarantees all verified against real endpoints
- [x] Region settled by evidence: `us-east-1` has 6 Claude models on mantle, `us-west-2`
      has 1. Use `us-east-1`.

Two findings worth keeping:

- **Claude needs a cross-region inference profile id** (`us.anthropic.…`). Every Anthropic
  model on Bedrock is `INFERENCE_PROFILE`-only; a bare `anthropic.claude-…` is refused
  with *"on-demand throughput isn't supported"*.
- **Bedrock's `openai.gpt-oss-120b` ignores `response_format` entirely** and emits
  `<reasoning>…` inline, so it scores 0 on a schema-constrained benchmark. Verified with
  raw curl — that is a true measurement of that deployment, not an adapter defect.

Open:

- [ ] **Enable the newer Claude models on the AWS account** — handed off:
      [`docs/handoffs/2026-09-17-bedrock-claude-model-access.md`](docs/handoffs/2026-09-17-bedrock-claude-model-access.md).
      8 of 15 `us.*` profiles work on account `354918377724`; `sonnet-5`, `opus-5`,
      `opus-4-7`, `opus-4-8`, `fable-5`, `fable-5-1` answer *"is not available for this
      account"*. Not a console toggle — `enableAccessToAllModelsByDefault` is already
      `true`, so it is an AWS-side allowlist. Not a benchy task.
- [x] `image` input round-trip against a vision model — **done live**. `examples/vision/`
      feeds rendered invoice PNGs to `claude-haiku-4-5` on Bedrock and scores 3/3,
      exercising workspace-confined artifact resolution, base64 inlining and Converse
      image blocks end to end.
- [ ] Rotate the Together and Bedrock keys — both were pasted into a chat transcript.

---

## 1b. `llm-client` upstream — **PR open, awaiting review**

Two PRs, **both straight to `main`, independent**. Reviewers: `marianbasti`, `KennBro`.

**[#5](https://github.com/surus-lat/llm-client/pull/5) — merge now, no review needed.**
Additive only: `finish_reason`, the Bedrock Converse profile, an explicit `profile`
argument, the README section. Every line it removes from main is a signature widening;
`extra_body` handling is byte-identical to main. 30 tests. Merging it is what makes
Claude on Bedrock work from a plain `pip install`.

**[#6](https://github.com/surus-lat/llm-client/pull/6) — needs an internal backend reviewer.**
The single behaviour change: delivering `extra_body` to the model. No impact today
(that backend does not depend on the package); at migration it removes a landmine. 22 tests.

- [x] Expose `finish_reason`, so a truncated reply is distinguishable from a malformed
      one. Verified live: `max_tokens 16` goes from `invalid_output` ("expected an
      object, got str") to `execution_error` ("hit its token limit… raise max_tokens").
      Contract change — the result dict gains a key.
- [x] Deliver `extra_body` to the model instead of nesting it, mapping
      `enable_thinking` to `chat_template_kwargs` the way the internal backend's own client does.
- [x] README section recording what a measurement caller must switch off and why.
- [x] **Checked the blast radius:** the internal backend that vendors this client does **not** depend on
      this package — every caller imports its own `src.services.llm_client`, and
      `llm-client` is absent from `backend/pyproject.toml`. No production impact today;
      the PR in fact *removes* a migration landmine, since the generic profile would have
      silently dropped its `enable_thinking` suppression.
- [x] **#4: Bedrock Converse profile**, so Claude is reachable at all. Also lets a caller
      *state* the profile instead of inferring it from the hostname — benchy does, because
      a `BEDROCK_BASE_URL` pointing at a gateway would otherwise silently get the wrong
      request shape and fail every example.
- [ ] Merge #1 then #4. Until #4 lands, benchy raises a named setup error for Claude on
      Bedrock, and two of its tests skip (`needs_converse`).
- [ ] After merging: widen benchy's accepted parameters beyond `temperature` /
      `max_tokens` for chat-completions endpoints.

Deliberately left open, tracked upstream:

- [ ] [#2](https://github.com/surus-lat/llm-client/issues/2) `OpenAIProfile` drops
      `extra_body`, so extras never reach `api.openai.com` or `gpt-*`. This is why
      benchy's parameter restriction cannot simply be lifted for the `openai` provider.
- [ ] [#3](https://github.com/surus-lat/llm-client/issues/3) `parse_response`
      substitutes `reasoning` for a null `content`, and reads `reasoning` where Together
      uses `reasoning_content`.

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
