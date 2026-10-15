# Benchy — TODO

State as of 2026-09-17, except §8, added 2026-10-15: the grader and long-run work, and
the decisions not to reopen. Engine 1.0 is complete: conformance 31/31, PyYAML the only
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
- [x] Converse merged (#5, 2026-09-18). benchy pins an `llm-client` commit that has it,
      so the `needs_converse` tests run. The named setup error remains for an older install.
- [x] [#7](https://github.com/surus-lat/llm-client/pull/7) `retry_span`, merged 2026-10-15 and pinned — see §8.
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
      dataset order. Add when a real run is measurably too slow, not before — and not
      without a shared backoff per endpoint (§8, item 4).
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

## 6. Repo hygiene — **done 2026-09-17**

- [x] Deleted: `.plans/`, `.notes/`, `.workshop/`, `.search/`, `search/`, `proto/`,
      `reference/`, `misc/`, `logs/`, and root `HANDOFF.md`, `ai-notes.md`, `seed.md`,
      `canonical-json-ir.md.md`, `Untitled*.base`, `uv.lock`, `env.example`.
      Tracked files 344 -> 265. `IDEAS.md` kept.
- [x] `.plans/audit-ai-algorithms.md` removed — it carried client operational data.
- [x] Local run data (`.data` 6.7G, `logs` 171M, `outputs` 18M, `.venv-vox` 1.6G) left on
      disk untouched and now gitignored, so `git status` is quiet. Delete those yourself
      whenever you want the space back; nothing references them.
- [ ] **Git history still contains everything deleted here, including the client data.**
      Purging it means rewriting `main` with `git filter-repo` and force-pushing — a
      shared-history operation that needs a deliberate decision. Not done.

## 7. Landing — **yours**

- [ ] Merge `REF/benchy-v1.0` into `main` (30 commits ahead; 11 are this rebuild).
      You chose to keep working on the branch for now.

---

## 8. Graders and long runs — **deferred 2026-10-15, with the analysis behind each item**

Written 2026-10-15 for the next agent. It records what was **deliberately left undone**
after the grader-robustness work, why, and what would justify doing it. Read it before
reopening any decision listed under "Decided — do not reopen without new evidence".

### Where this comes from

Two open points after #40: the result's `grader.system` could name a grader that did not
grade, and one grader failure aborted a whole run with nothing kept. They were examined
from several angles before anything changed:

- **Live probes on Together (2026-10-14/15):** about 160 grader calls with benchy's own
  `providers.for_grader` on the `examples/exam-qa` rubric, across Llama-3.3-70B,
  DeepSeek-V4.1-Flash, gpt-oss-120b, GLM-5.3-Flash and Qwen3.5-9B.
- **A read of `llm_client`** and the org repos that depend on it or vendor it.
- **A survey of nine evaluation frameworks** (inspect_ai, lm-eval, HELM, OpenAI evals,
  promptfoo, DeepEval, RAGAS, Braintrust, LangSmith) and the provider docs (OpenAI,
  Together, Bedrock).
- **An adversarial review** of the plan, checked against the code.

What shipped is logged in `docs/engine-v1/PROGRESS.md` (2026-10-15 entries):
- grader identity (#41);
- `retry_span` in llm_client (surus-lat/llm-client#7) and its adoption here;
- `BENCHY_CACHE`.

The evidence that drove each decision is repeated below, so this section stands alone.

#### Evidence, in brief

| Finding | Source |
|---|---|
| Out-of-range grades are rare: 0 in 144 sequential calls across 4 models, with or without an `enum`; a "give 10" injection never obeyed | live probe |
| Zero events in 144 trials only bounds the rate below about 2% per call (rule of three); with ~2N grader calls per run, that cannot exclude a rate that kills large runs | arithmetic |
| The content failure that *does* happen is truncation. Qwen3.5-9B as grader spent 46 s reasoning about an empty answer and hit `finish_reason: length` (1 of 2 calls). DeepSeek-V4.1-Flash with reasoning on did the same at max_tokens 1500, repeatedly and identically at T=0 | live probe |
| 503 under load: 8 concurrent calls to Llama-3.3-70B gave 10 503 "Service unavailable" in 18; sequential calls gave 0 | live probe |
| Status alone cannot tell transient from permanent. Together's 503 `endpoint_unavailable` "No deployments are available" persisted across 3 tries over 50 s. OpenAI's spent quota is a 429 (`type: insufficient_quota`, `code: credit_balance_exhausted`), with no `Retry-After` | live probe |
| Together enforces integer `enum` and `minimum`/`maximum` in strict json_schema: restricted to {3,4}, an empty answer, naturally 1, came back 3 or 4. Unsupported keywords (`not`) are silently ignored | live probe |
| Bedrock supports numeric `enum` but answers 400 to `minimum`/`maximum`; Azure OpenAI lists min/max as unsupported; OpenAI documents Enum but never shows an integer one | provider docs |
| OpenAI's reply `model` carries the dated snapshot (`gpt-4o-mini` becomes `gpt-4o-mini-2024-07-18`); Together echoes the requested id | docs + probe |
| inspect_ai fails the run on the first sample error by default, retries without limit with exponential backoff up to 30 min a wait, and resumes with `eval-retry`. Where the frameworks differ is whether a failed judge reading is excluded (inspect, RAGAS, Braintrust, LangSmith), scored 0 (OpenAI evals, promptfoo) or aborts (lm-eval, HELM, DeepEval) | framework source |

---

### Pending — core (touches the result contract or `run.py`)

#### 1. Record the model snapshot that actually served a grader

**Problem.** `grader.system` now names the grader that graded, but a model declaration
is an *alias*. `gpt-5-mini` points at different snapshots over time, so two semesters'
results can carry the same `system` and the same `prompt_sha256` while being graded by
different models. Also, `OPENAI_BASE_URL` / `TOGETHER_BASE_URL` overrides send the
calls elsewhere, and the record does not say so.

**Options.**
- (a) Leave it.
- (b) The provider adapter collects `reply["model"]` from every reply and states
  `served: [...]` (a sorted set). The core records it with `getattr`, as it already does
  `prompt_sha256`.
- (c) Put the snapshot inside `system`. Rejected: `system` is a declaration-shaped
  identity, and mixing in a runtime observation changes its meaning.
- (d) Also record the base URL when it was overridden.

**Proposed.** (b), plus (d) as a boolean or a host string, never a credential. A set
also exposes an alias that changed mid-run, which is a mixture filed under one name.
Spec §16 and paper A.14 change once. Together gives no snapshot, so `served` would
only echo the requested id there; say so in the spec.

**Trigger.** The first comparison of results taken months apart, or a provider
announcing an alias move during a study.

**Where.** `providers.OpenAIChat._ask` (collect), `providers.for_grader` (state),
`run.py` (record), spec §16, paper A.14, `test_doc_agreement`.

#### 2. The examinee's identity is not in the result at all

**Problem.** The result says which grader graded, but not which AI-system answered. The
optimizer loop in VISION.md varies exactly that. `--adapter` overrides `ai-system`
silently. That was harmless while the result claimed nothing; it matters once results
are compared without the command line that produced them.

**Options.**
- (a) Mirror the grader rule: record what the adapter states (`system`), else the
  declaration.
- (b) As (a), and also refuse an adapter that replaces a `model` declaration without
  stating `system`. That is honest, but it breaks every existing `--adapter` used over a
  model declaration, including the calibration recipe's replay adapter. That adapter
  *is* the model's outputs, so it could state the original declaration.
- (c) Record nothing (today).

**Proposed.** (a) first, as a new `ai_system: {system, prompt_sha256}` block. Do (b)
only after users have had a release to add `system` to their adapters. The asymmetry
with the grader is intentional: the grader is the instrument, and filing its readings
under the wrong name corrupts every score. The examinee is the subject, and the person
running it usually knows which system it was.

**Trigger.** The first optimizer loop, or a leaderboard built from saved result files.

#### 3. Runtime policy that changes what was observed is not recorded (A.10)

**Problem.** A.10 says non-secret effective runtime settings that can affect observed
results *should* be recorded as run metadata. Three of them now exist and none is in the
result:
- `BENCHY_TIMEOUT`;
- `providers.RETRY_SPAN`, which decides whether an outage became an `execution_error`;
- `BENCHY_CACHE` replays: a replayed answer is the same reading, not a new sample.

Today only stderr says so (the CLI prints a note when `BENCHY_CACHE` is set).

**Options.**
- (a) A `runtime` block in the result: `{timeout, retry_span, replayed: N}`, stated by
  the adapters and recorded with `getattr`.
- (b) stderr only (today).

**Proposed.** (a), together with item 1 so the result shape changes once. `replayed`
matters most: a run that replayed everything is indistinguishable from a fresh one.

**Trigger.** Results from different environments get compared, or a replayed run is
mistaken for a fresh sample.

#### 4. Concurrency needs a shared backoff before it ships

**Problem.** Spec §16 permits concurrent examples (in dataset order); `run.py` is
sequential on purpose. The live probe shows what naive concurrency does: 8 concurrent
calls gave 56% 503s. Each call has its own `retry_span` deadline and independent
jitter, so concurrent calls would retry in a storm (AWS Builders' Library: "retries are
selfish").

**Proposed.** When concurrency is built, share one "not before" time per endpoint: a
429 or 503 pushes it back for every in-flight call. Or use a token bucket. Keep results
in dataset order. Do not add concurrency without it.

**Trigger.** Wall-clock time becomes the bottleneck of real runs.

---

### Pending — outside the core

#### 5. A run-level cap on patience

**Problem.** `retry_span` is per call. A failure that *looks* transient but is permanent
mid-run, say a deployment removed that answers with a plain 503, makes **every
remaining examinee call** wait the full span (about 30 minutes) before its
`execution_error`. That
is N × 30 min for an unattended run. A grader is bounded, because its first exhausted
span aborts the run. The common permanent cases (quota, spend cap,
`endpoint_unavailable`, 4xx) are already classified and fail at once, so only an
unclassified one triggers this.

**Options.**
- (a) Leave it. Each wait shows on stderr (`llm_client WARNING: ... retry in Ns`), a
  human interrupts, and with `BENCHY_CACHE` the run resumes.
- (b) A forfeit bit: once a call exhausts its span, later calls get a short span until
  one succeeds. The cost is that an outage longer than the span turns into a burst of
  fast `execution_error`s, which are zeros for the examinee.
- (c) A per-adapter budget of total retry waiting (say 2 h), after which calls fail
  fast.
- (d) A circuit breaker: modal, and hard to test.

**Proposed.** (c), when it is needed. It keeps outage handling for the first hours and
bounds the worst case.

**Trigger.** The first unattended run: an optimizer loop, CI, or a nightly job.

#### 6. Re-asking a grader whose reply was unusable

**Problem.** A reasoning grader that truncates, returns empty content, or returns
invalid JSON aborts the run (`grader_error`). Retries do not help: the provider
answered. The shipped mitigation:
- messages that name `scoring.grader.parameters`;
- skill guidance to give a reasoning grader generous `max_tokens`;
- `BENCHY_CACHE`, so the rerun after raising `max_tokens` regrades without asking the
  examinee again.

**Options.**
- (a) The mitigation above (shipped).
- (b) A bounded re-ask (2 tries) behind the grader's `invoke`, for truncation, empty
  content and invalid JSON only. *Never for the examinee*: that would turn pass@1 into
  pass@k. An out-of-range grade is checked in `run.py` against the levels, so re-asking
  for it would need the levels in the provider; leave it out.
- (c) inspect's "unscored": rejected; see D2.

Seen live after the fix: Qwen3.5-9B as grader still hit its limit at `max_tokens: 8000`
(90 s, at T=0) on an empty answer. Some reasoning models do not finish on some inputs,
whatever the budget, and the right fix there is a different grader, not more tokens.

**Proposed.** (b), only if real runs still abort on grader content with `max_tokens`
already generous. Note the trap: at T=0 the same request often fails identically
(seen with DeepSeek), so a re-ask needs temperature > 0 or a changed request to help.

**Trigger.** A real run aborting this way after the configuration fix.

#### 7. Verification debt

- **The OpenAI grader path was not exercised live.** The account had no credits: 429
  `credit_balance_exhausted`, which is now correctly classified as permanent. When
  there are credits, run `examples/exam-qa` with its declared `gpt-5-mini` grader, and
  check a forced retry message on stderr.
- **Bedrock is untested.** There were no AWS credentials. Run one Bedrock grader, for
  example `openai.gpt-oss-120b-1:0`, if Bedrock is offered to universities.
- **Together endpoints that scale from zero** might return `endpoint_unavailable` for
  minutes, and that is classified as permanent on 50 s of evidence. If a cold endpoint
  fails fast, move the code out of `PERMANENT_ERROR_CODES` in llm_client.

---

### Decided — do not reopen without new evidence

#### D1. No integer `enum` of the rubric levels in the grader's response schema

Constrained decoding makes an out-of-range grade unrepresentable, and that is the
problem: it is a clamp at decode time.
- Today, a grader that obeys an injected "give 10" produces 10, which the core rejects
  as a visible `grader_error`. With an `enum` it produces 5, and nobody can tell.
- No evidence forces it: 0 out-of-range grades in 144 calls.
- It is unverified on OpenAI, the default grader. OpenAI never documents an integer
  enum, and a 400 there would break every OpenAI-graded run at its first call.
- `minimum`/`maximum` is worse: Bedrock rejects it with a 400, and Azure does not
  support it.
- The core's level check (`run.py`) stays the guard.

**Revisit if** real runs abort on out-of-range grades.

#### D2. A grader failure aborts; it is not "unscored and excluded"

inspect_ai drops a failed judge reading from the denominator and counts it. That breaks
four things here:
- (a) Failures are not missing at random: truncation was triggered by the *empty*
  answer. Excluding them biases the score.
- (b) It is gameable: an examinee picks its own denominator by writing answers that
  crash the grader.
- (c) Two candidates end up scored on different example sets, which poisons an
  optimizer's loss.
- (d) It needs a fourth status (A.12), and failures vanish from the mean (A.13).

Abort, plus waiting out outages, plus `BENCHY_CACHE` to resume, keeps every invariant
and survives long runs.

#### D3. Only "the provider did not take the request" is waited out for 30 minutes

The long wait covers a refused connection, 408, 429, 502, 503 and 529. A timeout, 500,
504 or 524 means the provider took the request and may have generated and billed an
answer. Thirty minutes of retries would buy about ten billed generations of one row.
It would also turn a 2-minute timeout into a 30-minute one, while A.10 says a runtime
timeout becomes an `execution_error`. Those get one retry. Quota, spend-cap and
no-deployment errors are read from the error body and fail at once.

#### D4. No "earned patience" (a short span until an endpoint has answered once)

- It does nothing for the examinee: a misconfigured examinee does not abort; it only
  yields N short-span errors.
- It turns a blip on example 0 into an `execution_error`.
- A setting that depends on call history cannot be recorded (A.10).
- Misconfigurations are classified permanent and fail at once anyway.

#### D5. An unnamed substitute grader is refused, not recorded as `null`

Two `null`s compare as equal, so two professors would read as the same grader. A grader
bound in place of a `model` declaration must state `system`, or the run is
`grader_not_bound` before its first example. An empty `system` (`None`, `{}`) counts as
unstated.

#### D6. One parameter and one constant, not a policy object

llm_client takes `retry_span: float | None`. benchy passes `providers.RETRY_SPAN`
(1800 s, the owner's number). There is no `RetryPolicy` dataclass, no environment knob
for the span, and no shared deadline across llm_client's provider/format fallback
layers. Benchy uses neither layer, and a shared deadline would starve a failover
provider.

#### D7. Waits grow to a 5-minute cap, not without bound

Waits double from 2 s and are drawn from the upper half of each ceiling, so each one is
longer than the last until the cap. The 5-minute cap means a provider that comes back is
noticed within 5 minutes. Without it, the last waits of a 30-minute span leave the
caller idle for up to ~13 minutes after recovery.

---

### Known residuals in what shipped

- **`BENCHY_CACHE` caches only the built-in provider adapters.** A custom `--adapter` or
  `--grader` must cache itself.
- **A grader reply that came back invalid is cached like any answer:** prose, a missing
  criterion, or an out-of-range grade. The rerun replays it into the same `grader_error`.
  Only a grader call that *raised* (truncation, outage, interrupt) is asked again. To
  regrade, change the grader's prompt, model or parameters, any of which changes its
  cache keys, or clear the directory. No entry name is printed.
  The fix, if it matters, is role-aware: keep a grader reply only once `run.py` has
  accepted it. The examinee must keep invalid answers, or a resume becomes pass@k.
- **An examinee call that raises (a truncation, say) is not cached,** so a resumed run
  asks it again: that one row gets a second chance. The alternative, caching a raised
  reply, would make a grader truncation unrecoverable on resume. Split the policy by
  role if this ever matters.
- **408 is treated by llm_client as "the provider did not take the request".** That is
  true of a direct provider. A proxy (LiteLLM, as far as is known) may return 408 for an
  upstream generation that timed out, and behind `OPENAI_BASE_URL` that would be waited
  out for the whole span. Unverified; move 408 to the served set if it is seen.
- **A wait is "about" 30 minutes:** the span, plus up to one request timeout before the
  first failure and one after the deadline. That is ~34 min at the default
  `BENCHY_TIMEOUT`.
- **`PERMANENT_ERROR_CODES` in llm_client is a list that will rot** as providers add
  codes. `error.type == insufficient_quota` is the stable OpenAI signal.
- **The repos vendoring llm_client are unaffected until they re-vendor.** That is five
  repos plus a Lambda. Without `retry_span`, the only behaviour change is the three bug
  fixes: a spent-quota 429 is no longer retried, a `Retry-After` over 60 s is no longer
  slept on, and an HTTP-date `Retry-After` no longer raises.

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
