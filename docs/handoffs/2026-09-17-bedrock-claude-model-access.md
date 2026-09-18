# Handoff — get the newest Claude models enabled on Bedrock

**For:** whoever picks up AWS account work (agent or human).
**From:** the benchy Engine 1.0 session, 2026-09-17.
**Not a benchy code task.** benchy already works with every model that *is* available;
this is purely about AWS entitlement.

---

## The ask

Make these six Claude models invocable on AWS account **`354918377724`**:

| inference profile id | needed for |
|---|---|
| `us.anthropic.claude-sonnet-5` | current-generation Sonnet evals |
| `us.anthropic.claude-opus-5` | current-generation Opus evals |
| `us.anthropic.claude-opus-4-8` | |
| `us.anthropic.claude-opus-4-7` | |
| `us.anthropic.claude-fable-5` | |
| `us.anthropic.claude-fable-5-1` | |

Already working, so nothing to do for these: `us.anthropic.claude-haiku-4-5-20251001-v1:0`,
`claude-sonnet-4-20250514-v1:0`, `claude-sonnet-4-5-20250929-v1:0`, `claude-sonnet-4-6`,
`claude-opus-4-1-20250805-v1:0`, `claude-opus-4-5-20251101-v1:0`, `claude-opus-4-6-v1`,
`claude-3-haiku-20240307-v1:0`.

## What "not available" looks like

```
POST https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-5/converse
Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK
{"messages":[{"role":"user","content":[{"text":"hi"}]}],"inferenceConfig":{"maxTokens":8}}

→ {"message": "anthropic.claude-sonnet-5 is not available for this account. You can
   explore other available models on Amazon Bedrock. For additional access options,
   contact AWS Sales at https://aws.amazon.com/contact-us/sales-support/"}
```

## Read this before you start — four things already ruled out

Each was verified with the live account, so please don't spend time re-deriving them.

**1. It is NOT the console model-access toggle.** The usual first guess. The control
plane already reports:

```
GET https://bedrock.us-east-1.amazonaws.com/foundation-models
→ {"enableAccessToAllModelsByDefault": true, ...}
```

So blanket model access is on. These six are gated *beyond* it.

**2. It is NOT a wrong model id.** All six are listed by the control plane as
`lifecycle=ACTIVE`, `inferenceTypesSupported=INFERENCE_PROFILE`, and all six have
`ACTIVE` `us.*` **and** `global.*` inference profiles (`GET /inference-profiles`
returns 27 Claude profiles). The ids above are the correct, active ones.

**3. It is NOT the "bare id vs inference profile" trap.** That trap is real and it
produces a *different* error. A bare `anthropic.claude-haiku-4-5-20251001-v1:0` gives:

> "Invocation of model ID … with on-demand throughput isn't supported. Retry your
> request with the ID or ARN of an inference profile that contains this model."

The six above fail with `us.` prefixed ids too, and their message is the
*entitlement* one, not this one. Don't confuse the two — I did at first.

**4. It is NOT a region problem, but region matters for where to verify.** `us-east-1`
exposes 6 Claude models on the `bedrock-mantle` model list; `us-west-2` exposes 1. Do
all checking in **`us-east-1`**.

## What the message points at

The error names AWS Sales, which suggests an allowlist / capacity / commitment process
rather than a self-serve switch. The most likely routes, in order:

1. **AWS console → Bedrock → Model access** for account `354918377724` in `us-east-1`.
   Even though `enableAccessToAllModelsByDefault` is true, the newest Anthropic models
   sometimes need an explicit per-model request with a use-case description. Check
   whether these six appear there as "Available to request" rather than "Access
   granted".
2. **AWS Support case** (Service: Bedrock, category: model access) asking for these
   specific model ids on this account and region.
3. **AWS Sales / the account's TAM or partner manager** — what the error literally
   directs to, and the likely path if 1 and 2 say "capacity/allowlist".
4. **AWS Marketplace subscription** — some third-party models need an accepted EULA.
   Worth checking whether Anthropic's offer shows as subscribed for this account.

## Boundary — do not do these without a human

- **Do not accept any EULA, model agreement, or Marketplace subscription on the
  company's behalf.** These are legal commitments. Prepare the request and hand the
  final click to a person.
- **Do not raise a paid-support case or anything with cost implications** without
  explicit sign-off.
- The credential available to this session is a **Bedrock API key** (bearer,
  `AWS_BEARER_TOKEN_BEDROCK`), scoped to runtime plus read-only control-plane calls.
  It reached `GET /foundation-models` and `GET /inference-profiles` and got
  `UnknownOperationException` on agreement-related paths — so **you will need proper
  IAM credentials** for anything mutating, which this session did not have.

## How to verify when it is done

```bash
export AWS_BEARER_TOKEN_BEDROCK=...    # a Bedrock API key
for m in us.anthropic.claude-sonnet-5 us.anthropic.claude-opus-5 \
         us.anthropic.claude-opus-4-7 us.anthropic.claude-opus-4-8 \
         us.anthropic.claude-fable-5 us.anthropic.claude-fable-5-1; do
  printf "%-40s " "$m"
  curl -s -X POST "https://bedrock-runtime.us-east-1.amazonaws.com/model/$m/converse" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" \
    -d '{"messages":[{"role":"user","content":[{"text":"Say OK."}]}],"inferenceConfig":{"maxTokens":16}}' \
    | head -c 120; echo
done
```

Success looks like `{"output":{"message":{"content":[{"text":"OK."}]}},"stopReason":"end_turn",...}`.

Then, end to end through benchy — edit the model id in
`examples/bedrock-claude/benchmark.yaml` and run it:

```bash
export AWS_REGION=us-east-1
benchy run examples/bedrock-claude/benchmark.yaml
```

A working model scores 3/3 on that fixture (`claude-haiku-4-5` already does).

## Context you may want

- Claude on Bedrock does **not** serve the OpenAI-compatible chat-completions
  endpoint; benchy reaches it through the Converse API. See
  [surus-lat/llm-client#5](https://github.com/surus-lat/llm-client/pull/5), where this
  entitlement gap is also noted for reviewers.
- benchy's side is done and needs nothing from this work — it is only that the newest
  models cannot currently be benchmarked.
- The keys used during this investigation were pasted into a chat transcript and
  **should be rotated**.
