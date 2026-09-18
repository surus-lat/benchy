---
name: add-provider
description: Add a hosted model provider to benchy, or work out whether a vendor needs a new request shape in llm-client instead. Use when someone wants to benchmark models on a provider benchy does not ship.
---
# Add a provider

Most of the time this is **one line**. Occasionally it is a new request shape in
`llm-client`, and telling the two apart is the whole skill.

## The easy case: an OpenAI-compatible endpoint

A provider is an endpoint plus the name of its credential, in `benchy/providers.py`:

```python
_ENDPOINTS = {
    "openai":   ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "bedrock":  ("https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1", "AWS_BEARER_TOKEN_BEDROCK"),
}
```

`{NAME}` is filled from the environment. `<PROVIDER>_BASE_URL` overrides the endpoint, so
vLLM, LM Studio, Ollama and any gateway already work without a new entry.

Add a row, add a test to `tests/test_providers.py` alongside the existing parametrized
endpoint cases, done.

## Check before you assume it is easy

Ask the endpoint, do not trust the marketing:

```bash
curl -s $BASE/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":16,
       "response_format":{"type":"json_schema","json_schema":{"name":"o","strict":true,
       "schema":{"type":"object","properties":{"a":{"type":"number"}},
       "required":["a"],"additionalProperties":false}}}}'
```

Three things to confirm, because each has bitten:

1. **Does the model serve chat completions at all?** Anthropic models on Bedrock do not —
   the endpoint says so explicitly — and needed a whole Converse profile in `llm-client`.
2. **Is `response_format` honoured?** Some deployments accept the field and ignore it,
   answering in prose. Those score 0 on any schema-constrained benchmark, which is a true
   measurement but worth knowing before you blame the adapter.
3. **Does the model id need a special form?** Bedrock requires cross-region inference
   profile ids (`us.anthropic.…`); a bare id is refused with a message about on-demand
   throughput that reads like a permissions problem and is not one.

## The hard case: a different request shape

If the vendor does not speak chat completions, the translation belongs in **`llm-client`**,
not here — every SURUS system then gets it. Implement a profile with `build_url`,
`build_payload` and `parse_response`, and return the structured answer as `content` JSON
so callers that `json.loads(content)` need no special case.

Then have benchy *state* which profile applies rather than letting `llm-client` infer it
from the hostname — a `<PROVIDER>_BASE_URL` pointing at a gateway defeats a heuristic, and
the failure is silent.

## Do not

- Add an SDK dependency. Transport is `llm-client` over `urllib`/`httpx`; benchy's core
  dependency is PyYAML alone and should stay that way.
- Put credentials in benchmark YAML. They are runtime policy, read from the environment.
