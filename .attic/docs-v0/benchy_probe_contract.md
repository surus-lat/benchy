# Benchy Probe Contract

This document is the source of truth for what `benchy probe` tests and how to interpret failures.

## Profiles

- `quick`: runs all currently implemented checks.

## Checks

### `access_readiness`

Goal:
- Fail fast on account/access blockers before benchmark runs.

What is tested:
- Minimal chat request to the target model.
- Best-effort `/models` lookup for context.

Pass criteria:
- No blocking auth/model/quota issue is detected.

Typical issue codes:
- `invalid_api_key`
- `forbidden`
- `model_not_found`
- `insufficient_credits`
- `rate_limited`
- `access_error`

Failure meaning:
- A smoke/full eval run is unlikely to make sense until access is fixed.

### `request_modes`

Goal:
- Detect which request modes are usable for the model/provider pair.

What is tested:
- `chat`: plain text request through chat endpoint.
- `completions`: plain text request through completions endpoint.
- `logprobs`: multiple-choice probe via completions + logprobs (only when capability advertises support).

Pass criteria:
- Request returns usable output with no API/parse error.

Failure meaning:
- Endpoint/mode is not usable or unreliable under probe conditions.

### `schema_transports`

Goal:
- Detect which schema transport should be used (`structured_outputs` vs `response_format`).

What is tested:
- Basic schema probe for `structured_outputs`.
- Basic schema probe for `response_format`.
- Stress schema probe with nested object/array fields.
- Stress probe includes an image when multimodal capability is available.

Pass criteria:
- No request error.
- `finish_reason` is not `length`.
- Parsed output is a JSON object.
- No repetition pattern in raw output.
- Stress probe succeeds when executed.

Failure meaning:
- Transport may be accepted syntactically but is not reliable for real structured extraction.

Reported states:
- `accepted_by_api=true`: provider accepted the transport parameter and produced a model response.
- `reliable_for_eval=true`: output met reliability checks for structured extraction.
- `ok=true`: equivalent to `reliable_for_eval=true` (used for automatic transport selection).

Interpretation:
- `accepted_by_api=true` and `reliable_for_eval=false` means "accepted but unreliable".
- `accepted_by_api=false` and `reliable_for_eval=false` means transport is effectively unsupported.

Summary option labels:
- `usable`: transport is selected-capable (`ok=true`).
- `accepted_but_unreliable`: API accepts the format, but output is unreliable for eval.
- `unsupported_or_failed`: transport is not usable under probe checks.
- `not_tested`: transport was skipped by profile/capability path.

### `multimodal`

Goal:
- Verify image+text request path works.

What is tested:
- Chat request with bundled probe test image.

Pass criteria:
- Request accepts image and returns non-empty output.

Failure meaning:
- Multimodal path is likely unsupported or unreliable.

### `truncation`

Goal:
- Detect degenerate repetition behavior under forced truncation.

What is tested:
- Chat request with intentionally low max token budget.

Pass criteria:
- No repetition/degenerate pattern detected.

Failure meaning:
- Model can degrade into repetitive garbage under truncation.

### `param_support` (Max Tokens Parameter)

Goal:
- Detect which max-output token parameter is accepted.

What is tested:
- Request with `max_tokens`.
- Fallback request with `max_completion_tokens` when needed.

Pass criteria:
- At least one of the parameters works.

Important:
- This check is only about max output token parameter naming.
- It does **not** test temperature support.

## Status Semantics

- `passed`: usable endpoint(s), at least one reliable schema transport, no degraded checks.
- `degraded`: usable endpoint(s), but one or more warnings/degraded checks exist (including schema transport reliability issues).
- `failed`: no tested request mode is usable.

## Probe Report Fields To Inspect During Debug

When debugging failures, inspect:

- `modes.*.ok`, `modes.*.error`, `modes.*.error_type`
- `schema_transports.*.ok`, `schema_transports.*.error`, `schema_transports.*.evidence`
- `schema_transports.*.accepted_by_api`, `schema_transports.*.reliable_for_eval`
- `selected_api_endpoint`
- `selected_schema_transport`
- `Schema transport options` section in `probe_summary.txt` (with per-option reason/error)
- `checks.*.status`, `checks.*.error`, `checks.*.evidence`
- `test_plan` (exact checks, timeout budget, and pass criteria)
- `known_blindspots`

## Known Blindspots

- Probe uses synthetic prompts/schemas; dataset-specific schemas may still fail.
- Probe samples are small; this is not a statistical reliability test.
- Real eval prompts can be much larger than probe prompts.
- Transient provider/server load can create false negatives.
- Not all request params are checked; `param_support` is scoped to max-token key detection.

## Useful Variations For Better Data

- Repeat quick probe with different run IDs and compare consistency.
- Run probe immediately before eval to capture current server behavior.
- Add a provider-specific diagnostic profile (future) with:
  - multiple repetitions per check,
  - larger schema stress payloads,
  - task-shaped prompts matching your extraction datasets.
