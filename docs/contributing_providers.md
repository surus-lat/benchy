# Contributing Providers

Providers define how Benchy connects to an API stack and which interface is used at runtime.
Today there are two main extension paths:

- OpenAI-compatible stacks (OpenAI, vLLM OpenAI server, Together, Anthropic routing in `OpenAIInterface`)
- Task-specific HTTP systems (for example SURUS endpoints via `HTTPInterface` subclasses)

This guide reflects current behavior in:

- `src/interfaces/openai_interface.py`
- `src/interfaces/surus/surus_remove_background_interface.py`
- `src/engine/connection.py`
- `src/benchy_cli_eval.py`

## Choose the Right Path

### Path A: OpenAI-compatible API (preferred when possible)

Use this when your endpoint speaks chat/completions in OpenAI style.
You usually do not need a new interface, only provider config and wiring.

### Path B: Custom HTTP endpoint (SURUS-style)

Use this when payload shape is custom (non-chat, task-specific JSON/body).
Create an `HTTPInterface` subclass and a system config under `configs/systems/`.

## Step 1: Add Configuration

### For model providers (OpenAI-compatible)

1. Add or update `configs/providers/<name>.yaml` (start from `configs/templates/provider_openai.yaml`).
2. In model config (`configs/models/*.yaml`), reference it via:

```yaml
openai:
  provider_config: openai
  overrides:
    timeout: 120
```

The merge is handled by `ConfigManager` for `openai`/`anthropic`/`together`/`vllm`.
If you introduce a new top-level model provider section key, update
`ConfigManager.load_model_config(...)` to merge `provider_config` for that key.

### For system providers (custom HTTP, SURUS-style)

Create `configs/systems/<name>.yaml` with:

- `provider_type`
- provider block matching that type (for example `surus_remove_background:`)
- endpoint/auth/timeouts/capabilities

Example ground truth: `configs/systems/surus-remove-background.yaml`.

## Step 2: Implement or Reuse an Interface

### Reuse `OpenAIInterface` when possible

`OpenAIInterface` already handles:

- prompt-based request construction (`task.get_prompt(...)`)
- multimodal chat payloads
- schema/logprobs capability gating
- API key resolution (`api_key` override, then `api_key_env`)
- optional image downscaling (`image_max_edge`)

If your provider fits this, avoid creating a new interface.

### Build an `HTTPInterface` subclass for custom payload APIs

For SURUS-style systems, subclass `HTTPInterface` and implement:

- `prepare_request(sample, task)`
- `_make_request_with_client(client, request)`
- `_parse_response(response)`

Optional: override `build_test_request()` and/or `test_connection()` for endpoint-specific health checks.
Authentication for `HTTPInterface` subclasses is env-var based (`api_key_env`), so document
the expected key in your system config and setup docs.

Ground truth example: `src/interfaces/surus/surus_remove_background_interface.py`.

## Step 3: Register Provider Wiring

Update `src/engine/connection.py`:

1. Add defaults in `PROVIDER_CAPABILITY_DEFAULTS`.
2. Extend `build_connection_info(...)`:
   - map provider config fields to `base_url`, `api_key_env`, timeout/retry values
   - keep capability mapping accurate
3. Extend `get_interface_for_provider(...)`:
   - instantiate your interface
   - adapt config shape if needed (SURUS-style nested provider blocks)

## Step 4: CLI and Config Entry Points

If you want CLI provider selection support (`--provider`), update:

- `PROVIDER_SPECS` in `src/benchy_cli_eval.py`
- provider `choices` for `--provider` in `src/benchy_cli_eval.py`
- optionally `CLI_PROVIDER_DEFAULTS` if CLI-only runs should work without a config file

Important:

- `--base-url` / `--api-key` overrides are only accepted for OpenAI-compatible providers.
- system providers are typically selected via `--config configs/systems/<name>.yaml`.

## Step 5: Capabilities and Compatibility

Declare capabilities under provider config:

- `supports_multimodal`
- `supports_schema`
- `supports_files`
- `supports_logprobs`
- `supports_streaming`
- `request_modes` (`chat`, `completions`, `raw_payload`)

These are merged with model/task constraints and checked before running tasks.

## Step 6: Shared Multimodal Image Handling

Do not duplicate image preprocessing logic in each interface.
Use shared helpers in `src/interfaces/common/image_preprocessing.py`:

- `encode_image_base64(...)`
- `encode_image_data_url(...)`
- `load_pil_image(...)`

Pass through `image_max_edge` from provider config/CLI so resizing happens in-memory only.
Original image files must never be modified.

## Step 7: Document and Test

1. Update docs:
   - `README.md` (user-facing usage)
   - `docs/evaluating_models.md` (flags/config examples)
   - `configs/systems/README.md` if adding a new system provider type
2. Run a smoke evaluation with low limit:

```bash
# OpenAI-compatible example
benchy eval --config configs/models/openai_gpt-5-mini.yaml --limit 2

# System provider example
benchy eval --config configs/systems/surus-remove-background.yaml --limit 2
```

3. Verify:
   - connection test passes
   - request mode is correct
   - result dicts follow the standard schema (`output`, `raw`, `error`, `error_type`)
