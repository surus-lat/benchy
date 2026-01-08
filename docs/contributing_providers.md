# Contributing Providers

Providers define how Benchy talks to an AI system. A provider includes a config file
(defaults, timeouts, capability flags), an interface implementation, and registry
wiring so the engine can select it automatically.

## When to Add a Provider

Add a provider when the API shape or authentication model is meaningfully different
from what existing interfaces already support. If the request/response format can be
expressed with an existing interface (for example, OpenAI-compatible endpoints), you
likely only need a new provider config, not a new interface.

## Steps

### 1. Create a Provider Config

Add a file under `configs/providers/` with defaults and capability flags. Start from
`configs/templates/provider_openai.yaml` or `configs/templates/provider_vllm.yaml` and
adjust values like `base_url`, `api_key_env`, `timeout`, and `max_retries`. This file is
merged with per-model overrides via `provider_config` in model configs.

### 2. Implement an Interface

Create a new interface in `src/interfaces/` that supports:
- `prepare_request(sample, task)`
- `generate_batch(requests)`
- `test_connection()`

Interfaces are responsible for turning task samples into provider requests and returning
a normalized response dict (`output`, `raw`, `error`, `error_type`). Follow the response
format documented in `src/interfaces/README.md` so the engine and metrics pipeline stay
consistent.

### 3. Register the Interface

If you want auto-selection based on `provider_type`, register the interface in
`get_interface_for_provider(...)` (see `src/engine/connection.py`). This is the factory
used by `TaskGroupRunner` to create the correct interface for the run.

### 4. Wire Provider Config Loading

If the provider is used in model configs (not just system configs), update
`src/config_manager.py` to recognize the new provider section and merge the provider
config with overrides. This is how `provider_config` names map to files under
`configs/providers/`.

Also add defaults to `PROVIDER_CAPABILITY_DEFAULTS` in `src/engine/connection.py` so
compatibility checks have sensible fallbacks even when a provider config omits fields.

If the provider should be selectable via `eval.py`, add it to `PROVIDER_SPECS` and
update `provider_types` in each task's `task.json` as needed.

### 5. Declare Capabilities

Set capabilities in the provider config under `capabilities`. Model configs can further
restrict capabilities using `metadata.supports_*` tags, which are mapped into
`model_capabilities`. Tasks declare requirements in `capability_requirements`, and Benchy
blocks incompatible runs at startup.

Request modes (`chat`, `completions`, `raw_payload`) are also part of the capability
matrix, so be explicit about what your provider supports.

### 6. Document It

Update the README and add a template in `configs/templates/` so others can reuse your
provider quickly. If the provider is system-only, add a system config example under
`configs/systems/` and update `configs/systems/README.md`.

## Testing

Run a small task with a low limit and verify connection checks:

```bash
python eval.py --config configs/tests/spanish-gptoss.yaml --limit 2
```

For custom endpoints, use `configs/templates/system_http.yaml` as a starting point.
