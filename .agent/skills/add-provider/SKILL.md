---
name: add-provider
description: Add a new inference provider to benchy. Covers both OpenAI-compatible endpoints (prefer this path) and custom HTTP systems (SURUS-style). Includes config files, interface code, CLI wiring, capability declarations, and smoke-test verification. Use when asked to add a new model provider, API endpoint, or custom system.
---
# Add Provider Skill

Two extension paths exist. Choose the right one before writing any code.

| Path | When to use |
|---|---|
| **A: OpenAI-compatible** | Endpoint speaks chat/completions in OpenAI style |
| **B: Custom HTTP (SURUS-style)** | Custom payload shape, task-specific API |

---

## Path A: OpenAI-Compatible Provider

### Step 1 — Provider config

Create `configs/providers/<name>.yaml`. Start from the template:

```bash
cp configs/templates/provider_openai.yaml configs/providers/<name>.yaml
```

Mandatory fields:

```yaml
base_url: "https://api.example.com/v1"
api_key_env: "MY_API_KEY"
timeout: 120
max_retries: 3
max_concurrent: 3
temperature: 0.0
max_tokens: 2048
max_tokens_param_name: "max_tokens"
api_endpoint: "auto"     # auto | chat | completions

capabilities:
  supports_multimodal: false
  supports_schema: false
  supports_logprobs: false
  supports_files: false
  supports_streaming: false
  request_modes:
    - chat
```

### Step 2 — Model config (optional)

If the model needs specific overrides, create `configs/models/<name>.yaml`:

```yaml
model:
  name: "org/model-name"

openai:
  provider_config: <name>   # references configs/providers/<name>.yaml
  overrides:
    max_tokens: 4096
    temperature: 0.0

tasks:
  - latam_board
```

### Step 3 — CLI wiring (only if adding a new provider TYPE)

If this is a completely new provider type (not one of the existing ones below), update `src/benchy_cli_eval.py`.

Existing model provider types: `openai`, `anthropic`, `together`, `alibaba`, `google`
Existing system provider types: `surus`, `surus_ocr`, `surus_factura`, `surus_classify`, `surus_remove_background`
Special: `api` — generic API benchmarking via `--api-url` (no config file needed, see evaluate skill)

Steps:

1. Add to `PROVIDER_SPECS`:
   ```python
   "myprovider": {
       "config_key": "myprovider",
       "log": "Using MyProvider for model: {model_name}",
   },
   ```
2. Add to `MODEL_PROVIDER_TYPES` (if it's a model provider, not a system provider).
3. Add to `CLI_PROVIDER_DEFAULTS` with sensible defaults.
4. Add to `--provider` choices in `add_eval_arguments`.

### Step 4 — No interface needed

`OpenAIInterface` handles all OpenAI-compatible providers. Skip custom interface code.

### Step 5 — Smoke test

```bash
benchy eval --config configs/models/<name>.yaml --limit 2 --exit-policy smoke
```

---

## Path B: Custom HTTP Provider (SURUS-style)

### Step 1 — System config

Create `configs/systems/<name>.yaml`:

```yaml
system_name: "my-system"
provider_type: "my_system"

my_system:
  endpoint: "https://api.example.com/v1/action"
  api_key_env: "MY_SYSTEM_API_KEY"
  timeout: 60
  max_retries: 3
  max_concurrent: 5
  capabilities:
    supports_multimodal: true
    supports_schema: false
    supports_files: true
    supports_logprobs: false
    request_modes:
      - raw_payload

model:
  name: "my-system"

tasks:
  - my_task
```

### Step 2 — Interface class

Create `src/interfaces/my_system/my_system_interface.py`:

```python
from ..http_interface import HTTPInterface

class MySystemInterface(HTTPInterface):
    def prepare_request(self, sample, task):
        """Build the custom payload from sample fields."""
        return {
            "input": sample.get("text"),
            # ... provider-specific fields
        }

    def _make_request_with_client(self, client, request):
        response = client.post(self.endpoint, json=request, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response):
        # Must return: {"output": ..., "raw": ..., "error": None}
        return {
            "output": response.get("result"),
            "raw": response,
            "error": None,
            "error_type": None,
        }

    def build_test_request(self):
        """Minimal health-check payload."""
        return {"input": "test"}
```

Export from `src/interfaces/my_system/__init__.py`:
```python
from .my_system_interface import MySystemInterface
__all__ = ["MySystemInterface"]
```

### Step 3 — Wire in connection.py

Edit `src/engine/connection.py`:

1. Add capability defaults in `PROVIDER_CAPABILITY_DEFAULTS`:
   ```python
   "my_system": {
       "supports_multimodal": True,
       "supports_schema": False,
       # ...
   },
   ```

2. Extend `build_connection_info(...)` to map provider config fields.

3. Extend `get_interface_for_provider(...)`:
   ```python
   elif provider_type == "my_system":
       from ..interfaces.my_system import MySystemInterface
       return MySystemInterface(connection_info)
   ```

### Step 4 — CLI wiring

Add to `PROVIDER_SPECS` in `src/benchy_cli_eval.py`:
```python
"my_system": {
    "config_key": "my_system",
    "log": "Using MySystem provider",
},
```

System providers are selected via `--config configs/systems/<name>.yaml`, not `--provider`.

### Step 5 — Smoke test

```bash
benchy eval --config configs/systems/<name>.yaml --limit 2 --exit-policy smoke
```

---

## Capabilities Reference

```yaml
supports_multimodal: false    # Accepts image inputs
supports_schema: false        # Enforces JSON schema on output
supports_logprobs: false      # Returns token log-probabilities
supports_files: false         # Accepts file uploads
supports_streaming: false     # Supports streaming responses
request_modes:
  - chat           # OpenAI chat/completions
  - completions    # Legacy completions endpoint
  - raw_payload    # Custom non-chat request
```

---

## Files Checklist

### Path A (OpenAI-compatible)
- [ ] `configs/providers/<name>.yaml`
- [ ] `configs/models/<name>.yaml` (if model-specific)
- [ ] `src/benchy_cli_eval.py` (only if new provider TYPE)

### Path B (Custom HTTP)
- [ ] `configs/systems/<name>.yaml`
- [ ] `src/interfaces/<name>/<name>_interface.py`
- [ ] `src/interfaces/<name>/__init__.py`
- [ ] `src/engine/connection.py` (PROVIDER_CAPABILITY_DEFAULTS + routing)
- [ ] `src/benchy_cli_eval.py` (PROVIDER_SPECS)

---

## Response Dict Contract

Every interface `_parse_response` must return:
```python
{
    "output": str | dict | None,   # model's answer
    "raw": Any,                    # unmodified provider response
    "error": str | None,           # error message if failed
    "error_type": str | None,      # "connectivity", "invalid_response", etc.
}
```

---

## Image Handling

Do NOT duplicate image preprocessing. Use shared helpers:
```python
from src.interfaces.common.image_preprocessing import (
    encode_image_base64,
    encode_image_data_url,
    load_pil_image,
)
```

Pass `image_max_edge` from provider config for in-memory resizing. Never modify original files.

**Always set `image_max_edge` in the provider config when the endpoint has a pixel-size limit.**
Example — SURUS /factura has a 2560px limit:
```yaml
surus_factura:
  image_max_edge: 2048   # hard API limit is 2560px; stay below it
  force_jpeg_payload: true
  jpeg_quality: 90
```
Without this, images from real datasets are often 3000–5000px and will get HTTP 400 errors
that manifest as `all_invalid_responses` in `run_outcome.json`.

---

## Ground Truth Examples

- OpenAI-compatible: `configs/providers/openai.yaml`, `src/interfaces/openai_interface.py`
- Custom HTTP: `configs/systems/surus-remove-background.yaml`, `src/interfaces/surus/surus_remove_background_interface.py`
