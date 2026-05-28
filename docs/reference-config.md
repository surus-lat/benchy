# Config Format Reference

Complete reference for Benchy YAML configuration files. Configs live in the `configs/`
directory and are selected with `--config` on the CLI.

## Config types

| Location | Purpose |
|----------|---------|
| `configs/models/*.yaml` | General-purpose models (OpenAI, Anthropic, Together, vLLM) |
| `configs/systems/*.yaml` | Task-specific system endpoints (custom HTTP, Surus) |
| `configs/providers/*.yaml` | Provider defaults (not referenced directly by users) |
| `configs/templates/*.yaml` | Documented starting-point templates |
| `configs/tests/*.yaml` | Small smoke-test configs |
| `configs/config.yaml` | Global settings, task groups, leaderboard config |

---

## Model config (`configs/models/*.yaml`)

Use for general-purpose language models served via OpenAI, Anthropic, Together AI, or
local vLLM.

```yaml
model:
  name: "org/model-name"          # String sent in API requests

# Provider block — choose one: openai, anthropic, together, vllm
openai:
  provider_config: openai         # References configs/providers/openai.yaml
  overrides:
    temperature: 0.0
    max_tokens: 2048
    max_concurrent: 10
    max_tokens_param_name: "max_completion_tokens"  # For newer OpenAI models

task_defaults:                    # Applied to every task in this run
  log_samples: true
  batch_size: 10

tasks:                            # Task list or task group names
  - "spanish"
  - "latam_board"                 # Expands to spanish, portuguese, translation, structured_extraction

metadata:                         # Capability restrictions (can only restrict, not extend)
  supports_multimodal: true
  supports_logprobs: true
  supports_schema: true
```

### Provider blocks

**OpenAI:**
```yaml
openai:
  provider_config: openai
  overrides:
    temperature: 0.0
    max_tokens: 2048
    timeout: 120
    max_concurrent: 10
```

**Anthropic:**
```yaml
anthropic:
  provider_config: anthropic
  overrides:
    max_tokens: 4096
    timeout: 180
```

**Together AI:**
```yaml
together:
  provider_config: together
  overrides:
    temperature: 0.0
    max_tokens: 2048
```

**vLLM (two GPUs, multimodal):**
```yaml
vllm:
  provider_config: vllm_two_cards_mm
  overrides:
    max_model_len: 8192
    gpu_memory_utilization: 0.9
```

### `metadata` tags

Tags in `metadata` can only restrict capabilities declared in the provider config.
Adding `supports_multimodal: false` prevents multimodal tasks from running on a
text-only model. Adding `supports_multimodal: true` has no effect if the provider
config does not declare multimodal support.

| Tag | Description |
|-----|-------------|
| `supports_multimodal` | Image inputs supported |
| `supports_logprobs` | Log-probability output supported |
| `supports_schema` | Structured output (JSON schema) supported |

---

## System config (`configs/systems/*.yaml`)

Use for task-specific endpoints that are not general-purpose LLMs — for example, an
OCR pipeline or document extraction API.

```yaml
system_name: "my-system"
provider_type: "http"             # or "surus", "surus_remove_background", etc.

http:                             # Provider section — name matches provider_type
  endpoint: "https://api.example.com/v1/endpoint"
  api_key_env: "MY_API_KEY"      # Environment variable name for the API key
  timeout: 60
  capabilities:
    supports_schema: true
    supports_multimodal: true

model:
  name: "my-system"              # Used as the label in output artifacts

tasks:                            # Tasks this system can run (intersection on CLI override)
  - "structured_extraction"
  - "document_extraction"
```

### `http` provider fields

| Field | Type | Description |
|-------|------|-------------|
| `endpoint` | string | Full URL of the API endpoint |
| `api_key_env` | string | Environment variable containing the API key |
| `timeout` | int | Request timeout in seconds |
| `capabilities.supports_schema` | bool | Endpoint returns structured JSON |
| `capabilities.supports_multimodal` | bool | Endpoint accepts image inputs |

---

## Provider config (`configs/providers/*.yaml`)

Provider configs contain shared defaults for an API stack. You rarely edit these
directly — reference them from model configs via `provider_config:`.

```yaml
# Example: configs/providers/openai.yaml (abbreviated)
provider_type: openai

capabilities:
  supports_multimodal: true
  supports_logprobs: true
  supports_schema: true
  supports_files: false
  supports_streaming: false

defaults:
  temperature: 0.0
  max_tokens: 2048
  timeout: 120
  max_retries: 3
  max_concurrent: 10
  batch_size: 10
```

---

## Global config (`configs/config.yaml`)

Central settings for the entire Benchy installation.

```yaml
paths:
  benchmark_outputs: "outputs/benchmark_outputs"
  reference_dir: "reference"
  publish_dir: "outputs/publish"
  logs: "logs"

evaluation:
  default_limit: null             # null = no limit; override with --limit on CLI

gpu_config:
  vllm:
    devices: "1,2"               # GPUs reserved for the vLLM server
  tasks:
    devices: "0"                 # GPU for evaluation tasks (empty = CPU only)
  validation:
    check_gpu_availability: true
    allow_overlap: true          # Allow vLLM and task GPUs to overlap

task_groups:
  latam_board:
    description: "Complete LATAM evaluation suite"
    tasks:
      - "spanish"
      - "portuguese"
      - "translation"
      - "structured_extraction"
  # Add custom groups here

leaderboard:
  overall_score_categories:
    - "latam_es"
    - "latam_pr"
    - "translation"
    - "structured_extraction"
  normalize_scores:
    - "translation"              # Divide scores by 100 (for CHRF 0-100 → 0-1)
```

---

## Capability compatibility

Benchy enforces capability compatibility at startup. The logic works as follows:

1. **Provider config** declares the baseline capabilities of the API stack.
2. **Model config `metadata`** can restrict those capabilities for a specific model.
3. **Task metadata.yaml** declares `capability_requirements` at three levels:
   - `required` — task is skipped with a clear log if the capability is missing
   - `preferred` — task runs without the capability but a warning is logged
   - `optional` — task uses the capability if available, silently degrades without it

Example: a multimodal task (`required: supports_multimodal`) on a text-only model
(`supports_multimodal: false`) will be skipped, not silently degraded. You'll see:
```
INFO - Skipping image_extraction.facturas: requires supports_multimodal
```

---

## Config resolution

When you pass `--config my-model.yaml`:

1. Benchy looks for the file by exact path first.
2. If not found, it searches `configs/models/`, `configs/systems/`, `configs/tests/`,
   and `configs/templates/` for a file named `my-model.yaml` or `my-model`.
3. For model configs, the `openai.provider_config` value (e.g., `openai`) is resolved
   to `configs/providers/openai.yaml`, and its contents are deep-merged under the
   model config's provider block. Model-level `overrides:` win.

---

## Template reference

Copy a template and adapt it for your use case:

| Template | Use for |
|----------|---------|
| `configs/templates/provider_openai.yaml` | Cloud providers (OpenAI, Anthropic, Together) |
| `configs/templates/provider_vllm.yaml` | Local vLLM inference |
| `configs/templates/system_http.yaml` | Custom HTTP API endpoints |
| `configs/templates/test-model_minimal.yaml` | Minimal model config |
| `configs/templates/test-model_new.yaml` | Full model config with comments |
| `configs/templates/test-model_two_cards.yaml` | Two-GPU vLLM config |

---

## Related docs

- [CLI Reference](reference-cli.md) — All CLI flags
- [Evaluating Models](evaluating_models.md) — Config usage in practice
- [Architecture](architecture.md) — How configs are loaded and merged
