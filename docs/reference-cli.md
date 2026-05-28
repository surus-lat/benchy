# CLI Reference

Complete reference for the `benchy` command-line interface.

## Commands

| Command | Description |
|---------|-------------|
| `benchy eval` | Run a benchmark evaluation |
| `benchy probe` | Test model capabilities without a full evaluation |
| `benchy tasks` | List available tasks and task groups |
| `benchy providers` | List available providers (from `configs/providers/`) |
| `benchy models` | List available model configs (from `configs/models/`) |
| `benchy datasets` | List auto-discovered datasets in `.data/` |

---

## `benchy eval`

Run a benchmark evaluation against a model or API system.

### Synopsis

```bash
benchy eval [--config CONFIG] [--model-name MODEL] [--tasks TASKS...] [OPTIONS]
```

### Provider selection

One of the following is required to identify the model or system:

| Flag | Description |
|------|-------------|
| `--config CONFIG` | Path or name of a config file (searches `configs/models/`, `configs/systems/`, etc.) |
| `--provider PROVIDER` | Provider name: `openai`, `anthropic`, `together`, `alibaba`, `google`, `vllm` |
| `--model-name MODEL` | Model identifier sent in API requests (e.g., `gpt-4o-mini`) |
| `--model-path PATH` | Local model path for vLLM (triggers local inference) |
| `--base-url URL` | OpenAI-compatible endpoint URL (e.g., `http://localhost:8000/v1`) |
| `--api-key KEY` | Explicit API key value (overrides env lookup) |
| `--api-key-env VAR` | Environment variable name to read the API key from |
| `--vllm-config NAME` | vLLM configuration name from `configs/templates/` (e.g., `vllm_two_cards_mm`) |

### Task selection

| Flag | Description |
|------|-------------|
| `--tasks TASKS...` | Space- or comma-separated task names (e.g., `--tasks spanish portuguese`) |
| `--tasks-file FILE` | Path to a text file with one task per line (comments with `#` are allowed) |
| `--task-group GROUP` | Task group name from `configs/config.yaml` (repeatable, e.g., `--task-group latam_board`) |

Task names can be:
- Group name only: `spanish` (runs all subtasks in the group)
- Group + subtask: `spanish.copa_es` (runs only that subtask)
- Task group alias: `latam_board` (expands to `spanish portuguese translation structured_extraction`)

### Dataset selection

| Flag | Description |
|------|-------------|
| `--dataset NAME` | **(Deprecated)** Use `--dataset-name` instead |
| `--dataset-name NAME` | Dataset name for zero-code CLI evaluation (HuggingFace or `.data/`) |
| `--dataset-source SOURCE` | Source: `auto`, `huggingface`, `local`, `parquet`, `directory` (default: `auto`) |
| `--dataset-split SPLIT` | HuggingFace dataset split (default: `test`) |
| `--dataset-input-field FIELD` | Input text field name (default: `text`) |
| `--dataset-output-field FIELD` | Expected output field name (default: `expected`/`label`) |
| `--dataset-id-field FIELD` | Sample ID field (auto-generated if missing) |
| `--dataset-labels JSON` | Label mapping for classification: `'{"0": "No", "1": "Yes"}'` |
| `--dataset-label-field FIELD` | Label field name (default: `label`) |
| `--dataset-choices-field FIELD` | Field containing per-sample answer choices (for classification) |
| `--dataset-schema-field FIELD` | Schema field in dataset (for structured extraction) |
| `--dataset-schema-path PATH` | JSON file containing schema (for structured extraction) |
| `--dataset-schema-json JSON` | Inline JSON schema string |

### Task type (zero-code mode)

Required when using `--dataset-name` without a task name:

| Flag | Description |
|------|-------------|
| `--task-type TYPE` | Task type: `classification`, `structured`, `freeform` |

### Multimodal and document rendering

| Flag | Description |
|------|-------------|
| `--multimodal-input` | Enable multimodal input (auto-enabled for binary parquet datasets) |
| `--multimodal-image-field FIELD` | Image path field (default: `image_path`) |
| `--render-documents` / `--no-render-documents` | Control PDF/TIFF to PNG rendering |
| `--render-dpi INT` | Rendering DPI for document-to-PNG conversion (default: 200) |
| `--render-max-pages INT` | Maximum pages to render per document (default: 1) |
| `--image-max-edge INT` | Downscale images so the longest edge is at most this many pixels before sending |

### Generic API mode

For benchmarking arbitrary HTTP endpoints directly:

| Flag | Description |
|------|-------------|
| `--api-url URL` | Target endpoint URL (sets provider to `api`) |
| `--api-body-template JSON` | JSON body template with `{{field}}` placeholders |
| `--api-response-path PATH` | Dot-notation path to extract output (e.g., `choices.0.message.content`) |
| `--api-method METHOD` | HTTP method (default: `POST`) |
| `--api-headers JSON` | Extra HTTP headers as a JSON object |

Template placeholder syntax in `--api-body-template`:
- `{{field}}` — plain string substitution from the dataset sample
- `{{field|base64_image}}` — reads an image file and encodes it as a base64 data URL
- `{{field|json}}` — embeds the value as raw JSON (preserves dicts/lists)

### Prompt customization

| Flag | Description |
|------|-------------|
| `--system-prompt TEXT` | Custom system prompt for zero-code task types |
| `--user-prompt-template TEXT` | Template with `{field}` placeholders for user message |

### Run control

| Flag | Description |
|------|-------------|
| `--limit N` | Maximum samples per task (useful for smoke tests) |
| `--run-id NAME` | Custom run folder name (default: timestamp-based) |
| `--output-path PATH` | Base path for output (default: `outputs/benchmark_outputs`) |
| `--log-samples` | Force sample-level logging for all tasks |
| `--no-log-samples` | Disable sample-level logging for all tasks |
| `--batch-size INT` | Batch size for task runners (default: task-level default, usually 20) |
| `--test` | Start vLLM server without running tasks (vLLM only) |
| `--exit-policy POLICY` | Process exit behavior: `relaxed`, `smoke`, `strict` |
| `--save-config PATH` | Save CLI parameters as a reusable YAML config file |
| `--verbose` / `-v` | Enable verbose logging |

### Model parameters

| Flag | Description |
|------|-------------|
| `--max-tokens INT` | Maximum output tokens per request |
| `--max-tokens-param-name NAME` | Override the max tokens parameter name (e.g., `max_completion_tokens`) |
| `--temperature FLOAT` | Sampling temperature |
| `--timeout INT` | Request timeout in seconds |
| `--max-retries INT` | Maximum retry attempts for API requests |
| `--max-concurrent INT` | Maximum concurrent API requests |
| `--api-endpoint MODE` | Request mode: `auto` (default), `chat`, `completions` |
| `--use-structured-outputs` / `--no-use-structured-outputs` | Enable/disable guided JSON schema via `extra_body.structured_outputs` (recommended for local vLLM) |
| `--probe-mode MODE` | Capability detection: `skip` (default, use inline probe) or `auto` (run full probe before eval) |
| `--compatibility MODE` | Incompatible task handling: `warn`, `skip`, or `error` |
| `--organization TEXT` | Organization metadata stored in run artifacts |
| `--url TEXT` | URL metadata stored in run artifacts |

### Exit policies

| Policy | Behavior |
|--------|----------|
| `relaxed` | Always exits 0, even on failures (default for manual runs) |
| `smoke` | Exits non-zero if any task is not `passed` (recommended for CI smoke tests) |
| `strict` | Exits non-zero if any task is not `passed` or `degraded` |

### Examples

```bash
# Smoke test: fast validation with 5 samples
benchy eval --provider openai --model-name gpt-4o-mini --tasks spanish --limit 5

# Full evaluation from a config file
benchy eval --config configs/models/openai_gpt-4o-mini.yaml

# Override tasks from a config
benchy eval --config openai_gpt-4o-mini.yaml --tasks spanish portuguese

# Local vLLM from a model directory
benchy eval --model-path /models/my-sft --model-name my-sft \
  --vllm-config vllm_two_cards_mm --tasks latam_board --limit 10

# Hugging Face model via vLLM
benchy eval --model-name unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit \
  --vllm-config vllm_two_cards_mm --tasks latam_board

# Together AI provider
benchy eval --provider together --model-name meta-llama/Llama-3.1-8B-Instruct \
  --tasks spanish --limit 2

# Zero-code classification from HuggingFace
benchy eval --provider openai --model-name gpt-4o-mini \
  --task-type classification \
  --dataset-name climatebert/environmental_claims \
  --dataset-labels '{"0": "No", "1": "Yes"}' --limit 10

# Generic API endpoint
benchy eval --api-url "https://api.example.com/extract" \
  --api-key-env MY_API_KEY \
  --api-body-template '{"image": "{{image_path|base64_image}}"}' \
  --api-response-path "data" \
  --tasks document_extraction --model-name "my-extractor" --limit 5

# CI automation: smoke policy
benchy eval --config openai_gpt-4o-mini.yaml --tasks spanish \
  --limit 5 --exit-policy smoke
```

---

## `benchy probe`

Test model capabilities without running a full evaluation. The probe detects which
API endpoints, schema transports, multimodal inputs, and parameter variants the model
actually supports. `benchy eval` runs this automatically, but you can run it standalone
to debug configuration issues.

### Synopsis

```bash
benchy probe --model-name MODEL [--provider PROVIDER] [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--model-name MODEL` | **(Required)** Model identifier to probe |
| `--provider PROVIDER` | Provider name: `openai`, `anthropic`, `together`, `alibaba`, `vllm` |
| `--base-url URL` | OpenAI-compatible endpoint URL |
| `--api-key KEY` | Explicit API key |
| `--api-key-env VAR` | Environment variable for the API key |
| `--profile PROFILE` | Probe profile: `quick` (default, 30–60s) |
| `--global-timeout INT` | Maximum probe duration in seconds (default: 180) |
| `--image-max-edge INT` | Max image edge for multimodal capability test (optional) |
| `--run-id NAME` | Custom run folder name |
| `--output-path PATH` | Base path for probe output (default: `outputs/probe_outputs`) |

### What the probe detects

1. **Access readiness** — invalid API key, model not found, insufficient quota
2. **API endpoints** — chat, completions, logprobs support
3. **Schema transports** — `structured_outputs` vs `response_format` support
4. **Multimodal support** — whether the model accepts image inputs
5. **Truncation behavior** — repetition patterns at token limits
6. **Max tokens parameter** — `max_tokens` vs `max_completion_tokens` (critical for newer OpenAI models)
7. **Provider fingerprint** — server metadata and version

### Outputs

Probe results are written to `outputs/probe_outputs/<run_id>/<model>/`:
- `probe_report.json` — machine-readable capability report
- `probe_summary.txt` — human-readable summary

### Examples

```bash
# Probe an OpenAI model
benchy probe --provider openai --model-name gpt-4o-mini

# Probe a local vLLM endpoint
benchy probe --base-url http://localhost:8001/v1 --model-name mymodel

# Probe with longer timeout
benchy probe --provider openai --model-name gpt-5-mini \
  --profile quick --global-timeout 180
```

---

## `benchy tasks`

List available tasks and task groups discovered by Benchy.

### Synopsis

```bash
benchy tasks [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--verbose` / `-v` | Expand task groups to show individual subtasks |
| `--json` | Machine-readable JSON output |

### Examples

```bash
# List all task groups and tasks
benchy tasks

# Show subtasks within each group
benchy tasks --verbose

# JSON output for scripts
benchy tasks --json
```

---

## `benchy providers`

List available providers discovered from `configs/providers/`.

### Synopsis

```bash
benchy providers [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |

### Examples

```bash
benchy providers
benchy providers --json
```

---

## `benchy models`

List available model configs discovered from `configs/models/`.

### Synopsis

```bash
benchy models [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |

### Examples

```bash
benchy models
benchy models --json
```

---

## `benchy datasets`

List auto-discovered datasets in the `.data/` directory.

### Synopsis

```bash
benchy datasets [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--verbose` | Show detailed information including schemas and labels |
| `--json` | Machine-readable JSON output |

### Examples

```bash
# List all datasets
benchy datasets

# Detailed view with schemas and labels
benchy datasets --verbose

# Machine-readable output for scripts
benchy datasets --json
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `TOGETHER_API_KEY` | Together AI API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope API key |
| `SURUS_API_KEY` | Surus AI API key |
| `BENCHY_ENABLE_PREFECT` | Set to `1` to enable Prefect workflow tracking |
| `BENCHY_EXTRAS` | Comma-separated extras for `setup.sh` (e.g., `local,translation`) |
| `BENCHY_SKIP_DATASET` | Set to `1` in `setup.sh` to skip dataset download |

---

## Related docs

- [Tutorial: Getting Started](tutorial-getting-started.md) — First evaluation walkthrough
- [Config Format Reference](reference-config.md) — YAML config file format
- [Task Catalog](reference-tasks.md) — Available tasks and what they measure
- [Output Artifacts Reference](reference-output-artifacts.md) — Output file schemas
- [Evaluating Models](evaluating_models.md) — Full usage guide with examples
