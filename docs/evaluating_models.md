# Evaluating Models and Systems

This guide explains how to run Benchy evaluations against both general-purpose models
(vLLM or cloud providers) and task-optimized AI systems (custom endpoints). Benchy is
entirely config-driven, so once you understand the config layout you can run any task
suite without touching Python code.

## Choose a Config

Benchy always runs from a single config file that defines the model or system and the
tasks you want to evaluate. The config also tells Benchy which provider config to use
and which defaults to apply across tasks.

- `configs/models/` for general-purpose models (vLLM, OpenAI, Anthropic, Together)
- `configs/systems/` for task-optimized systems (custom HTTP or Surus endpoints)
- `configs/tests/` for small smoke-test configs

Templates live in `configs/templates/` and include commented examples you can copy
and adapt for your own models or systems.

## Run an Evaluation

```bash
benchy eval --config configs/tests/spanish-gptoss.yaml --limit 10
```

Cloud example:

```bash
benchy eval --config openai_gpt-4o-mini.yaml --limit 10
```

Local model (vLLM) example:

```bash
benchy eval --model-path /path/to/local-model --model-name my-sft --vllm-config vllm_two_cards_mm --tasks latam_board --limit 10
```

Local model (vLLM) with outputs next to the model:

```bash
benchy eval --model-path /path/to/local-model --output-path model --tasks latam_board --limit 10
```

Hugging Face model via vLLM (no model config file) example:

```bash
benchy eval --model-name unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit --vllm-config vllm_two_cards_mm --tasks latam_board --limit 10
```

If you are using cloud providers, create a `.env` file from `env.example` and set the
API keys. You can also set them directly in your shell environment.

Useful flags:
- `--limit N` limits the number of samples per task for fast smoke tests.
- `--run-id NAME` sets a custom run folder name.
- `--log-samples` forces sample logging for all tasks.
- `--test` is only supported for vLLM and starts the server without running tasks.
- `--tasks a,b,c` overrides the task list from the config.
- `--tasks-file path.txt` loads tasks from a file (one per line, comments allowed).
- `--task-group name` injects task groups from `configs/config.yaml` (repeatable).

## Model Config Structure

A model config selects a provider config and optional overrides. The provider config
contains defaults (timeouts, retries, capabilities), while overrides are per-model
settings that change those defaults without copy-pasting the whole provider file.

```yaml
model:
  name: "org/model-name"

vllm:
  provider_config: vllm_single_card
  overrides:
    max_model_len: 8192

# or
openai:
  provider_config: openai
  overrides:
    max_tokens: 2048
```

Tasks can be listed directly or by group names defined in `configs/config.yaml`. This
lets you reuse the `latam_board` group without spelling out each task every time.

```yaml
tasks:
  - "latam_board"
  - "spanish"
```

You can also add `task_defaults` and `metadata.supports_*` tags in the model config.
Task defaults apply to every task, and metadata tags restrict capabilities so Benchy
won't attempt unsupported features for that model.

If you override tasks with CLI flags, those overrides replace the config task list for
model providers. For system providers (custom endpoints), overrides are intersected with
the tasks declared in the system config so you cannot run unsupported tasks by accident.

## System Config Structure

Systems define a `provider_type` and a provider section. These configs are for task
specific endpoints that don't behave like normal LLMs, such as OCR or extraction APIs.

```yaml
system_name: "my-system"
provider_type: "http"

http:
  endpoint: "https://api.example.com/v1/endpoint"
  api_key_env: "MY_API_KEY"
  capabilities:
    supports_schema: true

model:
  name: "my-system"

tasks:
  - "structured_extraction"
```

Systems usually run a smaller set of tasks, and their provider section must describe
capabilities clearly so Benchy can block incompatible evaluations early.

## Capabilities and Compatibility

Provider configs declare stack capabilities under `capabilities`. Model configs can
add `metadata.supports_*` tags to restrict those capabilities. Task configs declare
requirements using `capability_requirements`. Benchy combines all three and blocks
runs that are incompatible (for example, a multimodal task on a text-only model or a
logprobs-required task on a provider that does not support logprobs).

Compatibility checks happen in the runner and are logged at startup so you can see
exactly why a task was skipped.

## Outputs

Results are written under:

```
outputs/benchmark_outputs/<run_id>/<model_name>/
```

Each task folder contains metrics, samples, and a report file. The pipeline also
writes `run_summary.json` at the model root with a summary of task results and the
overall run status.

## Tips

- Use small limits to validate configs and providers before full runs.
- Keep API keys in `.env` and never commit them.
- If a task is skipped, check the log for capability mismatch details.
