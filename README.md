<h1 align="center"><strong>Benchy</strong></h1>
<p align="center"> A benchmarking engine for evaluating AI systems on task-specific performance.</p>

<p align="center">
    <img src="./docs/benchy_2.png" alt="readme_image" style="width:220px;height:220px;" />
</p>

Benchy is a benchmarking suite for evaluating AI systems (models, hosted endpoints, or task-specific
pipelines) on task-specific performance. It currently powers the [LatamBoard](https://latamboard.ai/), so if
you are here to browse results, the leaderboard site is the best starting point. If you are using
Benchy to run evaluations (not contributing code), start with `docs/evaluating_models.md` for the
step-by-step usage guide.

## What Benchy Offers

- **AI systems first**: Evaluate general models and task-optimized endpoints with the same task suite.
- **Task/interface decoupling**: Tasks define data and metrics; interfaces handle provider IO.
- **Local or cloud**: Start vLLM automatically for local runs or use cloud providers via configs.
- **Reproducible outputs**: Organized run folders with task summaries and metadata.
- **Contributor-friendly**: Add tasks or providers without rewriting the rest of the system.

## How Benchy Works

- **Tasks** are built using **format handlers** (MultipleChoice, Structured, Freeform, Multimodal)
- **Handlers** provide data loading, prompt formatting, metrics, and capability checking
- **Interfaces** translate task samples into provider-specific requests
- **TaskGroupRunner** builds connection info, instantiates tasks, and dispatches to the engine
- **BenchmarkRunner** batches requests, retries failures, and aggregates metrics

This design lets you add a new task with ~30-50 lines of code (vs. 200-400 in the old system),
and add a new provider without changing evaluation logic. Tasks focus on **what to evaluate**,
while handlers and interfaces handle **how to evaluate it**.

## Quickstart (Developers)

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU(s) for local vLLM (not required for cloud providers)
- Docker (optional, for Prefect UI)
- [uv](https://github.com/astral-sh/uv) (recommended, but optional - traditional venv + pip also works)

### Install

**Option 1: Using the setup script (recommended)**

```bash
bash setup.sh
```

This will:
- Create a virtual environment (`.venv`)
- Install all dependencies
- Optionally download structured extraction dataset

Optional extras (comma-separated) and dataset skip:

```bash
BENCHY_EXTRAS=translation,prefect BENCHY_SKIP_DATASET=1 bash setup.sh
```

**Option 2: Manual setup with uv (recommended for developers)**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

**Option 3: Manual setup with traditional venv + pip**

```bash
# Create virtual environment (use Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -e .
```

**Optional extras**

Prefect orchestration (optional):

```bash
pip install -e '.[prefect]'
# or with uv:
uv sync --extra prefect
```

Translation metrics (only for translation tasks):

```bash
pip install '.[translation]'
# or with uv:
uv sync --extra translation
```

**Environment setup**

If you use cloud providers, copy `env.example` to `.env` and fill in API keys:

```bash
cp env.example .env
# Edit .env with your API keys
```

### Prefect UI (Optional)

Prefect is disabled by default; enable it with `BENCHY_ENABLE_PREFECT=1` to automatically track
runs in the Prefect UI. Install the extra dependency first.

```bash
# Start Prefect server (if not already running)
docker run -p 4200:4200 -d --rm prefecthq/prefect:3-python3.12 prefect server start --host 0.0.0.0

# Enable Prefect tracking (runs will automatically appear in UI)
export BENCHY_ENABLE_PREFECT=1
benchy eval configs/models/your_model.yaml --tasks document_extraction --limit 10
```

**Note:** The `--register` flag is for deploying flows as long-running workers (different use case).
When `BENCHY_ENABLE_PREFECT=1` is set, runs are automatically tracked in the UI without needing `--register`.

### Run a First Benchmark

```bash
# Local model via vLLM (merged safetensors + tokenizer assets)
benchy eval --model-path /path/to/local-model --model-name my-sft --vllm-config vllm_two_cards_mm --tasks latam_board --limit 10

# Local model via vLLM, writing outputs next to the model
benchy eval --model-path /path/to/local-model --output-path model --tasks latam_board --limit 10

# Hugging Face model via vLLM (no config file needed)
benchy eval --model-name unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit --vllm-config vllm_two_cards_mm --tasks latam_board --limit 10

# Config-based smoke test (limited samples)
benchy eval --config configs/tests/spanish-gptoss.yaml --limit 10

# Cloud example (config name lookup searches under configs/models, configs/systems, etc.)
benchy eval --config openai_gpt-4o-mini.yaml --limit 10
```

If `benchy` is not on your PATH (for example when running directly from the repo), use:
`python -m src.benchy_cli ...`

If you want to run the same task list across multiple models, you can override tasks
on the command line with `--tasks`, `--tasks-file`, or `--task-group`. See
`docs/evaluating_models.md` for full examples and behavior.

## Providerless CLI (OpenAI-compatible)

When no config file is provided, Benchy infers the provider from CLI flags:

- `--model-path` or `--vllm-config` -> local vLLM (Benchy starts the server)
- `--base-url` -> OpenAI-compatible remote endpoint (defaults to OpenAI behavior unless `--provider` is set)
- no provider hints -> OpenAI defaults (`https://api.openai.com/v1`, `OPENAI_API_KEY`)

This means the model name alone does not determine the provider. The provider comes from
flags like `--provider` and `--base-url`. The model name is just the string sent in requests.

### Common use cases

```bash
# OpenAI default (model name + OPENAI_API_KEY)
benchy eval --model-name gpt-4o-mini --tasks spanish --limit 2

# Together AI defaults (TOGETHER_API_KEY + together base URL)
benchy eval --model-name meta-llama/Llama-3.1-8B-Instruct --provider together  --tasks spanish --limit 2

# Custom OpenAI-compatible endpoint
benchy eval --model-name mymodel --base-url http://host:8000/v1 --tasks spanish --limit 2

# Local vLLM from Hugging Face (server started by Benchy)
benchy eval --model-name meta-llama/Llama-3.1-8B-Instruct --provider vllm  --vllm-config vllm_two_cards_mm --tasks spanish --limit 2

# Local vLLM from a model directory
benchy eval --model-name my-sft --model-path /models/my-sft --vllm-config vllm_two_cards_mm --tasks spanish --limit 2
```

### Same model name on multiple providers

If a model is available on multiple providers (or a local vLLM server), you choose where it runs:

```bash
# Together-hosted model
benchy eval --model-name mymodel --provider together  --tasks spanish --limit 2

# OpenAI-hosted model
benchy eval --model-name mymodel --provider openai  --tasks spanish --limit 2

# Local vLLM for the same model name
benchy eval --model-name mymodel --provider vllm  --vllm-config vllm_two_cards_mm --tasks spanish --limit 2
```

### Benchmarking SURUS AI nodes
Surus AI nodes are preconfigured with their relevant tasks. Remember to add the necesary SURUS_API_KEY in your .env

```bash
# surus extraction endpoint
benchy eval --config surus-extract --limit 5

# Surus classification endpoint
benchy eval --config surus-classify --limit 5
```
### Benchmarking a new OpenAI model (example: "gpt-5.2")

```bash
benchy eval --provider openai --model-name gpt-5.2 --tasks spanish --limit 2
```

If the model requires a nonstandard max-tokens parameter or API key name, set:

```bash
benchy eval --provider openai --model-name gpt-5.2 \
  --max-tokens-param-name max_completion_tokens \
  --api-key-env OPENAI_API_KEY \
  --tasks spanish --limit 2
```

## Configuration Overview

### Project Structure

```
benchy/
├── configs/
│   ├── config.yaml          # Global settings and task groups
│   ├── models/              # Model configs (vLLM or cloud)
│   ├── systems/             # Task-optimized system configs
│   ├── providers/           # Provider defaults (vLLM, OpenAI, etc.)
│   ├── templates/           # Fully documented config templates
│   └── tests/               # Small configs for smoke tests
├── src/
│   ├── benchy_cli.py         # Benchy CLI entrypoint (`benchy ...`)
│   ├── pipeline.py          # Main Prefect pipeline
│   ├── interfaces/          # Provider interfaces
│   ├── tasks/
│   │   ├── common/          # Format handlers and shared utilities
│   │   ├── spanish/         # Spanish language tasks
│   │   ├── portuguese/      # Portuguese language tasks
│   │   ├── structured_extraction/  # JSON extraction tasks
│   │   ├── image_extraction/       # Vision-language tasks
│   │   └── _template_handler/      # Task templates
│   └── leaderboard/         # Results processing
└── eval.py                  # Legacy CLI wrapper (deprecated)
```

### Model and System Configs

- **Models** live in `configs/models/` and include a provider block (`vllm`, `openai`, `anthropic`, `together`).
- **Systems** live in `configs/systems/` and include `provider_type` plus a provider section (for custom APIs).
- **Task groups** (like `latam_board`) are defined in `configs/config.yaml` and can be used inside `tasks`.

### Capabilities and Compatibility

Provider configs declare capabilities (multimodal, logprobs, schema, files, etc.).
Model configs can add `metadata.supports_*` tags, which are mapped to `model_capabilities` and
can only *restrict* provider capabilities. Tasks declare required capabilities in their task config.
If a required capability is missing, the task is skipped with a clear log message.

## Results and Publishing

After runs finish, process results for the leaderboard:

```bash
python ./src/leaderboard/process_all.py
```

This generates per-model summaries and leaderboard tables under `outputs/publish/`.

## Documentation

### For Users
- `docs/evaluating_models.md` - Running benchmarks and understanding results

### For Contributors
- `docs/contribute_tasks.md` - **Adding new tasks with the handler system** (recommended read!)
- `docs/contributing_providers.md` - Adding new model providers
- `src/tasks/_template_handler/README.md` - Complete task examples and patterns

### Architecture & Internals
- `docs/architecture.md` - System design and component interaction
- `docs/GENERATION_CONFIG.md` - Generation parameters and sampling
- `docs/VLLM_VERSION_MANAGEMENT.md` - Managing vLLM versions

### Quick References
- `src/tasks/_template_handler/` - Copy-paste task templates
- `src/tasks/common/` - Handler classes with extensive documentation
- `configs/templates/` - Fully documented configuration examples

## Contributing

See `CONTRIBUTING.md` for workflow details and the docs above for task/provider guides.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- [Prefect](https://www.prefect.io/) for workflow orchestration
- [Surus](https://surus.lat/) for starting this project
- LATAM community for benchmark development
