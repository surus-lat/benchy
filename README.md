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

- **Tasks** define data loading, prompt formatting, metrics, and capability requirements.
- **Interfaces** translate task samples into provider-specific requests.
- **TaskGroupRunner** builds connection info, instantiates tasks, and dispatches to the engine.
- **BenchmarkRunner** batches requests, retries failures, and aggregates metrics.

This design lets you add a new task without reworking inference, and add a new provider without
changing evaluation logic. The architecture doc below goes deeper into how the engine, tasks, and
interfaces fit together if you want a more detailed mental model.

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

Prefect is disabled by default; enable it with `BENCHY_ENABLE_PREFECT=1` (or `--register`)
and install the extra dependency.

```bash
docker run -p 4200:4200 -d --rm prefecthq/prefect:3-python3.12 prefect server start --host 0.0.0.0
```

### Run a First Benchmark

```bash
# Local vLLM example (limited samples)
python eval.py --config configs/tests/spanish-gptoss.yaml --limit 10

# Cloud example
python eval.py --config configs/models/openai_gpt-4o-mini.yaml --limit 10
```

If you want to run the same task list across multiple models, you can override tasks
on the command line with `--tasks`, `--tasks-file`, or `--task-group`. See
`docs/evaluating_models.md` for full examples and behavior.

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
│   ├── pipeline.py          # Main Prefect pipeline
│   ├── interfaces/          # Provider interfaces
│   ├── tasks/               # Task implementations + task.json configs
│   └── leaderboard/         # Results processing
└── eval.py                  # CLI entry point
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

- `docs/evaluating_models.md`
- `docs/contribute_tasks.md`
- `docs/contributing_providers.md`
- `docs/architecture.md`
- `src/tasks/TASK_TEMPLATE.md` (deep task implementation details)
- `docs/GENERATION_CONFIG.md` and `docs/VLLM_VERSION_MANAGEMENT.md`

## Contributing

See `CONTRIBUTING.md` for workflow details and the docs above for task/provider guides.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- [Prefect](https://www.prefect.io/) for workflow orchestration
- [Surus](https://surus.lat/) for starting this project
- LATAM community for benchmark development
