<h1 align="center"><strong>Benchy</strong></h1>
<p align="center"> LATAM Leaderboard Benchmarking Suite </p>

<p align="center">
    <img src="./docs/benchy_2.png" alt="readme_image" style="width:220px;height:220px;" />
</p>

Benchy is a modular benchmarking suite for evaluating AI systems on Spanish and Portuguese tasks for the [LATAM Leaderboard](https://latamboard.ai/). It separates evaluation logic from inference providers so you can evolve tasks and serving stacks independently.

## What Benchy Offers

- **AI Systems Ready**: Run the same tasks across local servers or cloud providers.
- **Task/Interface Decoupling**: Tasks handle data and metrics; interfaces handle provider IO; the engine orchestrates.
- **Local or Cloud**: Starts vLLM automatically for local testing and supports cloud providers via API configs.
- **Reproducible Runs**: Organized outputs per run ID with aggregated summaries.
- **Contributor-Friendly**: Add tasks or providers without touching the other side.

## How Benchy Works

- **Tasks** define data loading, prompt formatting, metrics, and capability flags.
- **Interfaces** handle request/response translation for providers (vLLM, cloud APIs, etc.).
- **TaskGroupRunner** prepares subtasks, builds connection info, and invokes the engine.
- **BenchmarkRunner** pairs tasks with interfaces, managing batching, retries, and aggregation.

This design lets you add a new task without reworking inference, and add a new provider without changing evaluation logic.

### Task and Interface Interaction

Tasks are interface-agnostic: they expose `get_prompt()` and metric methods. Interfaces call
`task.get_prompt()` (LLM-style APIs) or read raw sample fields (HTTP-style APIs). The runner
connects them by building `connection_info` and selecting the right interface.

## Quickstart

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU(s) for local vLLM
- Docker for Prefect server (recommended)

### Installation

```bash
git clone <repository-url>
cd benchy
git submodule update --init --recursive

# Recommended setup
bash setup.sh

# Alternative (uv)
uv sync
```

### Start Prefect Server

```bash
sudo docker run -p 4200:4200 -d --rm prefecthq/prefect:3-python3.12 prefect server start --host 0.0.0.0
```

### Run a First Benchmark

```bash
python eval.py --config configs/single_card/qwen34b.yaml --limit 10
```

## Usage

### Minimal Model Config (vLLM)

```yaml
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"

vllm:
  provider_config: "vllm_single_card"
  overrides:
    max_model_len: 2000

tasks:
  - "spanish"
  - "portuguese"
```

### Common Commands

```bash
# Full evaluation
python eval.py --config configs/single_card/qwen34b.yaml

# Limited samples
python eval.py --config configs/single_card/qwen34b.yaml --limit 10

# Test vLLM server only
python eval.py --config configs/single_card/qwen34b.yaml --test

# Custom run ID
python eval.py --config configs/single_card/qwen34b.yaml --run-id my_experiment_001
```

## Configuration (More Technical)

### Project Structure

```
benchy/
├── configs/                 # Global + provider + model configs
│   ├── config.yaml          # Global settings
│   ├── providers/           # Provider configs (vLLM, cloud APIs)
│   ├── single_card/         # Pre-configured model configs
│   └── templates/           # Model config templates
├── src/
│   ├── pipeline.py          # Main Prefect pipeline
│   ├── interfaces/          # Provider interfaces
│   ├── tasks/               # Task code + task.json configs
│   └── leaderboard/         # Results processing
├── outputs/                 # Benchmark outputs
└── eval.py                  # CLI entry point
```

### Task Configs

Task configs live beside their implementation in `src/tasks/<task>/task.json`. These define datasets, prompts, and metric metadata for the task or its subtasks.
Use `dataset` for single-task configs or `tasks` + `task_configs` for grouped tasks.

### Provider Configs

Provider configs live in `configs/providers/` and define defaults for a serving stack (e.g., vLLM). Model configs can override specific fields without duplicating everything.

### Capabilities

Compatibility checks use capability flags defined under provider configs. You can override them per model
via provider overrides, or by adding tags under `metadata` (e.g., `supports_multimodal`, `supports_logprobs`),
which are mapped into `model_capabilities` automatically. Model capabilities can only restrict provider
capabilities; they never enable features that the provider config does not support.

### Global Config

`configs/config.yaml` centralizes paths, logging, and leaderboard scoring settings.

## Results and Publishing

After runs finish, process results for the leaderboard:

```bash
python ./src/leaderboard/process_all.py
```

This generates:

- `leaderboard_table.json` and `leaderboard_table.csv`
- Per-model summaries in `publish/summaries/`
- `tasks_list.json` and `tasks_groups.json` derived from task configs

## Contributing

### Contributing a Task

1. Copy the template: `cp -r src/tasks/_template src/tasks/my_task`
2. Update `src/tasks/my_task/task.json` with datasets, prompts, and metrics.
3. Implement task logic in `src/tasks/my_task/`.
4. Update `src/tasks/my_task/run.py` with a `TaskGroupSpec` and `run_task_group` call.
5. Register the task in `src/pipeline.py` `TASK_REGISTRY`.
6. Add the task name to your model config under `tasks`.

### Contributing a Provider

1. Add a provider config in `configs/providers/`.
2. Implement an interface in `src/interfaces/` to format requests/responses.
3. Wire it into the pipeline via `build_connection_info` and `get_interface_for_provider`.

### Contributing to the Project

1. Open an issue to discuss changes.
2. Create a feature branch.
3. Submit a PR with a concise description and any tests you ran.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- [Prefect](https://www.prefect.io/) for workflow orchestration
- [Surus](https://surus.lat/) for starting this project
- LATAM community for benchmark development
