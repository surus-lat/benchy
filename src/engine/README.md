# Benchmark Engine

The engine module provides the core infrastructure for running benchmarks.

## Overview

The engine is designed with a clear separation of concerns:

1. **Tasks** define what to evaluate (data, prompts, metrics)
2. **Interfaces** handle how to communicate with AI systems
3. **Runner** orchestrates execution (batching, checkpoints, logging)

Tasks are completely interface-agnostic - they just provide data and metrics.
Interfaces adapt task data to their specific API format.

## Components

### `protocols.py`
Defines the `BaseTask` and `BaseInterface` protocols that all tasks and interfaces must implement.

### `benchmark_runner.py`
Generic benchmark runner that works with any task + interface combination:
- Batch processing with configurable batch size
- Checkpoint support for resumable runs
- Progress logging and result aggregation

### `connection.py`
Utilities for building connection info from provider configurations:
- `build_connection_info()`: Creates standardized connection dict from provider config
- `get_interface_for_provider()`: Returns appropriate interface instance

## Usage

```python
from src.engine import BenchmarkRunner, build_connection_info, get_interface_for_provider

# Build connection info from provider config
connection_info = build_connection_info(
    provider_type="openai",
    provider_config={"base_url": "https://api.openai.com/v1"},
)

# Get interface for the provider
interface = get_interface_for_provider(
    provider_type="openai",
    connection_info=connection_info,
    model_name="gpt-4o-mini",
)

# Create task instance
task = MyTask(config)

# Run benchmark
runner = BenchmarkRunner(task, interface, {
    "model_name": "gpt-4o-mini",
    "batch_size": 20,
    "output_dir": "./results",
})
results = await runner.run(limit=100)
```

## Task Protocol

Tasks must implement:

```python
class BaseTask(Protocol):
    def load(self) -> None: ...
    def get_samples(self, limit: int = None) -> Iterator[Dict]: ...
    def get_prompt(self, sample: Dict) -> Tuple[str, str]: ...
    def get_task_name(self) -> str: ...
    def calculate_metrics(self, prediction, expected, sample) -> Dict: ...
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict: ...
```

## Interface Protocol

Interfaces must implement:

```python
class BaseInterface(Protocol):
    def prepare_request(self, sample: Dict, task: BaseTask) -> Dict: ...
    async def generate_batch(self, requests: List[Dict]) -> List[Dict]: ...
    async def test_connection(self, max_retries: int, timeout: int) -> bool: ...
```

## Connection Info

A standardized dict format used by interfaces:

```python
connection_info = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "...",  # or None to read from env
    "api_key_env": "OPENAI_API_KEY",
    "timeout": 120,
    "max_retries": 3,
    "temperature": 0.0,
    "max_tokens": 2048,
    "use_guided_json": False,  # True for vLLM
}
```

## Design Principles

1. **Tasks don't know about interfaces**: Tasks provide data and metrics, interfaces handle communication
2. **Interfaces adapt to tasks**: Interfaces call `task.get_prompt()` if needed, or use raw sample data
3. **Runner is the bridge**: Orchestrates task + interface without knowing specifics of either
4. **Connection info abstracts providers**: Tasks/runner don't care if it's vLLM, OpenAI, or Anthropic




