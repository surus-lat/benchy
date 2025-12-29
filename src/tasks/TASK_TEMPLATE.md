# Task Implementation Guide

This guide shows how to implement new benchmark tasks using the modular engine.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Pipeline                                   │
│  - Manages provider (vLLM server, cloud APIs)                       │
│  - Builds connection_info dict                                       │
│  - Dispatches tasks via TASK_REGISTRY                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────┐
│   TaskGroupRunner        │
│   - Reads TaskGroupSpec  │
│   - Builds interface     │
│   - Runs subtasks        │
└─────────────────────────┘
                │
                ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│   Generic Engine         │    │   Interface             │
│   - BenchmarkRunner      │───▶│   - prepare_request()   │
│   - Checkpointing        │    │   - generate_batch()    │
│   - Batching             │    │                         │
└─────────────────────────┘    └─────────────────────────┘
                │
                ▼
┌─────────────────────────┐
│   Task Class             │
│   - load()               │
│   - get_samples()        │
│   - get_prompt()         │
│   - calculate_metrics()  │
└─────────────────────────┘
```

## Key Principle: Tasks are Interface-Agnostic

Tasks define:
- **Data**: How to load and iterate samples
- **Prompts**: How to format inputs for LLMs
- **Metrics**: How to evaluate predictions
- **Answer type**: Whether outputs are freeform, structured, or multiple-choice

Tasks do NOT know about:
- Which provider is being used
- How requests are formatted
- Network communication details

TaskGroupRunner calls BenchmarkRunner to bridge tasks and interfaces.

## Adding a New Task

### 1. Create Task Config (`src/tasks/my_task/task.json`)

Single-task config (no subtasks):

```json
{
  "name": "my_task",
  "description": "Brief description of what this task evaluates",
  "dataset": {
    "data_file": "data.jsonl"
  },
  "prompts": {
    "system": "You are a helpful assistant for [task description].",
    "user": "Input:\n{text}\n\nPlease respond with [expected format]."
  },
  "defaults": {
    "batch_size": 20,
    "log_samples": false,
    "temperature": 0.0,
    "max_tokens": 2048,
    "timeout": 120,
    "max_retries": 3
  },
  "metrics": {
    "my_metric": {
      "threshold": 0.5
    }
  },
  "output": {
    "subdirectory": "my_task"
  },
  "metrics_manifest": [
    "my_metric"
  ]
}
```

Grouped task config (with subtasks):

```json
{
  "name": "my_task",
  "description": "Brief description of what this task evaluates",
  "tasks": [
    "subtask1",
    "subtask2"
  ],
  "task_configs": {
    "subtask1": {
      "dataset": {
        "data_file": "subtask1_data.jsonl"
      },
      "dataset_path": "org/dataset-name",
      "split": "test"
    }
  },
  "prompts": {
    "system": "You are a helpful assistant for [task description].",
    "user": "Input:\n{text}\n\nPlease respond with [expected format]."
  },
  "defaults": {
    "batch_size": 20,
    "log_samples": false,
    "temperature": 0.0,
    "max_tokens": 2048,
    "timeout": 120,
    "max_retries": 3
  },
  "output": {
    "subdirectory": "my_task"
  },
  "metrics_manifest": [
    "my_metric"
  ]
}
```

Use `dataset.data_file` for local JSONL files, or `dataset_path` / `dataset_name`
for HuggingFace datasets. Only include the fields you need.

`metrics_manifest` lists the aggregate metric keys to surface in `run_summary.json`.

### 2. Create Task Class (`src/tasks/my_task/task.py`)

```python
"""My Task implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, List

logger = logging.getLogger(__name__)


class MyTask:
    """Task for my_task evaluation.
    
    Implements the BaseTask protocol.
    """

    def __init__(self, config: Dict):
        """Initialize task with config."""
        self.config = config
        self.data_file = Path(config["dataset"]["data_file"])
        self.dataset = None

    def load(self) -> None:
        """Load dataset from JSONL file."""
        logger.info(f"Loading dataset from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_file}")
        
        self.dataset = []
        with open(self.data_file, "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples")

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over samples.
        
Each sample should have at minimum:
- id: Unique identifier
- text: Input text
- expected: Expected output for metrics
        """
        if self.dataset is None:
            raise RuntimeError("Call load() first")
        
        data = self.dataset[:limit] if limit else self.dataset
        for sample in data:
            yield sample

    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Build prompts for LLM interfaces.
        
        This is called by interfaces that need prompts.
        HTTP interfaces may use sample["text"] directly instead.
        
        Returns:
            (system_prompt, user_prompt)
        """
        system = self.config["prompts"]["system"]
        user_template = self.config["prompts"]["user"]
        
        user = user_template.format(
            text=sample["text"],
            # Add other placeholders as needed
        )
        
        return system, user

    def get_task_name(self) -> str:
        """Return task identifier."""
        return "my_task"

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for one prediction.
        
        Args:
            prediction: Model output (parsed)
            expected: Expected output from sample
            sample: Full sample dict for context
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            
        Returns:
            Dict of metric_name -> value. Must include 'valid' (bool).
        """
        # Handle errors - delegate to get_error_metrics for consistency
        if error or prediction is None:
            return self.get_error_metrics(
                error=error or "No prediction",
                error_type=error_type,
            )
        
        # Example: simple exact match
        exact_match = prediction == expected
        
        return {
            "valid": True,
            "exact_match": exact_match,
            "score": 1.0 if exact_match else 0.0,
        }
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get error metrics structure for failed predictions.
        
        The engine calls this when calculate_metrics() raises an exception
        or when handling connectivity errors. Override to provide task-specific
        error metric structures.
        
        Args:
            error: Error message
            error_type: Type of error ('connectivity_error' or 'invalid_response')
            
        Returns:
            Dict of error metrics matching your task's metric structure.
            Must include 'valid': False.
        """
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "score": 0.0,
            "exact_match": False,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary.
        
        Args:
            all_metrics: List of per-sample metric dicts
            
        Returns:
            Aggregated summary metrics
        """
        if not all_metrics:
            return {"total_samples": 0, "score": 0.0}
        
        valid = [m for m in all_metrics if m.get("valid")]
        
        return {
            "total_samples": len(all_metrics),
            "valid_samples": len(valid),
            "exact_match_rate": sum(m["exact_match"] for m in valid) / len(valid) if valid else 0,
            "score": sum(m["score"] for m in valid) / len(valid) if valid else 0,
        }

    # Optional capability flags
    @property
    def is_multimodal(self) -> bool:
        """Does this task use images/audio?"""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Does this task use JSON schemas?"""
        return False

    @property
    def answer_type(self) -> str:
        """Expected answer type: 'freeform', 'structured', or 'multiple_choice'."""
        return "freeform"

    @property
    def requires_logprobs(self) -> bool:
        """Whether this task requires logprobs-based scoring."""
        return False
```

### 3. Create Task Runner (`src/tasks/my_task/run.py`)

```python
"""My Task Prefect runner."""

import logging
from typing import Dict, Any, Optional
from prefect import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
from .task import MyTask

logger = logging.getLogger(__name__)


def _prepare_my_task(context: SubtaskContext) -> MyTask:
    dataset_config = context.subtask_config.get("dataset") or context.task_config.get("dataset", {})
    return MyTask({
        "dataset": dataset_config,
        "prompts": context.prompts,
    })


MY_TASK_SPEC = TaskGroupSpec(
    name="my_task",
    display_name="My Task",
    output_subdir="my_task",
    supports_subtasks=False,
    prepare_task=_prepare_my_task,
)


@task
def run_my_task(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run my_task evaluation."""
    return run_task_group(
        spec=MY_TASK_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )
```

Adjust `dataset_config` to match your task's config shape (for example,
`dataset_path` / `dataset_name` instead of `dataset.data_file`).

For grouped tasks, set `supports_subtasks=True` and implement `aggregate_metrics`,
`write_summary`, or `run_subtask` in the `TaskGroupSpec`.

### 4. Register Task in Pipeline (`src/pipeline.py`)

Add an entry to `TASK_REGISTRY`:

```python
TASK_REGISTRY = {
    "my_task": {
        "run": run_my_task,
        "config_name": "my_task",
        "display_name": "My task",
        "set_api_endpoint": True,
        "set_generation_config": True,
        "provider_types": ["openai", "anthropic", "surus", "together"],
    },
}
```

## Sample Data Format

Samples should be stored in JSONL with at minimum:

```json
{"id": "sample_001", "text": "Input text here", "expected": "Expected output"}
{"id": "sample_002", "text": "Another input", "expected": {"key": "structured output"}}
```

For structured extraction tasks, include schema:

```json
{"id": "001", "text": "...", "schema": {"type": "object", ...}, "expected": {...}}
```

For multiple-choice tasks, include choices and the expected index:

```json
{"id": "mcq_001", "text": "Question text", "choices": ["A", "B", "C"], "expected": 1}
```

## Dataset Preprocessing

For HuggingFace datasets, create a download script:

```python
# src/tasks/my_task/download.py
from datasets import load_dataset
import json

def download_and_preprocess(output_file, cache_dir="./cache"):
    dataset = load_dataset("org/dataset-name", cache_dir=cache_dir)
    
    samples = []
    for idx, item in enumerate(dataset["train"]):
        samples.append({
            "id": f"sample_{idx}",
            "text": item["input"],
            "expected": item["output"],
        })
    
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
```

## Testing Your Task

```bash
# Test with few samples
python eval.py --config configs/models/openai_gpt-4o-mini.yaml --limit 5

# Full run
python eval.py --config configs/models/openai_gpt-4o-mini.yaml
```

## Example: Complete Reference

See `src/tasks/structured/` for a complete implementation with:
- Multiple subtasks (paraloq, chat_extract)
- Custom metrics calculator
- Dataset preprocessing
- Checkpoint support
- Multi-provider compatibility

## Error Handling

The engine handles errors from interfaces and passes them to tasks via `error` and `error_type` parameters:

- **`error`**: Error message string (or None if successful)
- **`error_type`**: Either `'connectivity_error'` (network/timeout issues) or `'invalid_response'` (API returned error)

### Best Practices

1. **In `calculate_metrics()`**: Check for errors first and delegate to `get_error_metrics()`:
   ```python
   if error or prediction is None:
       return self.get_error_metrics(error=error or "No prediction", error_type=error_type)
   ```

2. **Implement `get_error_metrics()`**: Return error metrics matching your task's structure:
   ```python
   def get_error_metrics(self, error: str, error_type: Optional[str] = None) -> Dict[str, Any]:
       return {
           "valid": False,
           "error": error,
           "error_type": error_type,
           # ... other task-specific error metrics with default values
       }
   ```

3. **Keep it simple**: Tasks should just pass errors through. The engine handles:
   - Catching exceptions from `calculate_metrics()`
   - Calling `get_error_metrics()` for fallback
   - Logging and reporting

## Answer Types and Logprobs

Tasks must declare what kind of answer they expect:

- `freeform`: raw text output (default)
- `structured`: JSON output with a schema
- `multiple_choice`: select one of N choices

For multiple-choice tasks:
- Populate `sample["choices"]` and set `sample["expected"]` to the correct index.
- Set `answer_type = "multiple_choice"`.
- Set `requires_logprobs = True` to enable logprob scoring and compatibility checks.

### Example: Using a Metrics Calculator

If your task uses a metrics calculator (like structured extraction), simply pass errors through:

```python
def calculate_metrics(self, prediction, expected, sample, error=None, error_type=None):
    return self.metrics_calculator.calculate_all(
        prediction=prediction,
        expected=expected,
        schema=sample.get("schema", {}),
        error=error,
        error_type=error_type,
    )

def get_error_metrics(self, error: str, error_type: Optional[str] = None):
    # Use calculator to get consistent error structure
    return self.metrics_calculator.calculate_all(
        prediction=None,
        expected={},
        schema={},
        error=error,
        error_type=error_type,
    )
```

## Quick Checklist

- [ ] Create `src/tasks/my_task/task.json`
- [ ] Create task class with `load()`, `get_samples()`, `get_prompt()`, `calculate_metrics()`, `aggregate_metrics()`
- [ ] Implement `get_error_metrics()` for error handling
- [ ] Set `answer_type` and `requires_logprobs` if applicable
- [ ] Create `TaskGroupSpec` and `run_task_group` wrapper in `src/tasks/my_task/run.py`
- [ ] Register in `src/pipeline.py` `TASK_REGISTRY`
- [ ] Add dataset download script if using HuggingFace
- [ ] Test with `--limit 5`
