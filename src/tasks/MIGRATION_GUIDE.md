# Migration Guide: lm-evaluation-harness → Benchy Engine

This guide provides step-by-step instructions for migrating evaluation tasks from `lm-evaluation-harness` to the Benchy engine.

## Overview

The Benchy engine uses a modular architecture where:
- **Tasks** define data, prompts, and metrics
- **Interfaces** handle communication with AI systems (vLLM, OpenAI, Anthropic, etc.)
- **BenchmarkRunner** orchestrates batch processing, checkpointing, and result aggregation

## Architecture Comparison

### lm-evaluation-harness
- Tasks defined in YAML files
- Uses `doc_to_text` and `doc_to_target` functions
- Metrics calculated via framework hooks
- Single monolithic execution

### Benchy Engine
- Tasks implement `BaseTask` protocol (Python classes)
- Tasks define their own data loading, prompts, and metrics
- Interfaces handle provider-specific communication
- Modular, extensible design

## Step-by-Step Migration Process

### Step 1: Understand the Original Task

1. **Locate the task in lm-evaluation-harness:**
   - Find the task directory: `external/lm-evaluation-harness/lm_eval/tasks/<task_name>/`
   - Read the YAML configuration file
   - Review any Python utilities (`utils.py`, `build_dataset.py`, etc.)

2. **Identify key components:**
   - **Dataset source**: Where does data come from? (HuggingFace, local files, API)
   - **Data format**: What does each sample look like?
   - **Prompt format**: How are prompts constructed?
   - **Metrics**: What metrics are calculated? (BLEU, accuracy, F1, etc.)
   - **Output format**: What does the expected output look like?

3. **Check for dependencies:**
   - External datasets that need downloading
   - Preprocessing scripts
   - Special metric libraries

### Step 2: Create Task Structure

Create the following directory structure:

```
src/tasks/<task_name>/
├── __init__.py          # Export run_<task_name>
├── base.py             # Base class (if shared across subtasks)
├── metrics.py          # Metrics calculator (if complex)
├── run.py              # Prefect task wrapper
└── datasets/           # If multiple datasets
    ├── __init__.py
    └── <dataset_name>/
        ├── __init__.py
        ├── download.py  # Dataset download/preprocessing
        └── task.py      # Task implementation
```

**Example from translation task:**
```
src/tasks/translation/
├── __init__.py
├── base.py              # TranslationTaskBase
├── metrics.py           # TranslationMetricsCalculator
├── run.py               # run_translation Prefect task
└── datasets/
    ├── opus/
    │   ├── __init__.py
    │   ├── download.py
    │   └── task.py
    └── flores/
        ├── __init__.py
        ├── download.py
        └── task.py
```

### Step 3: Implement Base Task Class

Create a class that implements the `BaseTask` protocol:

```python
from typing import Dict, Iterator, Optional, Any, List
from ...engine.protocols import BaseTask

class YourTaskBase(BaseTask):
    """Base class for your task."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset = []
    
    def load(self) -> None:
        """Load and preprocess dataset."""
        # Download/preprocess data if needed
        # Store in self.dataset
        pass
    
    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Yield samples from dataset."""
        data = self.dataset[:limit] if limit else self.dataset
        for sample in data:
            yield sample
    
    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Generate system and user prompts from sample.
        
        Returns:
            (system_prompt, user_prompt) tuple
        """
        system_prompt = self.config.get("prompts", {}).get("system", "")
        user_template = self.config.get("prompts", {}).get("user", "")
        
        user_prompt = user_template.format(**sample)
        return system_prompt, user_prompt
    
    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction."""
        # Handle errors
        if error or prediction is None:
            return {
                "valid": False,
                "error": error or "No prediction",
                "error_type": error_type,
                # ... task-specific metrics with default values
            }
        
        # Calculate metrics
        # Return dict with "valid": True and metric values
        pass
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics."""
        # Calculate averages, totals, etc.
        pass
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return error metrics structure."""
        return self.calculate_metrics(
            prediction=None,
            expected="",
            sample={},
            error=error,
            error_type=error_type,
        )
    
    def get_task_name(self) -> str:
        """Return task name."""
        return "your_task_name"
    
    @property
    def is_multimodal(self) -> bool:
        """Return True if task uses images/video."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Return True if task requires JSON schema for structured output."""
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

### Step 4: Implement Dataset Download/Preprocessing

If your task needs to download or preprocess data:

```python
# src/tasks/<task_name>/datasets/<dataset>/download.py

import json
import logging
from pathlib import Path
from typing import Dict, List

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)

def download_and_preprocess_<dataset>(
    output_dir: Path,
    dataset_name: str = "huggingface/dataset",
    cache_dir: str = "./cache",
    **kwargs
) -> Dict[str, int]:
    """Download and preprocess dataset.
    
    Args:
        output_dir: Where to save preprocessed data
        dataset_name: HuggingFace dataset name
        cache_dir: Cache directory for datasets library
        
    Returns:
        Dictionary with counts/stats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required")
    
    # Download from HuggingFace
    logger.info(f"Downloading {dataset_name}...")
    dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Preprocess and save as JSONL
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            output_file = output_dir / f"{split}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset[split]:
                    # Transform to your format
                    processed = {
                        "id": item.get("id", str(hash(str(item)))),
                        # ... your fields
                    }
                    f.write(json.dumps(processed, ensure_ascii=False) + '\n')
    
    return {"total": len(dataset)}
```

**Key points:**
- Save preprocessed data as JSONL in `.data/<task_name>/<dataset>/`
- Check if data already exists to avoid re-downloading
- Use consistent data format across splits

### Step 5: Implement Metrics Calculator

If metrics are complex, create a separate calculator:

```python
# src/tasks/<task_name>/metrics.py

class YourMetricsCalculator:
    """Calculator for task-specific metrics."""
    
    def __init__(self, config: Dict):
        self.config = config
        # Initialize any models/libraries needed
    
    def calculate(
        self,
        prediction: Optional[Any],
        expected: Any,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction."""
        if error or prediction is None:
            return {
                "valid": False,
                "error": error,
                # ... default metric values
            }
        
        # Calculate metrics
        metrics = {
            "valid": True,
            "metric1": self._calculate_metric1(prediction, expected),
            "metric2": self._calculate_metric2(prediction, expected),
        }
        return metrics
    
    def aggregate(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics."""
        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        
        return {
            "total_samples": len(all_metrics),
            "valid_samples": len(valid_metrics),
            "metric1": sum(m.get("metric1", 0) for m in valid_metrics) / len(valid_metrics),
            # ...
        }
```

**Important for expensive metrics (like COMET):**
- Defer calculation to `aggregate()` method
- Batch process multiple samples at once
- Load models once and reuse

### Step 6: Create Prefect Task Wrapper

```python
# src/tasks/<task_name>/run.py

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from prefect import task

from ...engine import (
    BenchmarkRunner,
    save_results,
    build_connection_info,
    get_interface_for_provider,
    mark_task_complete,
)
from .datasets.<dataset>.task import YourTask

logger = logging.getLogger(__name__)

@task
def run_<task_name>(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run <task_name> evaluation.
    
    Args:
        model_name: Model identifier
        output_path: Base output directory
        server_info: Server connection info
        api_test_result: API test results
        task_config: Task configuration from YAML
        limit: Optional sample limit
        cuda_devices: CUDA devices for vLLM
        provider_config: Provider-specific config
        
    Returns:
        Results dictionary
    """
    logger.info(f"Starting <task_name> evaluation for model: {model_name}")
    
    # Get provider type
    provider_type = "vllm"
    if provider_config:
        provider_type = provider_config.get('provider_type', 'vllm')
    
    # Build connection info
    connection_info = build_connection_info(
        provider_type=provider_type,
        provider_config=provider_config or {},
        server_info=server_info,
        model_config=task_config.get('defaults', {}),
    )
    
    # Setup output directory
    output_subdir = task_config.get('output', {}).get('subdirectory', '<task_name>')
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    defaults = task_config.get('defaults', {})
    prompts = task_config.get('prompts', {})
    
    # Create task instance
    task_instance = YourTask({
        **task_config.get('task_configs', {}).get('<subtask>', {}),
        'prompts': prompts,
        'defaults': defaults,
    })
    
    # Load data
    task_instance.load()
    
    # Get interface
    interface = get_interface_for_provider(
        provider_type=provider_type,
        connection_info=connection_info,
        model_name=model_name,
    )
    
    # Run benchmark
    runner_config = {
        "model_name": model_name,
        "batch_size": defaults.get('batch_size', 20),
        "output_dir": str(task_output_path),
        "log_samples": defaults.get('log_samples', False),
    }
    
    runner = BenchmarkRunner(task_instance, interface, runner_config)
    results = asyncio.run(runner.run(limit=limit, no_resume=False))
    
    # Save results
    save_results(
        results=results,
        output_dir=task_output_path,
        model_name=model_name,
        task_name=task_instance.get_task_name(),
        log_samples=defaults.get('log_samples', False),
        mark_complete=True,
    )
    
    logger.info(f"<task_name> evaluation completed")
    
    return {
        "model_name": model_name,
        "task": "<task_name>",
        "output_path": str(task_output_path),
        "metrics": results.get('aggregate_metrics', {}),
    }
```

### Step 7: Update Configuration File

Create/update `configs/tasks/<task_name>.yaml`:

```yaml
name: "<task_name>"
description: "Description of the task"

# If multiple subtasks/datasets
tasks:
  - "<subtask1>"
  - "<subtask2>"

task_configs:
  <subtask1>:
    dataset_name: "huggingface/dataset1"
    # ... subtask-specific config
  
  <subtask2>:
    dataset_name: "huggingface/dataset2"
    # ... subtask-specific config

prompts:
  system: |
    Your system prompt here
  user: |
    Your user prompt template with {placeholders}

defaults:
  batch_size: 20
  log_samples: false
  temperature: 0.0
  max_tokens: 512
  timeout: 120
  max_retries: 3

output:
  subdirectory: "<task_name>"
```

### Step 8: Register in Pipeline

Update `src/pipeline.py`:

```python
from .tasks.<task_name>.run import run_<task_name>

# In the main pipeline function:
if "<task_name>" in pending_tasks:
    logger.info("Running <task_name> evaluation...")
    task_config = config_manager.get_task_config("<task_name>", task_defaults_overrides)
    task_config['use_chat_completions'] = use_chat_completions
    task_config['generation_config'] = generation_config
    
    if log_setup:
        log_setup.log_task_config("<task_name>", task_config)
    
    # Get provider config
    cloud_provider_config = None
    if provider_type in ['openai', 'anthropic', 'surus', 'together'] and provider_config:
        cloud_provider_config = {
            **provider_config,
            'provider_type': provider_type
        }
    
    results = run_<task_name>(
        model_name=model_name,
        output_path=model_output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=task_config,
        limit=limit,
        cuda_devices=gpu_manager.get_task_cuda_devices() if provider_type == 'vllm' else None,
        provider_config=cloud_provider_config
    )
    task_results["<task_name>"] = results
```

### Step 9: Update __init__.py

```python
# src/tasks/<task_name>/__init__.py

from .run import run_<task_name>

__all__ = ["run_<task_name>"]
```

## Common Patterns

### Pattern 1: Multiple Datasets (like translation)

If your task uses multiple datasets:

1. Create `datasets/` subdirectory
2. Each dataset gets its own folder with `download.py` and `task.py`
3. In `run.py`, iterate over subtasks and run each
4. Aggregate results across subtasks

### Pattern 2: Structured Output

If your task requires JSON schema:

1. Set `requires_schema = True` in task class
2. Set `answer_type = "structured"`
3. Include schema in sample dict: `sample["schema"] = {...}`
4. The engine will enforce compatibility and pass the schema to the interface

### Pattern 3: Multiple Choice (Logprobs)

If your task is multiple choice:

1. Include `choices` in each sample and set `expected` to the correct index
2. Set `answer_type = "multiple_choice"`
3. Set `requires_logprobs = True` to enable logprob scoring
4. The engine will enforce compatibility and error if the interface does not support logprobs

### Pattern 4: Expensive Metrics

If metrics are expensive to calculate (like COMET):

1. Defer calculation to `aggregate_metrics()`
2. Store predictions/references in per-sample metrics
3. Batch process in `aggregate_metrics()`
4. Load models once and reuse

### Pattern 5: Multimodal Tasks

If your task uses images/video:

1. Set `is_multimodal = True` in task class
2. Include image URLs/paths in samples
3. The engine will enforce compatibility and error if the interface does not support multimodal inputs

## Testing Checklist

- [ ] Task loads data correctly
- [ ] Prompts are generated correctly
- [ ] Metrics calculate correctly
- [ ] Error handling works (connection errors, invalid responses)
- [ ] Results are saved correctly
- [ ] Checkpointing/resume works
- [ ] Task appears in pipeline
- [ ] Configuration is loaded correctly

## Common Pitfalls

1. **Forgetting to call `load()`**: Always call `task_instance.load()` before running
2. **Not handling errors**: Always return proper error metrics structure
3. **Memory issues**: For large datasets, use iterators, not lists
4. **Path issues**: Use `Path(__file__).parent` for relative paths
5. **Schema issues**: Only set `requires_schema = True` if actually needed
6. **Metric batching**: Don't calculate expensive metrics per-sample

## Reference Implementation

See `src/tasks/translation/` for a complete example with:
- Multiple datasets (OPUS, FLORES)
- Complex metrics (BLEU, chrF, COMET)
- Batch processing for expensive metrics
- Proper error handling
- Configuration management

## Getting Help

- Check `src/tasks/TASK_TEMPLATE.md` for basic structure
- Review `src/tasks/structured/` for structured output example
- Review `src/tasks/translation/` for multi-dataset example
- Check `src/engine/protocols.py` for BaseTask interface

