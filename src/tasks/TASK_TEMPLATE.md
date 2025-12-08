# Task Implementation Guide

This guide provides templates and best practices for implementing new benchmark tasks in benchy.

## Overview

Tasks are modular evaluation units that:
1. Load and preprocess datasets
2. Generate prompts for AI systems
3. Perform inference via interfaces
4. Calculate metrics
5. Report results

## Task Structure

### Simple Tasks (Single File)

For straightforward benchmarks with one dataset and simple metrics:

```
src/tasks/
└── my_task.py  # Single file with @task decorator
```

### Complex Tasks (Directory)

For tasks with multiple datasets, subtasks, or complex metrics:

```
src/tasks/
└── my_task/
    ├── __init__.py           # Exports main task function
    ├── my_task.py            # Main task entrypoint (@task decorator)
    ├── tasks/                # Subtask implementations
    │   ├── __init__.py
    │   ├── subtask1.py
    │   └── subtask2.py
    ├── metrics/              # Task-specific metrics
    │   ├── __init__.py
    │   └── custom_metrics.py
    ├── utils/                # Task-specific preprocessing
    │   ├── __init__.py
    │   └── preprocessing.py
    └── README.md             # Task documentation
```

## Required Components

### 1. Task Configuration (`configs/tasks/my_task.yaml`)

```yaml
# Task identification
name: "my_task"
description: "Brief description of the task"

# Subtasks to run (for complex tasks)
tasks:
  - "subtask1"
  - "subtask2"

# Per-subtask configurations
task_configs:
  subtask1:
    dataset_file: "subtask1_data.jsonl"
    dataset_name: "org/dataset-name"
  subtask2:
    dataset_file: "subtask2_data.jsonl"
    dataset_name: "org/other-dataset"

# Prompt templates
prompts:
  system: |
    System instruction here.
  user: |
    User prompt template with {placeholders}.

# Default evaluation parameters
defaults:
  batch_size: 20
  log_samples: true
  temperature: 0.0
  max_tokens: 2048
  timeout: 120
  max_retries: 3

# Metrics configuration
metrics:
  primary_metric:
    enabled: true
    weights:
      component1: 0.5
      component2: 0.5

# Output configuration
output:
  subdirectory: "my_task"
```

### 2. Task Entrypoint Function

```python
from typing import Dict, Any, Optional
from prefect import task
from ...interfaces.llm_interface import LLMInterface
from ...common.dataset_utils import load_jsonl_dataset
import logging

logger = logging.getLogger(__name__)

@task
def run_my_task(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run my_task evaluation.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info (None for cloud providers)
        api_test_result: API test result
        task_config: Task configuration dictionary
        limit: Limit number of examples
        cuda_devices: CUDA devices to use
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    logger.info(f"Starting my_task evaluation for model: {model_name}")
    
    # Determine provider and base URL
    provider_type = "vllm"
    base_url = None
    
    if provider_config:
        provider_type = provider_config.get('provider_type', 'vllm')
        if provider_type in ['openai', 'anthropic']:
            base_url = provider_config.get('base_url')
    
    if base_url is None and server_info:
        base_url = server_info['url'] + '/v1'
    
    # Create task-specific output path
    output_subdir = task_config.get('output', {}).get('subdirectory', 'my_task')
    task_output_path = f"{output_path}/{output_subdir}"
    
    # Load dataset and run evaluation
    # ... implementation details ...
    
    return {
        "model_name": model_name,
        "task": "my_task",
        "output_path": task_output_path,
        "metrics": metrics,
    }
```

### 3. Subtask Classes (for complex tasks)

```python
from pathlib import Path
from typing import Dict, Iterator, Optional
import json

class MySubtask:
    """Subtask for specific dataset."""

    def __init__(self, config: Dict):
        """Initialize subtask.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_file = Path(config["dataset"]["data_file"])
        self.dataset = None

    def load(self) -> None:
        """Load the dataset from local JSONL file."""
        from ...common.dataset_utils import load_jsonl_dataset
        self.dataset = load_jsonl_dataset(self.data_file)

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples."""
        from ...common.dataset_utils import iterate_samples
        yield from iterate_samples(self.dataset, limit=limit)

    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Build prompt messages for a sample.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config["prompts"]["system"]
        user_template = self.config["prompts"]["user"]
        
        user_prompt = user_template.format(
            field1=sample["field1"],
            field2=sample["field2"]
        )
        
        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Get the subtask identifier."""
        return "my_subtask"
```

### 4. Dataset Preprocessing

```python
from pathlib import Path
from typing import Dict, Any
from ...common.dataset_utils import (
    download_huggingface_dataset,
    save_to_jsonl
)

def download_and_preprocess_my_dataset(
    dataset_name: str,
    output_file: Path,
    cache_dir: str = "./cache",
    split: str = "train",
) -> Dict[str, int]:
    """Download and preprocess dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_file: Path to save processed JSONL
        cache_dir: Cache directory
        split: Dataset split to use
        
    Returns:
        Dictionary with processing statistics
    """
    # Download dataset using common utility
    dataset = download_huggingface_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir
    )
    
    # Task-specific preprocessing
    processed_samples = []
    skipped = 0
    
    for idx, sample in enumerate(dataset):
        processed = preprocess_sample(sample, idx)
        if processed:
            processed_samples.append(processed)
        else:
            skipped += 1
    
    # Save using common utility
    save_to_jsonl(processed_samples, output_file)
    
    return {
        "processed": len(processed_samples),
        "skipped": skipped,
    }

def preprocess_sample(sample: Dict, idx: int) -> Dict[str, Any]:
    """Task-specific sample preprocessing logic."""
    # Validation
    if not sample.get("required_field"):
        return None
    
    # Transform
    processed = {
        "id": f"sample_{idx}",
        "text": sample["text"],
        "expected": sample["label"],
        "metadata": sample.get("metadata", {}),
    }
    
    return processed
```

## Using Common Infrastructure

### Interface Pattern (LLM and HTTP)

**Key Insight**: Interfaces handle request formatting, not tasks. This keeps tasks provider-agnostic.

#### Request Preparation

All interfaces implement `prepare_request(sample, task)`:

```python
# Tasks provide raw data and a get_prompt() method
class MyTask:
    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Build prompts for LLM interfaces."""
        system_prompt = self.config["prompts"]["system"]
        user_prompt = self.config["prompts"]["user"].format(**sample)
        return system_prompt, user_prompt

# Interface decides what it needs
# LLMInterface calls task.get_prompt() and formats with prompts
# HTTPInterface uses raw sample data (text, schema) directly

# In your runner code:
requests = [
    interface.prepare_request(sample, task)
    for sample in batch
]
```

#### LLM Interface Usage

```python
from ...interfaces.llm_interface import LLMInterface

# Configure interface
config = {
    "model": {
        "base_url": base_url,
        "api_key": api_key,
        "temperature": defaults.get('temperature', 0.0),
        "max_tokens": defaults.get('max_tokens', 2048),
        "timeout": defaults.get('timeout', 120),
        "max_retries": defaults.get('max_retries', 3),
    },
    "performance": {
        "batch_size": defaults.get('batch_size', 20),
    },
}

# Initialize
llm = LLMInterface(config, model_name, provider_type=provider_type)

# Test connection
if not await llm.test_connection():
    raise ConnectionError("Cannot connect to LLM provider")

# Prepare requests (interface handles formatting)
requests = [
    llm.prepare_request(sample, task)
    for sample in batch
]

# Generate
results = await llm.generate_batch(requests)
```

#### HTTP Interface Usage

```python
from ...interfaces.surus_interface import SurusInterface

# Configure interface
config = {
    "surus": {
        "endpoint": "https://api.surus.dev/functions/v1/extract",
        "api_key_env": "SURUS_API_KEY",
        "timeout": 30,
        "max_retries": 3,
    },
}

# Initialize
interface = SurusInterface(config, "surus-extract", provider_type="surus")

# Test connection
if not await interface.test_connection():
    raise ConnectionError("Cannot connect to SURUS")

# Prepare requests (interface uses raw data, not prompts)
requests = [
    interface.prepare_request(sample, task)  # Uses sample["text"] directly
    for sample in batch
]

# Generate
results = await interface.generate_batch(requests)
```

#### Task Requirements

For maximum compatibility, tasks should provide:

1. **Raw data** in samples: `text`, `schema`, `expected`, `id`
2. **Prompt method**: `get_prompt(sample) -> tuple[str, str]` for LLM interfaces
3. **Task identifier**: `get_task_name() -> str`

Example:
```python
class MyTask:
    def load(self) -> None:
        """Load dataset with raw data."""
        self.dataset = [
            {
                "id": "sample_0",
                "text": "Raw input text",
                "schema": {...},
                "expected": {...},
            }
        ]
    
    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """For LLM interfaces - build prompts."""
        return system_prompt, user_prompt
    
    # Sample dict already has "text" for HTTP interfaces
```

This design ensures tasks work with **any interface type** without modification.

### Checkpointing

```python
from ...common.checkpoint_utils import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint
)

# Setup
checkpoint_path = get_checkpoint_path(output_dir, model_name, "my_task")
config_hash = get_config_hash({
    "model": model_name,
    "temperature": temperature,
})

# Load existing
completed_ids = load_checkpoint(checkpoint_path, config_hash)

# Save periodically
if len(completed_ids) % 50 == 0:
    save_checkpoint(checkpoint_path, list(completed_ids), config_hash)
```

## Integration with Pipeline

Tasks are called from `src/pipeline.py`:

```python
if "my_task" in pending_tasks:
    logger.info("Running my_task evaluation...")
    my_task_config = config_manager.get_task_config("my_task", task_defaults_overrides)
    
    # Add system configuration
    my_task_config['use_chat_completions'] = use_chat_completions
    my_task_config['generation_config'] = generation_config
    
    # Log configuration
    if log_setup:
        log_setup.log_task_config("my_task", my_task_config)
    
    # Run task
    my_task_results = run_my_task(
        model_name=model_name,
        output_path=model_output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=my_task_config,
        limit=limit,
        cuda_devices=gpu_manager.get_task_cuda_devices(),
        provider_config=cloud_provider_config
    )
    task_results["my_task"] = my_task_results
```

## Best Practices

### Code Style
1. Add type hints to all functions
2. Keep functions focused and simple
3. Avoid excessive try-except blocks
4. Use clear, descriptive variable names
5. Document with concise docstrings

### Task Design
1. Keep task logic separate from interface logic
2. Make prompts configurable via YAML
3. Support both cloud and local providers
4. Include example outputs in README
5. Test with `--limit 10` before full runs

### Performance
1. Use async/await for I/O operations
2. Process in batches (default: 20)
3. Implement checkpointing for long tasks
4. Log progress regularly
5. Handle errors gracefully

### Configuration
1. Use meaningful default values
2. Document all config parameters
3. Support task-specific overrides
4. Keep provider configuration separate
5. Validate inputs early

## Testing Your Task

```bash
# Test with limited samples
python eval.py --config configs/models/my-model.yaml --limit 10

# Test specific task only
# (Edit config to include only your task)
python eval.py --config configs/models/my-model.yaml

# Test with different providers
export OPENAI_API_KEY="sk-..."
python eval.py --config configs/models/gpt-4.yaml --limit 5
```

## Complete Example

See `src/tasks/structured/` for a complete implementation reference with:
- Multiple subtasks (paraloq, chat_extract)
- Custom metrics
- Dataset preprocessing
- Checkpoint support
- Multi-provider compatibility

