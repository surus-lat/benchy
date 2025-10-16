# Modular Task System

The Benchy system has been designed to be modular and easily extensible for adding new evaluation tasks. This guide covers both adding new tasks to the evaluation pipeline and configuring them for leaderboard reporting.

## How to Add a New Task

### 1. Create Task Configuration

Create a new task configuration file at `configs/tasks/your_new_task.yaml`:

```yaml
# configs/tasks/your_new_task.yaml
name: "your_new_task"
description: "Description of your new task"

# Task-specific configuration
task_config:
  # Add your task-specific settings here
  dataset: "your_dataset"
  metrics: ["accuracy", "f1"]
  
# Output configuration
output:
  subdirectory: "your_new_task"  # Directory name for results
  format: "json"  # Output format

# Default evaluation settings
defaults:
  batch_size: 4
  limit: null
  log_samples: false
```

### 2. Create Task Implementation

Create a new task function in `src/tasks/your_new_task.py`:

```python
# src/tasks/your_new_task.py
from prefect import task
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@task
def run_your_new_task_evaluation(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: int = None,
    cuda_devices: str = None
) -> Dict[str, Any]:
    """
    Run your new task evaluation.
    
    Args:
        model_name: Name of the model being evaluated
        output_path: Output directory for results
        server_info: vLLM server information
        api_test_result: API test results
        task_config: Task configuration
        limit: Optional limit on number of examples
        cuda_devices: CUDA devices to use
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Running {task_config['name']} evaluation for {model_name}")
    
    # Your task implementation here
    # This could involve:
    # - Running lm-evaluation-harness
    # - Custom evaluation logic
    # - Processing results
    
    return {
        "task_name": task_config['name'],
        "model_name": model_name,
        "results": {},  # Your results here
        "status": "completed"
    }
```

### 3. Add Task to Pipeline

Update `src/pipeline.py` to include your new task:

```python
# Add import at the top
from .tasks.your_new_task import run_your_new_task_evaluation

# Add task execution in the benchmark_pipeline function
if "your_new_task" in tasks:
    logger.info("Running your new task evaluation...")
    your_new_task_config = config_manager.get_task_config("your_new_task", task_defaults_overrides)
    your_new_task_config['use_chat_completions'] = use_chat_completions
    your_new_task_config['generation_config'] = generation_config
    
    if log_setup:
        log_setup.log_task_config("your_new_task", your_new_task_config)
    
    your_new_task_results = run_your_new_task_evaluation(
        model_name=model_name,
        output_path=model_output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=your_new_task_config,
        limit=limit,
        cuda_devices=cuda_devices
    )
    task_results["your_new_task"] = your_new_task_results
```

### 4. Update Results Gathering

Update the `gather_results` function in `src/tasks/lm_harness.py` to include your new task results.

## Results Reporting for the Leaderboard

The leaderboard processing system has been refactored to be modular and easily configurable for adding new tasks to the leaderboard.

### 1. Add Task Configuration

Edit `configs/config.yaml` and add your new task to the `leaderboard.tasks` section:

```yaml
leaderboard:
  tasks:
    your_new_task:
      processor: "standard_results_processor"  # or "portuguese_results_processor" or "translation_results_processor"
      category_score_key: "your_category_key"
      output_prefix: "your_task"
      # Optional: subcategories for nested scoring
      subcategories:
        - name: "subcategory_name"
          prefix: "subcategory"
          filter_prefix: "subcategory_"
```

### 2. Add to Overall Score Categories

Add your task's category score key to the `overall_score_categories` list:

```yaml
leaderboard:
  overall_score_categories:
    - "latam_es"
    - "latam_pr" 
    - "translation"
    - "your_category_key"  # Add this
```

### 3. Add to Normalize Scores (if needed)

If your task needs score normalization (e.g., dividing by 100), add it to `normalize_scores`:

```yaml
leaderboard:
  normalize_scores:
    - "translation"
    - "your_new_task"  # Add this if needed
```

### 4. Available Processors

#### `standard_results_processor`
- For tasks that follow the standard LM-Evaluation-Harness results format
- Automatically extracts scores from `results_*.json` files
- Works with any task that has the standard structure

#### `portuguese_results_processor`
- Specialized for Portuguese tasks
- Looks for `results.json` file (different from standard pattern)
- Uses the same score extraction logic as standard processor

#### `translation_results_processor`
- Specialized for translation tasks
- Supports metric selection (BLEU, CHRF, etc.)
- Handles score normalization
- Can exclude specific tasks from individual scores

### 5. Task Configuration Options

- `processor`: Which processor function to use
- `category_score_key`: The key in category_scores to use for the main score
- `output_prefix`: Prefix for output columns (e.g., "spanish", "translation")
- `exclude_tasks`: List of task names to exclude from individual scores
- `metrics`: Available metrics (for translation tasks)
- `primary_metric`: Which metric to use for scoring (for translation tasks)
- `subcategories`: Nested categories with their own scoring

### 6. Example: Adding a New Task to Leaderboard

```yaml
# In config.yaml
leaderboard:
  tasks:
    reasoning:
      processor: "standard_results_processor"
      category_score_key: "reasoning_score"
      output_prefix: "reasoning"
      subcategories:
        - name: "math"
          prefix: "math"
          filter_prefix: "math_"
        - name: "logic"
          prefix: "logic" 
          filter_prefix: "logic_"
  
  overall_score_categories:
    - "latam_es"
    - "latam_pr"
    - "translation"
    - "reasoning_score"  # Add this
```

This will automatically:
1. Process reasoning results using the standard processor
2. Add `reasoning_score` to the leaderboard table
3. Add individual reasoning task scores with `reasoning_` prefix
4. Add math and logic subcategory scores
5. Include reasoning in the overall LATAM score calculation

### 7. Creating Custom Processors

If you need a custom processor for a task with a different format:

1. Add a new processor function in `parse_model_results.py`
2. Register it in the `get_task_processor()` function
3. Use it in your task configuration

The system is designed to be simple and avoid complex inheritance patterns while remaining flexible for different task types.
