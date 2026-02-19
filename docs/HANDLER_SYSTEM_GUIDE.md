# Format Handler Task System Guide

## Overview

The format handler system provides a streamlined way to create new benchmark tasks with minimal boilerplate. Instead of writing 300+ lines of configuration and code, most tasks can be defined in **5-20 lines** using pre-built format handlers.

## Quick Start

### Creating a Multiple Choice Task

```python
# src/tasks/my_task/my_subtask.py
from ..common import MultipleChoiceHandler

class MySubtask(MultipleChoiceHandler):
    """Your task description here."""
    
    # Dataset configuration
    dataset = "org/my-dataset"
    split = "test"
    text_field = "question"
    label_field = "answer"
    labels = {0: "No", 1: "Yes"}
    
    # Prompts
    system_prompt = "You are a helpful classifier."
```

That's it! The handler takes care of:
- Loading and caching the dataset from HuggingFace
- Preprocessing samples
- Building prompts
- Calculating metrics (accuracy, etc.)
- Aggregating results

### Adding Metadata

Create `metadata.yaml` in your task group folder:

```yaml
# src/tasks/my_task/metadata.yaml
name: my_task
display_name: My Task
description: Brief description of what this evaluates

capability_requirements:
  requires_logprobs: preferred
  
subtasks:
  my_subtask:
    description: Detailed description
    citation: "@article{...}"
    url: https://huggingface.co/datasets/org/my-dataset
```

### Running Your Task

```bash
# Run all subtasks in the group
benchy eval --model-name gpt-4o-mini --tasks my_task

# Run specific subtask
benchy eval --model-name gpt-4o-mini --tasks my_task.my_subtask
```

## Available Format Handlers

### 1. MultipleChoiceHandler

For classification tasks with discrete label choices.

**Use cases**: Binary classification, multi-class classification, multiple-choice QA

**Example**:
```python
from ..common import MultipleChoiceHandler

class SentimentAnalysis(MultipleChoiceHandler):
    dataset = "imdb"
    labels = {0: "Negative", 1: "Positive"}
    system_prompt = "You are a sentiment analyzer."
```

**Key attributes**:
- `labels`: Dict mapping label values to display text
- `text_field`: Input text field name (default: "text")
- `label_field`: Label field name (default: "label")
- `prefers_logprobs`: Whether to use logprobs scoring (default: True)

### 2. StructuredHandler

For tasks requiring JSON output following a schema.

**Use cases**: Information extraction, structured generation, entity recognition

**Example**:
```python
from ..common import StructuredHandler

class EntityExtraction(StructuredHandler):
    dataset = "org/extraction-dataset"
    system_prompt = "Extract entities from the text."
    schema_field = "schema"
    
    metrics_config = {
        "partial_matching": {
            "string": {"exact_threshold": 0.95}
        }
    }
```

**Key attributes**:
- `requires_schema`: Whether schema is required (default: True)
- `schema_field`: Field name for schema in samples
- `metrics_config`: Configuration for partial matching and metrics

### 3. FreeformHandler

For open-ended text generation.

**Use cases**: Translation, summarization, open-ended QA, creative writing

**Example**:
```python
from ..common import FreeformHandler
from ..metrics import ExactMatch, F1Score, BLEUScore

class Translation(FreeformHandler):
    dataset = "opus-100"
    system_prompt = "You are a professional translator."
    metrics = [BLEUScore(), F1Score()]
    normalize_prediction = True
    case_sensitive = False
```

**Key attributes**:
- `metrics`: List of Metric objects to compute
- `normalize_prediction`: Whether to normalize text (default: True)
- `case_sensitive`: Case-sensitive comparison (default: False)

### 4. MultimodalStructuredHandler

For image/file input with structured JSON output.

**Use cases**: Image extraction, document parsing, OCR with structure

**Example**:
```python
from ..common import MultimodalStructuredHandler

class InvoiceExtraction(MultimodalStructuredHandler):
    source_dir = "./invoice_data"
    system_prompt = "Extract invoice data from the image."
    
    metrics_config = {
        "partial_matching": {
            "number": {"relative_tolerance": 0.001}
        }
    }
```

**Key attributes**:
- `source_dir`: Directory containing images and expected outputs
- `input_type`: Type of input (default: "image")
- `requires_multimodal`: Multimodal required (default: True)
- `requires_files`: File inputs required (default: True)

## Customizing Handler Behavior

### Custom Prompts

Override the `get_prompt` method for dynamic prompts:

```python
class MyTask(MultipleChoiceHandler):
    dataset = "org/dataset"
    labels = {0: "No", 1: "Yes"}
    
    def get_prompt(self, sample):
        """Build custom prompt logic."""
        from ...common import format_choices
        
        choices_text = format_choices(
            sample["choices"], 
            sample["choice_labels"]
        )
        
        user_prompt = f"Question: {sample['text']}\n\nOptions:\n{choices_text}\n\nAnswer:"
        
        return self.system_prompt, user_prompt
```

### Custom Metrics

Specify custom metrics via class attribute:

```python
from ..metrics import CustomMetric, F1Score

class MyTask(FreeformHandler):
    dataset = "org/dataset"
    metrics = [
        CustomMetric(threshold=0.8),
        F1Score(),
    ]
```

Or with configuration:

```python
class MyTask(StructuredHandler):
    dataset = "org/dataset"
    metrics_config = {
        "extraction_quality_score": {
            "weights": {
                "schema_validity": 0.3,
                "field_f1_partial": 0.7
            }
        }
    }
```

### Custom Preprocessing

Override `preprocess_sample` for custom data transformations:

```python
class MyTask(MultipleChoiceHandler):
    dataset = "org/dataset"
    labels = {0: "No", 1: "Yes"}
    
    def preprocess_sample(self, raw_sample, idx):
        """Custom preprocessing logic."""
        # Extract and transform fields
        text = self._clean_text(raw_sample.get("input"))
        label = self._normalize_label(raw_sample.get("gold_label"))
        
        if text is None or label is None:
            return None  # Skip invalid samples
        
        return {
            "id": f"sample_{idx}",
            "text": text,
            "expected": self.label_to_index[label],
            "choices": list(self.choice_texts),
            "choice_labels": list(self.choice_labels),
        }
```

## File Structure

### Convention-Based Discovery

The system discovers tasks by scanning folder structure:

```
src/tasks/
  my_task/                      # Task group
    metadata.yaml               # Group metadata (optional)
    subtask_one.py             # Subtask 1
    subtask_two.py             # Subtask 2
    .data/                     # Cached datasets
```

### Naming Conventions

- **Folder name** = task group name (`my_task`)
- **File name** = subtask name in snake_case (`subtask_one`)
- **Class name** = PascalCase version of filename (`SubtaskOne`)

Example:
```
src/tasks/classify/
  environmental_claims.py    # File name
    class EnvironmentalClaims:  # Class name (auto-discovered)
```

## Metadata Configuration

### Group Metadata

```yaml
name: my_task
display_name: My Task
description: Brief task description

# Capability requirements
capability_requirements:
  requires_logprobs: preferred  # or required, optional
  requires_multimodal: optional
  requires_schema: optional
  requires_files: optional
  requires_streaming: optional

# Subtask metadata
subtasks:
  subtask_one:
    description: Detailed description
    citation: "BibTeX citation"
    url: https://link-to-dataset
    credits: Attribution if needed
```

### Capability Requirements

Replace the deprecated `provider_types` field:

**Old way** (deprecated):
```json
{
  "provider_types": ["openai", "anthropic", "surus"]
}
```

**New way**:
```yaml
capability_requirements:
  requires_logprobs: preferred
  requires_multimodal: required
```

Levels:
- `required`: Task won't run without this capability
- `preferred`: Task prefers it but can fall back
- `optional`: Task can use it if available

## Migration from Old System

### Before (Old System)

```
tasks/my_task/
  ├── task.json (103 lines)
  ├── run.py (80 lines)
  └── task.py (192 lines)
```

### After (Handler System)

```
tasks/my_task/
  ├── metadata.yaml (15 lines)
  ├── subtask_one.py (12 lines)
  └── subtask_two.py (10 lines)
```

**90% reduction in boilerplate!**

### Conversion Steps

1. **Create metadata.yaml** from task.json
2. **Create handler classes** for each subtask
3. **Test** using `benchy eval --model-name gpt-4o-mini --tasks my_task`
4. **Remove** old task.json, run.py, task.py

### Migration Status

The handler system is now the only supported task registration method. Legacy
`task.json` task registration has been removed.

## Best Practices

### 1. Use Appropriate Handler

Choose the handler that matches your task format:
- **Classification/MCQ** → `MultipleChoiceHandler`
- **Extraction/Schema** → `StructuredHandler`
- **Translation/Generation** → `FreeformHandler`
- **Image→JSON** → `MultimodalStructuredHandler`

### 2. Keep Handlers Minimal

Only override what you need. Most handlers work with just dataset config:

```python
class MyTask(MultipleChoiceHandler):
    dataset = "org/dataset"
    labels = {0: "No", 1: "Yes"}
    # That's enough for many tasks!
```

### 3. Document in Docstrings

Use clear docstrings since they appear in documentation:

```python
class MyTask(MultipleChoiceHandler):
    """Binary sentiment classification on IMDB reviews.
    
    This task evaluates models on their ability to classify
    movie reviews as positive or negative sentiment.
    """
```

### 4. Use Metadata for Citations

Put citations and URLs in metadata.yaml, not code:

```yaml
subtasks:
  my_subtask:
    citation: "@article{...}"
    url: https://dataset-link
```

### 5. Test Incrementally

Test your handler before adding complexity:

```bash
# Test with small limit first
benchy eval --model-name gpt-4o-mini --tasks my_task --limit 10

# Then full evaluation
benchy eval --model-name gpt-4o-mini --tasks my_task
```

## Troubleshooting

### Handler Not Discovered

**Problem**: Task not found when running `benchy eval --tasks my_task ...`

**Solutions**:
1. Check file is named correctly (snake_case)
2. Check class name matches PascalCase of filename
3. Ensure file is in `src/tasks/my_task/` directory
4. File should not start with `_` or be named `__init__`, `run`, `base`, `task`

### Dataset Not Loading

**Problem**: Dataset fails to load or download

**Solutions**:
1. Check `dataset` attribute is set correctly
2. Verify HuggingFace dataset path is correct
3. Check `split` attribute matches available splits
4. For local datasets, ensure `.data/` directory exists

### Metrics Not Calculating

**Problem**: Metrics show as 0 or missing

**Solutions**:
1. Check `expected` field is in preprocessed samples
2. Verify label mapping is correct for MCQ tasks
3. For structured tasks, ensure `schema` field is present
4. Check metric objects are imported correctly

### Custom Prompt Not Working

**Problem**: Custom `get_prompt` method not being called

**Solutions**:
1. Check method signature matches: `get_prompt(self, sample)`
2. Ensure method returns tuple: `(system_prompt, user_prompt)`
3. Verify sample has required fields

## Examples

### Complete MCQ Task

```python
# src/tasks/sentiment/imdb.py
from ..common import MultipleChoiceHandler

class Imdb(MultipleChoiceHandler):
    """IMDB movie review sentiment classification."""
    
    dataset = "imdb"
    split = "test"
    text_field = "text"
    label_field = "label"
    labels = {0: "Negative", 1: "Positive"}
    system_prompt = "You are a sentiment analyzer."
```

```yaml
# src/tasks/sentiment/metadata.yaml
name: sentiment
display_name: Sentiment Analysis
description: Binary sentiment classification tasks

capability_requirements:
  requires_logprobs: preferred

subtasks:
  imdb:
    description: IMDB movie review sentiment (binary)
    url: https://huggingface.co/datasets/imdb
```

### Complete Structured Task

```python
# src/tasks/extraction/entities.py
from ..common import StructuredHandler

class Entities(StructuredHandler):
    """Named entity extraction from news articles."""
    
    dataset = "conll2003"
    split = "test"
    system_prompt = "Extract named entities from the text."
    
    metrics_config = {
        "partial_matching": {
            "string": {"exact_threshold": 0.90}
        }
    }
```

## Advanced Topics

### Sharing Logic Between Subtasks

Create a base class for shared logic:

```python
# src/tasks/my_task/base.py
from ..common import MultipleChoiceHandler

class MyTaskBase(MultipleChoiceHandler):
    """Shared logic for all my_task subtasks."""
    
    system_prompt = "You are a helpful classifier."
    
    def preprocess_sample(self, raw_sample, idx):
        # Shared preprocessing
        cleaned = self._clean_text(raw_sample["text"])
        return super().preprocess_sample(
            {**raw_sample, "text": cleaned}, 
            idx
        )

# src/tasks/my_task/subtask_one.py
from .base import MyTaskBase

class SubtaskOne(MyTaskBase):
    dataset = "org/dataset-1"
    labels = {0: "No", 1: "Yes"}
```

### Dynamic Dataset Loading

Override `load_dataset` for custom loading:

```python
class CustomTask(FreeformHandler):
    def load_dataset(self):
        """Custom dataset loading."""
        # Load from custom source
        raw_data = self._load_custom_format()
        
        # Process into standard format
        samples = []
        for idx, item in enumerate(raw_data):
            sample = self.preprocess_sample(item, idx)
            if sample:
                samples.append(sample)
        
        return samples
```

## Summary

The handler system makes task creation:
- **Simple**: 5-20 lines vs 300+ lines
- **Discoverable**: Convention-based, no entrypoint strings
- **Reusable**: Format logic shared across tasks
- **Flexible**: Override only what you need
- **Composable**: Mix and match capabilities

Start with a handler, add minimal config, and you're done!
