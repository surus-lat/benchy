# Contributing Tasks

This guide explains how to add a new task to Benchy using the modern **handler-based system**. This system dramatically reduces boilerplate and makes task authoring straightforward.

## Quick Start

For most tasks, you'll write **30-50 lines of code** by inheriting from a format handler and declaring your configuration. The handler provides all the common logic for dataset loading, prompt generation, metrics calculation, and interface compatibility.

### Example: Multiple Choice Task

```python
# src/tasks/my_benchmark/my_task.py
from ..common import CachedDatasetMixin, MultipleChoiceHandler

class MyTask(CachedDatasetMixin, MultipleChoiceHandler):
    """Classify sentiment in movie reviews."""
    
    # Task metadata
    name = "my_task"
    display_name = "My Classification Task"
    description = "Classify movie reviews as positive or negative"
    
    # Dataset configuration
    dataset_name = "imdb"
    split = "test"
    dataset_file = "my_task_data.jsonl"
    
    # Task configuration
    labels = {0: "Negative", 1: "Positive"}
    system_prompt = "You are a sentiment classifier."
    
    # That's it! Handler provides everything else.
```

## Task System Overview

### Handler-Based Architecture

Benchy uses **format handlers** that encapsulate all common logic for a specific task format:

- **`MultipleChoiceHandler`** - Multiple-choice questions with logprobs scoring
- **`StructuredHandler`** - JSON extraction with schema validation
- **`FreeformHandler`** - Open-ended text generation
- **`MultimodalStructuredHandler`** - Vision-language models with structured output

These handlers implement the `BaseTask` protocol (see `src/engine/protocols.py`) which defines the core interface: `load()`, `get_samples()`, `get_prompt()`, `calculate_metrics()`, `aggregate_metrics()`, and capability properties.

### Dataset Loading Mixins

Common dataset patterns are available as mixins:

- **`CachedDatasetMixin`** - Download from HuggingFace, cache as JSONL
- **`CachedTSVMixin`** - Load from TSV files with caching
- **`CachedCSVMixin`** - Load from CSV files with caching

### Discovery and Registration

Handler-based tasks use **convention-based discovery**:

1. Create a folder under `src/tasks/` (e.g., `src/tasks/my_benchmark/`)
2. Add a `metadata.yaml` describing your task group
3. Create `.py` files for each subtask (e.g., `sentiment.py`, `toxicity.py`)
4. Export task classes in `__init__.py`

Tasks are automatically discovered and registered - no explicit entrypoint configuration needed.

## Step-by-Step Guide

### 1. Choose Your Handler

Select the handler that matches your task format:

| Task Type | Handler | Use When |
|-----------|---------|----------|
| Multiple choice (A/B/C/D) | `MultipleChoiceHandler` | Questions with predefined choices |
| JSON extraction | `StructuredHandler` | Extracting structured data from text |
| Open-ended text | `FreeformHandler` | Classification, generation, custom metrics |
| Vision + structure | `MultimodalStructuredHandler` | Extract data from images/documents |

### 2. Copy the Template

```bash
cp -r src/tasks/_template_handler src/tasks/my_benchmark
```

The template includes:
- `metadata.yaml` - Task group description
- `mcq_example.py` - Multiple choice example
- `structured_example.py` - Structured extraction example
- `freeform_example.py` - Freeform text example
- `README.md` - Complete documentation

### 3. Create `metadata.yaml`

Describe your task group:

```yaml
name: my_benchmark
display_name: My Benchmark Suite
description: A collection of tasks for evaluating my domain

capability_requirements:
  requires_logprobs: optional
  requires_multimodal: none
  requires_schema: optional

subtasks:
  sentiment:
    description: Sentiment classification on movie reviews
    dataset_url: https://huggingface.co/datasets/imdb
  
  toxicity:
    description: Detect toxic comments
    dataset_url: https://huggingface.co/datasets/toxicity
```

Valid capability values: `required`, `preferred`, `optional`, `none`

### 4. Implement Task Classes

#### Multiple Choice Task

```python
from ..common import CachedDatasetMixin, MultipleChoiceHandler

class Sentiment(CachedDatasetMixin, MultipleChoiceHandler):
    # Task metadata
    name = "sentiment"
    display_name = "Sentiment Analysis"
    
    # Dataset
    dataset_name = "imdb"
    split = "test"
    dataset_file = "sentiment_data.jsonl"
    
    # Configuration
    labels = {0: "Negative", 1: "Positive"}
    system_prompt = "Classify the sentiment."
    
    # Optional: Custom preprocessing
    def _download_and_cache(self, output_path):
        raw = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache")
        )
        
        processed = []
        for idx, sample in enumerate(raw):
            processed.append({
                "id": f"sentiment_{idx}",
                "text": sample["text"],
                "expected": sample["label"]
            })
        
        save_to_jsonl(processed, output_path)
```

#### Structured Extraction Task

```python
from ..common import CachedDatasetMixin, StructuredHandler

class EntityExtraction(CachedDatasetMixin, StructuredHandler):
    # Task metadata
    name = "entity_extraction"
    display_name = "Entity Extraction"
    
    # Dataset
    dataset_name = "org/entities"
    split = "test"
    dataset_file = "entities_data.jsonl"
    
    # Schema
    schema = {
        "type": "object",
        "properties": {
            "person": {"type": "string"},
            "organization": {"type": "string"},
            "location": {"type": "string"}
        },
        "required": ["person", "organization", "location"],
        "additionalProperties": False
    }
    
    # Prompts
    system_prompt = "Extract entities from the text."
    
    def _download_and_cache(self, output_path):
        # Download and preprocess data
        # Each sample needs: id, text, schema, expected
        ...
```

#### Freeform Text Task

```python
from ..common import FreeformHandler, F1Score, ExactMatch

class Summarization(FreeformHandler):
    name = "summarization"
    display_name = "Text Summarization"
    
    def __init__(self, config=None):
        super().__init__(config)
        # Define custom metrics
        self.metrics = [F1Score(), ExactMatch()]
    
    def load_dataset(self):
        # Load your dataset
        return [
            {"id": "1", "text": "...", "expected": "summary"},
            ...
        ]
    
    def get_prompt(self, sample):
        system = "You are a summarization assistant."
        user = f"Summarize this text:\n\n{sample['text']}"
        return system, user
```

### 5. Export Task Classes

Create `__init__.py` in your task folder:

```python
"""My benchmark module."""

from .sentiment import Sentiment
from .toxicity import Toxicity

__all__ = ["Sentiment", "Toxicity"]
```

### 6. Test Your Task

Run with a small sample limit:

```bash
benchy eval --config configs/models/openai_gpt-4o-mini.yaml \
    --tasks my_benchmark.sentiment --limit 5
```

## Handler Reference

### MultipleChoiceHandler

**Use for:** Questions with predefined answer choices

**Key attributes:**
```python
class MyMCQ(MultipleChoiceHandler):
    # Option 1: Task-level labels (all samples use same choices)
    labels = {0: "No", 1: "Yes", 2: "Maybe"}
    
    # Option 2: Per-sample choices (provide in data)
    # No labels attribute needed, each sample provides its own
    # choices, choice_labels, and expected index
    
    # Prompts
    system_prompt = "..."
    
    # Parsing behavior
    strict_parsing = True  # False for more lenient answer extraction
    
    # Scoring
    prefers_logprobs = True  # Use first-token logprobs when available
```

**Data format:**
```python
# Task-level labels mode:
{"id": "1", "text": "Is this positive?", "expected": 1}

# Per-sample choices mode:
{
    "id": "1",
    "text": "What is the capital?",
    "choices": ["Paris", "London", "Berlin"],
    "choice_labels": ["A", "B", "C"],
    "expected": 0
}
```

**Automatic features:**
- Choice formatting with labels
- Logprobs-based scoring (when available)
- Text parsing fallback
- Accuracy metrics

See extensive documentation in `src/tasks/common/multiple_choice.py`

### StructuredHandler

**Use for:** Extracting structured JSON from text

**Key attributes:**
```python
class MyExtraction(StructuredHandler):
    # Schema (can be per-sample or task-level)
    schema = {...}
    
    # Prompts
    system_prompt = "..."
    
    # Metrics configuration
    metrics_config = {
        "partial_matching": {
            "string": {
                "exact_threshold": 0.95,  # String similarity threshold
                "partial_threshold": 0.50
            }
        }
    }
```

**Data format:**
```python
{
    "id": "1",
    "text": "Extract from this...",
    "schema": {...},  # JSON schema
    "expected": {...}  # Expected output
}
```

**Automatic metrics:**
- Schema validity
- Exact match rate
- Field-level F1 (strict and partial)
- Hallucination rate
- Type compliance
- Extraction Quality Score (EQS)

### FreeformHandler

**Use for:** Open-ended generation or custom evaluation

**Key attributes:**
```python
class MyGeneration(FreeformHandler):
    def __init__(self, config=None):
        super().__init__(config)
        # Define your metrics
        self.metrics = [
            ExactMatch(),
            F1Score(),
            # Or custom metrics
        ]
    
    # Optional: postprocess model output
    def postprocess_prediction(self, prediction, sample):
        return prediction.strip().lower()
```

**Available metrics:**
- `ExactMatch` - Exact string match
- `F1Score` - Token-level F1
- `MeanSquaredError` - For regression
- `PearsonCorrelation` - For regression
- Custom metrics implementing `Metric` or `ScalarMetric` protocol

### MultimodalStructuredHandler

**Use for:** Vision-language models with structured output

**Key attributes:**
```python
class MyVision(MultimodalStructuredHandler):
    # Capability requirements
    requires_multimodal = True
    requires_files = True
    requires_schema = False  # Optional for some endpoints
    
    # File configuration
    input_type = "image"
    image_field = "image_path"
    schema_field = "schema"
    
    # Metrics (inherits from StructuredHandler)
    metrics_config = {
        "partial_matching": {
            "string": {
                "exact_threshold": 0.85,  # More lenient for OCR
                "partial_threshold": 0.40
            }
        }
    }
```

**Data format:**
```python
{
    "id": "1",
    "image_path": "/path/to/image.jpg",
    "schema": {...},
    "expected": {...}
}
```

## Advanced Topics

### Custom Data Loading

Override `_download_and_cache` for HuggingFace datasets:

```python
def _download_and_cache(self, output_path: Path):
    raw = download_huggingface_dataset(
        dataset_name=self.dataset_name,
        split=self.split,
        cache_dir=str(self.data_dir / "cache")
    )
    
    processed = []
    for idx, raw_sample in enumerate(raw):
        # Transform to your format
        processed.append({
            "id": f"task_{idx}",
            "text": raw_sample["input"],
            "expected": raw_sample["output"]
        })
    
    save_to_jsonl(processed, output_path)
```

Or override `load_dataset` for complete control:

```python
def load_dataset(self) -> list:
    # Load from any source
    data = load_my_custom_format("data.json")
    
    return [
        {"id": "1", "text": "...", "expected": "..."},
        ...
    ]
```

### Few-Shot Examples

```python
class MyTask(MultipleChoiceHandler):
    fewshot_split = "train"
    fewshot_id_column = "id"
    fewshot_id_list = ["example_1", "example_2", "example_3"]
    num_fewshot = 3
    exclude_from_task = True  # Don't include in test set
    
    def _load_fewshot_prompt(self) -> str:
        # Load and format few-shot examples
        fewshot_file = self.data_dir / "fewshot.jsonl"
        examples = self.load_dataset_from_path(fewshot_file)
        
        formatted = [self._format_fewshot_example(ex) for ex in examples]
        return build_fewshot_block(formatted)
    
    def _format_fewshot_example(self, sample: Dict) -> str:
        return f"Q: {sample['text']}\nA: {sample['answer']}"
```

### Custom Metrics

```python
from dataclasses import dataclass
from ..common import ScalarMetric

@dataclass(frozen=True)
class MyMetric(ScalarMetric):
    name: str = "my_metric"
    
    def compute(self, prediction, expected, sample):
        # Return a numeric score
        return calculate_score(prediction, expected)

class MyTask(FreeformHandler):
    def __init__(self, config=None):
        super().__init__(config)
        self.metrics = [MyMetric()]
```

### Strict vs. Lenient Parsing

Control parsing behavior with configuration:

```python
# Multiple choice - lenient parsing
class MyMCQ(MultipleChoiceHandler):
    strict_parsing = False  # More permissive answer extraction

# Structured extraction - lenient metrics
class MyExtraction(StructuredHandler):
    metrics_config = {
        "strict": False,  # Use lenient thresholds
        "partial_matching": {
            "string": {
                "exact_threshold": 0.85,  # Lower than default 0.95
                "partial_threshold": 0.40  # Lower than default 0.50
            }
        }
    }
```

## Common Utilities

Import from `src.tasks.common`:

**Text Processing:**
- `normalize_spaces(text)` - Collapse whitespace
- `remove_accents(text)` - Remove diacritics
- `normalize_text(text)` - Full normalization
- `extract_float_score(text, min, max)` - Extract numeric scores
- `format_choices(choices, labels)` - Format MCQ choices
- `extract_choice_letter(response, labels)` - Parse MCQ answers
- `build_fewshot_block(examples)` - Format few-shot examples

**Dataset I/O:**
- `download_huggingface_dataset(name, split)` - Download from HF
- `save_to_jsonl(data, path)` - Save as JSONL
- `load_jsonl(path)` - Load JSONL
- `load_tsv(path)` - Load TSV
- `load_csv(path)` - Load CSV

**Metrics:**
- `ExactMatch()` - Exact string match
- `F1Score()` - Token-level F1
- `MultipleChoiceAccuracy()` - MCQ accuracy
- `MeanSquaredError()` - MSE for regression
- `PearsonCorrelation()` - Correlation for regression

## Capability Requirements

Declare what your task needs in `metadata.yaml`:

```yaml
capability_requirements:
  requires_logprobs: preferred    # Nice to have for MCQ scoring
  requires_multimodal: required   # Must support images
  requires_schema: preferred      # Nice to have structured output
  requires_files: required        # Must support file uploads
  requires_streaming: none        # Don't need streaming
```

Tasks are automatically skipped if required capabilities are missing.

## Output Directory Structure

Handler-based tasks produce:

```
outputs/benchmark_outputs/<run_id>/<model>/
└── my_benchmark/
    ├── task_status.json
    ├── sentiment/
    │   ├── model_timestamp_metrics.json
    │   ├── model_timestamp_samples.json
    │   └── model_timestamp_report.txt
    ├── toxicity/
    │   └── ...
    └── ...
```

## Task System

Benchy now supports only the handler-based task system. Tasks are discovered from
`src/tasks/<group>/metadata.yaml` plus subtask modules in the same directory.

## Best Practices

1. **Start simple**: Use a handler and only override what you need
2. **Cache datasets**: Use mixins to avoid repeated downloads
3. **Validate early**: Test with `--limit 5` before full runs
4. **Document prompts**: Clear system prompts improve reproducibility
5. **Test metrics**: Verify metrics on a few samples manually
6. **Use strict by default**: Lenient parsing should be opt-in
7. **Declare capabilities**: Help users know what's required

## Examples

See real-world examples in:
- `src/tasks/spanish/` - Multiple choice tasks with Spanish text
- `src/tasks/portuguese/` - MCQ and regression tasks
- `src/tasks/structured_extraction/` - JSON extraction from text
- `src/tasks/image_extraction/` - Document extraction from images
- `src/tasks/_template_handler/` - Annotated templates

## Getting Help

- Read handler docstrings: `src/tasks/common/multiple_choice.py` has 200+ lines of docs
- Check templates: `src/tasks/_template_handler/README.md`
- Review examples: Look at migrated tasks for patterns
- Ask questions: Open an issue with the `task-development` label

## Contributing

After implementing your task:

1. Test with multiple providers (vLLM, OpenAI, etc.)
2. Verify metrics are sensible
3. Update task group description in `metadata.yaml`
4. Consider adding to leaderboard in `configs/config.yaml`
5. Submit a PR with your task and any new utilities

Your task becomes part of Benchy's growing benchmark suite!
