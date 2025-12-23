# Image Extraction Task

Evaluates vision-language models on structured data extraction from document images (invoices, forms, receipts, etc.).

## Overview

Given an image and a JSON schema, the model must extract structured data that conforms to the schema. This task measures how accurately models can read and interpret visual documents.

**Key characteristics:**
- Requires multimodal (vision) model support
- Uses JSON schema for structured output
- Optimized for document extraction where numeric accuracy is critical

## Data Structure

The task expects data in `.data/` (or a custom `source_dir`):

```
.data/
├── schema.json          # JSON Schema defining expected output structure
├── datos.json           # Ground truth: list of expected extractions
├── metrics_config.json  # Dataset-specific metrics configuration (optional)
└── jpgs/                # Document images
    ├── doc_001.jpg
    ├── doc_002.jpg
    └── ...
```

### schema.json

Standard JSON Schema (draft-07) defining the extraction target:

```json
{
  "type": "object",
  "properties": {
    "invoice_number": { "type": "string" },
    "date": { "type": "string", "format": "date" },
    "total": { "type": "number" }
  },
  "required": ["invoice_number", "date", "total"]
}
```

### datos.json

Array of ground truth extractions, one per image:

```json
[
  {
    "filename": "doc_001",
    "invoice_number": "00012345",
    "date": "2025-01-15",
    "total": 1500.00
  }
]
```

The `filename` field maps to `jpgs/{filename}.jpg`.

### metrics_config.json (optional)

Dataset-specific metrics configuration. If present, overrides task-level metrics config:

```json
{
  "numeric_string_fields": ["punto_de_venta", "numero_de_comprobante"],
  "ignored_fields": ["IVA", "IBB"],
  "partial_matching": {
    "string": {
      "exact_threshold": 0.85,
      "partial_threshold": 0.40
    }
  },
  "document_extraction_score": {
    "weights": {
      "numeric_precision": 0.50,
      "field_f1_partial": 0.35,
      "schema_validity": 0.15
    }
  }
}
```

This allows different datasets to have different evaluation criteria without changing the task config.

## Metrics

The task outputs comprehensive metrics at both sample and aggregate levels.

### Primary Metrics

| Metric | Description |
|--------|-------------|
| `document_extraction_score` | Weighted score emphasizing numeric accuracy (0-1) |
| `field_f1_partial` | F1 score with partial credit for close matches |
| `exact_match_rate` | Percentage of samples with perfect extraction |
| `schema_validity_rate` | Percentage of outputs conforming to schema |

### Document Extraction Score

Combines three components with weights optimized for document extraction:

- **Numeric precision (50%)**: Exact matches on numeric fields
- **Field F1 partial (35%)**: Overall extraction quality with partial credit
- **Schema validity (15%)**: Structural compliance

### Match Distribution

Shows how field matches break down:

- `exact`: Perfect match
- `partial`: Close match (e.g., minor string variations)
- `incorrect`: Wrong value
- `missed`: Field not extracted
- `spurious`: Extra field not in ground truth (hallucination)

**Note:** Fields listed in `ignored_fields` are excluded from evaluation and won't appear in match distribution. They're still validated for schema compliance but don't affect scores.

## Value Normalization

The metrics system applies smart normalization before comparison:

1. **DateTime fields** (schema `format: "date"` or `"date-time"`): Compared as dates only. `2025-01-15T00:00:00` equals `2025-01-15`.

2. **Numeric string fields** (configurable): Converted to integers. `"0001"` equals `"1"`.

3. **String fields**: Lenient matching tolerates minor variations in casing, spacing, and typos.

## Configuration

### Task Configuration

Task-level configuration (typically in `src/tasks/image_extraction/task.json`):

```json
{
  "source_dir": "/path/to/data",
  "prompts": {
    "system": "Extract structured data from the document image.",
    "user": "Extract data following this schema:"
  },
  "metrics": {
    "partial_matching": {
      "string": {
        "exact_threshold": 0.85,
        "partial_threshold": 0.4
      }
    }
  }
}
```

### Dataset-Specific Configuration

Metrics configuration is **dataset-specific** and should be placed in `.data/metrics_config.json`:

```json
{
  "numeric_string_fields": ["punto_de_venta", "numero_de_comprobante"],
  "ignored_fields": ["IVA", "IBB"],
  "partial_matching": {
    "string": {
      "exact_threshold": 0.85,
      "partial_threshold": 0.40
    }
  },
  "document_extraction_score": {
    "weights": {
      "numeric_precision": 0.50,
      "field_f1_partial": 0.35,
      "schema_validity": 0.15
    }
  }
}
```

**Priority:** Dataset config (`.data/metrics_config.json`) overrides task config. This allows different datasets to have different evaluation criteria.

## Adapting for New Datasets

1. **Prepare your data**: Create `schema.json`, `datos.json`, and image files
2. **Update schema**: Define your extraction target fields with appropriate types
3. **Configure normalization**: Add any numeric string fields to `numeric_string_fields`
4. **Run**: The task auto-copies data from `source_dir` on first run

### Adding a New Document Type

For a new schema, the metrics system automatically handles:
- Date fields via schema `format` attribute
- Numeric fields via schema `type: "number"` or `"integer"`
- String leniency via configurable thresholds

**Configuration updates needed (in `.data/metrics_config.json`):**
- `numeric_string_fields`: String fields that should be compared as integers (e.g., zero-padded IDs)
- `ignored_fields`: Fields in schema but not in expected data (e.g., intermediate calculations like `IVA`, `IBB` that sum to `total_impuestos`)

**Example:** If your schema includes `IVA`, `IBB`, and `total_impuestos`, but your ground truth only has `total_impuestos`, add `IVA` and `IBB` to `ignored_fields` in `metrics_config.json`. The model can return them (and they'll be validated), but they won't affect the evaluation score.

## Module Structure

```
image_extraction/
├── task.py      # ImageExtractionTask - data loading, prompts, orchestration
├── metrics.py   # DocumentExtractionMetrics - value normalization, scoring
├── run.py       # Prefect task entry point for benchmark pipeline
└── .data/       # Dataset (auto-populated from source_dir)
```

## Usage

The task integrates with the benchmark engine. Typical invocation:

```python
from src.tasks.image_extraction.task import ImageExtractionTask

task = ImageExtractionTask({
    'source_dir': '/path/to/data',
    'prompts': {'system': '...', 'user': '...'},
})
task.load()

for sample in task.get_samples(limit=10):
    # sample contains: id, image_path, schema, expected, filename
    system_prompt, user_prompt = task.get_prompt(sample)
    # ... run model inference ...
    metrics = task.calculate_metrics(prediction, sample['expected'], sample)
```

