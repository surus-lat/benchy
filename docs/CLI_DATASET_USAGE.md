# CLI Dataset Usage Guide

This guide explains how to evaluate custom datasets using Benchy's CLI without writing any code.

## Table of Contents

- [Overview](#overview)
- [Task Types](#task-types)
- [Dataset Sources](#dataset-sources)
- [Dataset Formats](#dataset-formats)
- [Complete Examples](#complete-examples)
- [CLI Reference](#cli-reference)
- [Advanced Usage](#advanced-usage)

## Overview

Benchy supports three ways to create tasks:

1. **Python Class**: Define tasks as Python classes (traditional approach)
2. **YAML Config**: Define tasks in configuration files
3. **CLI Flags**: Create ad-hoc tasks directly from command line (new!)

All three approaches use the same underlying handler system, so there's no difference in evaluation quality or features.

### When to Use CLI Tasks

✅ **Use CLI tasks when**:
- Evaluating a custom dataset that matches standard formats
- Quick experimentation with different datasets
- Automating evaluations across multiple datasets
- You don't need custom preprocessing logic

❌ **Use Python/YAML tasks when**:
- Dataset requires complex preprocessing
- Need custom metrics or evaluation logic
- Building reusable task definitions for a benchmark suite

## Task Types

Benchy supports three task types via CLI:

### 1. Classification

Binary or multi-class classification tasks. The model chooses from a fixed set of labels.

**Use for**: Sentiment analysis, topic classification, yes/no questions, entity type detection

**Required**:
- Input text field
- Output label field
- Label mapping (dict of label values to choice text)

**Example**:
```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name climatebert/environmental_claims \
  --dataset-labels '{"0": "No", "1": "Yes"}' \
  --system-prompt "Classify if the text contains an environmental claim." \
  --limit 10
```

### 2. Structured Extraction

Extract structured data (JSON) from text according to a schema.

**Use for**: Information extraction, form filling, data normalization, entity extraction

**Required**:
- Input text field
- Output JSON field
- Schema (from dataset field, file, or inline)

**Example**:
```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name my-org/invoice-extraction \
  --dataset-schema-path schemas/invoice_schema.json \
  --system-prompt "Extract invoice information as JSON." \
  --limit 10
```

### 3. Freeform Generation

Open-ended text generation with reference answers.

**Use for**: Question answering, summarization, translation, creative writing

**Required**:
- Input text field
- Expected output field

**Example**:
```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type freeform \
  --dataset-name ./data/qa_pairs.jsonl \
  --dataset-source local \
  --user-prompt-template "Question: {text}\nAnswer:" \
  --limit 10
```

## Dataset Sources

Benchy can load datasets from three sources:

### 1. HuggingFace Hub

Load datasets directly from HuggingFace:

```bash
--dataset-name climatebert/environmental_claims
--dataset-source huggingface  # or 'auto' (default)
--dataset-split test
```

### 2. Local JSONL Files

Load from local JSONL files:

```bash
--dataset-name ./data/my_dataset.jsonl
--dataset-source local  # or 'auto' (default)
```

### 3. Directory Structures

Load from directories (useful for multimodal tasks):

```bash
--dataset-name ./data/images/
--dataset-source directory
--multimodal-input
```

**Auto-detection** (default): Benchy automatically detects the source based on the path/name.

## Dataset Formats

### Classification Format

**Minimum required fields**:
```jsonl
{"id": "1", "text": "Input text here", "label": 0}
{"id": "2", "text": "Another input", "label": 1}
```

**With custom field names**:
```jsonl
{"sample_id": "1", "question": "Is this positive?", "answer": "yes"}
```

Use field mapping:
```bash
--dataset-id-field sample_id \
--dataset-input-field question \
--dataset-output-field answer \
--dataset-labels '{"yes": "Positive", "no": "Negative"}'
```

### Structured Extraction Format

**With schema in dataset**:
```jsonl
{"id": "1", "text": "Invoice from...", "schema": {...}, "expected": {...}}
```

**With external schema file**:
```jsonl
{"id": "1", "text": "Invoice from...", "expected": {...}}
```

Use schema file:
```bash
--dataset-schema-path schemas/invoice_schema.json
```

**Schema file format** (JSON):
```json
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": "string"},
    "date": {"type": "string", "format": "date"},
    "total": {"type": "number"}
  },
  "required": ["invoice_number", "date", "total"]
}
```

### Freeform Format

**Minimum required fields**:
```jsonl
{"id": "1", "text": "What is AI?", "expected": "AI is..."}
{"id": "2", "text": "Explain ML", "expected": "ML is..."}
```

### Multimodal Format

**With metadata file** (`metadata.jsonl` in directory):
```jsonl
{"id": "1", "image_path": "images/001.jpg", "text": "Describe", "expected": "..."}
{"id": "2", "image_path": "images/002.jpg", "text": "Describe", "expected": "..."}
```

**Directory scan** (no metadata file):
```
data/images/
├── 001.jpg
├── 002.jpg
└── 003.jpg
```

Benchy will auto-discover images and use filenames as IDs.

## Complete Examples

### Example 1: Sentiment Classification

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name sentiment140 \
  --dataset-split test \
  --dataset-input-field text \
  --dataset-output-field sentiment \
  --dataset-labels '{"0": "Negative", "2": "Neutral", "4": "Positive"}' \
  --system-prompt "Classify the sentiment of the following text." \
  --limit 100
```

### Example 2: Invoice Extraction

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name ./data/invoices.jsonl \
  --dataset-source local \
  --dataset-input-field document_text \
  --dataset-output-field extracted_data \
  --dataset-schema-path schemas/invoice_schema.json \
  --system-prompt "Extract structured information from the invoice." \
  --save-config configs/invoice-extraction.yaml \
  --limit 50
```

### Example 3: Question Answering

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type freeform \
  --dataset-name squad \
  --dataset-split validation \
  --dataset-input-field question \
  --dataset-output-field answer \
  --user-prompt-template "Context: {context}\n\nQuestion: {question}\n\nAnswer:" \
  --limit 100
```

### Example 4: Multimodal Image Classification

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name ./data/animal_images/ \
  --dataset-source directory \
  --multimodal-input \
  --multimodal-image-field image_path \
  --dataset-labels '{"0": "Cat", "1": "Dog", "2": "Bird"}' \
  --system-prompt "Classify the animal in the image." \
  --limit 50
```

### Example 5: Override Existing Task Dataset

```bash
# Use your own dataset with an existing task
benchy eval --config my-model.yaml \
  --tasks classify.environmental_claims \
  --dataset-name my-org/custom-climate-dataset \
  --dataset-split validation \
  --limit 100
```

## CLI Reference

### Task Type Selection

```bash
--task-type {classification,structured,freeform}
```

### Dataset Configuration

```bash
--dataset-name <name>              # HF dataset, local file, or directory
--dataset-source <source>          # auto, huggingface, local, directory
--dataset-split <split>            # HF split (default: test)
```

### Field Mappings

```bash
--dataset-input-field <field>      # Input text field (default: text)
--dataset-output-field <field>     # Expected output (default: expected/label)
--dataset-id-field <field>         # Sample ID (default: id, auto-generated)
```

### Classification Options

```bash
--dataset-label-field <field>      # Label field (default: label)
--dataset-labels <json>            # Label mapping: '{"0": "No", "1": "Yes"}'
--dataset-choices-field <field>    # Per-sample choices field
```

### Structured Extraction Options

```bash
--dataset-schema-field <field>     # Schema field in dataset
--dataset-schema-path <path>       # JSON file with schema
--dataset-schema-json <json>       # Inline JSON schema
```

### Multimodal Options

```bash
--multimodal-input                 # Enable multimodal input
--multimodal-image-field <field>   # Image path field (default: image_path)
```

### Prompts

```bash
--system-prompt <text>             # System prompt
--user-prompt-template <text>      # Template with {field} placeholders
```

### Config Generation

```bash
--save-config <path>               # Save as reusable YAML config
```

## Advanced Usage

### Custom Field Mappings

If your dataset uses non-standard field names:

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name my-dataset \
  --dataset-id-field sample_id \
  --dataset-input-field question_text \
  --dataset-output-field correct_answer \
  --dataset-label-field answer_label \
  --dataset-labels '{"A": "Option A", "B": "Option B"}' \
  --limit 10
```

### Per-Sample Choices

For classification tasks where each sample has different choices:

```jsonl
{"id": "1", "text": "What is 2+2?", "choices": ["3", "4", "5"], "expected": 1}
{"id": "2", "text": "What is 3+3?", "choices": ["5", "6", "7"], "expected": 1}
```

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name ./data/math_questions.jsonl \
  --dataset-source local \
  --dataset-choices-field choices \
  --limit 10
```

### Inline Schema

For simple schemas, you can provide JSON inline:

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name my-dataset \
  --dataset-schema-json '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "required": ["name", "age"]}' \
  --limit 10
```

### Prompt Templates

Use field placeholders in prompts:

```bash
--user-prompt-template "Context: {context}\n\nQuestion: {question}\n\nProvide a detailed answer:"
```

All fields from your dataset are available as placeholders.

### Config Generation Workflow

1. **Create and test** with CLI:
```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name my-dataset \
  --dataset-labels '{"0": "No", "1": "Yes"}' \
  --save-config configs/my-task.yaml \
  --limit 5
```

2. **Review** the generated config:
```bash
cat configs/my-task.yaml
```

3. **Run full evaluation** with config:
```bash
benchy eval --config configs/my-task.yaml --limit 1000
```

4. **Reuse** across models:
```bash
benchy eval --config configs/my-task.yaml --model-name claude-3-5-sonnet --provider anthropic
```

### Combining with Existing Configs

You can combine CLI dataset parameters with model configs:

```bash
benchy eval --config configs/models/my-model.yaml \
  --task-type classification \
  --dataset-name my-dataset \
  --dataset-labels '{"0": "No", "1": "Yes"}' \
  --limit 100
```

### Batch Evaluation Across Datasets

Use shell scripting to evaluate multiple datasets:

```bash
for dataset in dataset1 dataset2 dataset3; do
  benchy eval --config my-model.yaml \
    --task-type classification \
    --dataset-name $dataset \
    --dataset-labels '{"0": "No", "1": "Yes"}' \
    --run-id ${dataset}_$(date +%Y%m%d) \
    --limit 100
done
```

## Troubleshooting

### Common Issues

**Issue**: "Invalid task configuration for type 'classification': Task type 'classification' requires at least one of: 'labels', 'choices_field'"

**Solution**: Provide either `--dataset-labels` for fixed labels or `--dataset-choices-field` for per-sample choices.

---

**Issue**: "Dataset configuration must include 'name'"

**Solution**: Provide `--dataset-name` with your dataset path or HuggingFace identifier.

---

**Issue**: "Structured tasks require schema via schema_field, schema_path, or schema_json"

**Solution**: Provide one of the schema options: `--dataset-schema-field`, `--dataset-schema-path`, or `--dataset-schema-json`.

---

**Issue**: "Sample X: Missing input field 'text'"

**Solution**: Your dataset uses a different field name. Use `--dataset-input-field` to specify the correct field.

---

**Issue**: "JSONL dataset not found"

**Solution**: Check the path and use `--dataset-source local` explicitly if auto-detection fails.

### Validation

Before running a full evaluation, test with a small limit:

```bash
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type classification \
  --dataset-name my-dataset \
  --dataset-labels '{"0": "No", "1": "Yes"}' \
  --limit 5  # Test with 5 samples first
```

Check the output for:
- Sample loading success
- Correct field mapping
- Expected prompt format
- Metric calculation

## Next Steps

- See handler documentation for task-specific details: `src/tasks/common/`
- Read the architecture guide: `docs/architecture.md`
- Explore example configs: `configs/templates/`
- Learn about metrics: Check handler class documentation
