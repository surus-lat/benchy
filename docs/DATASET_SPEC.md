# Benchy Dataset Specification

This document defines the conventions for building datasets that work with Benchy's zero-code CLI evaluation. Following this spec ensures datasets are auto-discoverable and work end-to-end without writing Python.

## Directory Layout

Place datasets under `.data/<dataset-name>/` at the repo root:

```
.data/
  my-dataset/
    data/
      train.parquet        # Optional
      validation.parquet   # Optional
      test.parquet         # Required (default eval split)
    dataset_info.json      # Required - metadata and column descriptions
    schema.json            # Required for extraction tasks, optional for classification
    metrics_config.json    # Optional - dataset-specific scoring overrides
    benchy.md              # Required - run commands and dataset description
    README.md              # Optional - provenance, methodology notes
    manifest.json          # Optional - build manifest for reproducibility
```

## Required Files

### `dataset_info.json`

Minimum fields:

```json
{
  "dataset_id": "org/dataset-name",
  "version": "1.0.0",
  "description": "One-line description of the dataset.",
  "workflow": "document-extraction | document-classification | text-classification | text-extraction",

  "splits": {
    "test": { "num_rows": 100 }
  },

  "features": {
    "record_id":    { "dtype": "string", "description": "Unique row identifier" },
    "input_bytes":  { "dtype": "large_binary", "description": "Raw file bytes (PDF/image)" },
    "label":        { "dtype": "string", "description": "Ground truth label" }
  }
}
```

**For classification datasets**, add:

```json
{
  "label_distribution": {
    "positive": 120,
    "negative": 80
  }
}
```

Benchy auto-discovers labels from `label_distribution` keys. Without it, you must pass `--dataset-labels` on the CLI.

### `schema.json` (extraction tasks)

Two formats are supported:

**Standard JSON Schema** (recommended for nested structures):

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Full name of the person",
      "ground_truth_available": true,
      "gt_field": "gt_name"
    },
    "address": {
      "type": "object",
      "properties": {
        "street": {
          "type": "string",
          "ground_truth_available": true,
          "gt_field": "gt_street"
        }
      }
    }
  }
}
```

**Custom fields format** (simpler, flat schemas only):

```json
{
  "fields": [
    {
      "name": "invoice_number",
      "type": "string",
      "description": "The invoice number",
      "required": true,
      "ground_truth_available": true,
      "ground_truth_field": "gt_invoice_number"
    }
  ]
}
```

Both formats are auto-detected. The `gt_field` / `ground_truth_field` annotations link schema fields to parquet columns so Benchy can build the expected output automatically.

### `benchy.md`

Every dataset must include a `benchy.md` with:

1. One-line description
2. Smoke test command (`--limit 3`)
3. Full run command
4. Custom prompt example
5. Key details (task type, input format, schema format, sample count)

See the `.data/` directory for reference examples.

### `metrics_config.json` (optional)

Use `metrics_config.json` for dataset-specific evaluation behavior. This file is loaded automatically from `.data/<dataset-name>/metrics_config.json` for zero-code `structured` and multimodal/document extraction datasets.

Typical uses:

- Make specific arrays order-insensitive during scoring
- Compare selected string fields as digits-only numeric IDs
- Ignore schema-valid fields that should not affect the score
- Tune partial-matching thresholds and score weights for a dataset
- Adjust field-diagnostics report verbosity

Example:

```json
{
  "unordered_arrays": {
    "cronograma": {
      "key_fields": ["fecha", "hora"]
    }
  },
  "numeric_string_fields": ["telefono", "n_de_beneficio"],
  "ignored_fields": ["metadata.*"],
  "partial_matching": {
    "string": {
      "exact_threshold": 0.95,
      "partial_threshold": 0.50
    },
    "number": {
      "relative_tolerance": 0.001,
      "absolute_tolerance": 1e-6
    }
  },
  "document_extraction_score": {
    "weights": {
      "numeric_precision_rate": 0.50,
      "field_f1_partial": 0.35,
      "schema_validity": 0.15
    }
  },
  "field_diagnostics": {
    "enabled": true,
    "max_examples_per_field": 10
  }
}
```

Supported keys:

- `unordered_arrays`: map of field path -> options for non-positional array scoring
- `unordered_arrays.<path>.key_fields`: fields used to align predicted and expected array items
- `ignored_fields`: field-path patterns excluded from scoring
- `numeric_string_fields`: field-path patterns compared as digits-only values
- `critical_string_fields`: field-path patterns treated as critical when mismatched
- `partial_credit`: weight assigned to partial matches in `field_f1_partial`
- `strict`: enables stricter matching thresholds globally
- `partial_matching.string.*`: string thresholds and weights
- `partial_matching.number.*`: numeric tolerances
- `normalization.case_sensitive`
- `normalization.normalize_whitespace`
- `normalization.unicode_normalize`
- `normalization_penalty.null_string_to_null`
- `normalization_penalty.max`
- `extraction_quality_score.weights.*`
- `document_extraction_score.weights.*`
- `field_diagnostics.enabled`
- `field_diagnostics.max_examples_per_field`
- `field_diagnostics.max_fields_in_report`
- `field_diagnostics.max_value_chars`

Notes:

- Field-path patterns support exact paths plus `[]` for array wildcards and `*` wildcards.
- `unordered_arrays` is especially useful for arrays of objects where order is not meaningful in downstream systems.
- `metrics_config.json` overrides task-level/default metrics for that dataset only.

## Parquet Column Conventions

### Common columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `record_id` | string | Recommended | Stable row identifier |
| `input_bytes` | large_binary | For documents | Raw file bytes (PDF, JPG, PNG, etc.) |
| `input_filename` | string | Recommended | Original filename (used for extension detection) |
| `input_file_extension` | string | Recommended | Lowercase extension without dot (`pdf`, `jpg`) |

### Classification columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `label` | string | Yes | Ground truth label (must match `label_distribution` keys) |

### Extraction columns (ground truth)

Ground truth columns use the `gt_` prefix:

| Column | Type | Description |
|--------|------|-------------|
| `gt_<field_name>` | varies | Ground truth value for the schema field |

The mapping between schema fields and `gt_*` columns is declared in `schema.json` via `gt_field` annotations. Benchy reads these at discovery time and builds the expected output dict automatically.

For nested schemas, the GT mapping uses dot-notation: field path `address.street` maps to column `gt_street`.

### Binary document handling

When a parquet dataset has a `large_binary` column (typically `input_bytes`):

1. Benchy materializes each row's bytes to `.data/<dataset>/cache/<hash>.<ext>`
2. For LLM providers, documents are rendered to PNG images at 200 DPI (configurable via `--render-dpi`)
3. Images are sent as base64 multimodal content to the API
4. For custom API endpoints (`--api-url`), raw files are used without rendering

This is all automatic. No CLI flags needed beyond `--dataset-name`.

## Validation Checklist

Before considering a dataset ready:

- [ ] `dataset_info.json` has `features` with correct `dtype` values
- [ ] `dataset_info.json` has `splits` with `num_rows`
- [ ] Classification: `label_distribution` present with all label values
- [ ] Extraction: `schema.json` has `gt_field` for every evaluable field
- [ ] Extraction: every `gt_field` value matches an actual parquet column name
- [ ] Binary datasets: `input_file_extension` column present for correct rendering
- [ ] `data/test.parquet` exists and is the evaluation split
- [ ] Optional: `metrics_config.json` exists when dataset-specific scoring rules are needed
- [ ] `benchy.md` has working smoke test command
- [ ] Smoke test passes: `benchy eval --dataset-name <name> --task-type <type> --provider openai --model-name gpt-4o --limit 3`

## Common Pitfalls

1. **Missing `label_distribution`**: Classification datasets without it require manual `--dataset-labels` on every run. Always include it.

2. **Mismatched `gt_field` names**: If `schema.json` says `gt_field: "gt_name"` but the parquet column is `gt_nombre`, the field will silently have no expected value. Verify column names match.

3. **`$ref` in nested schemas**: OpenAI strict mode only supports `$ref` to top-level `$defs`. Benchy resolves inline refs automatically, but avoid deep circular references.

4. **Missing `input_file_extension`**: Without it, binary materialization defaults to `.bin` and rendering may fail. Always include the extension column.

5. **Sparse ground truth**: Fields with `ground_truth_available: false` are excluded from evaluation. Mark them explicitly in the schema rather than leaving `gt_field` empty.

6. **Labels as integers vs strings**: The `label` column value must match the `label_distribution` keys exactly. If parquet stores `0`/`1` but `label_distribution` has `"positive"`/`"negative"`, samples will be skipped.

7. **Positional array scoring when order is irrelevant**: Arrays of objects are scored positionally unless you opt into `unordered_arrays` in `metrics_config.json`. Use this for fields like `cronograma` when entry presence matters more than order.
