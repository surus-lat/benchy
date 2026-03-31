# Extraction Scoring Reference

This document defines every metric produced by Benchy's structured extraction evaluator, explains when each one is meaningful, and shows how to configure them via `metrics_config.json`.

## Table of Contents

- [Headline Metrics](#headline-metrics)
- [Tiered Evaluation](#tiered-evaluation)
- [Field-Level Metrics](#field-level-metrics)
- [String Comparison Pipeline](#string-comparison-pipeline)
- [Score Formulas](#score-formulas)
- [Configuration Reference](#configuration-reference)
- [Checklist: Setting Up a New Extraction Dataset](#checklist-setting-up-a-new-extraction-dataset)

---

## Headline Metrics

Every extraction run outputs these top-level scores in `*_performance_summary.json` and `*_metrics.json`:

| Metric | Range | Primary use |
|---|---|---|
| `document_extraction_score` | 0–1 | **Headline.** Configurable composite — see [Score Formulas](#score-formulas) |
| `extraction_quality_score` | 0–1 | Schema-focused composite (schema validity + F1 + anti-hallucination) |
| `reliable_f1_partial` | 0–1 or null | F1 on bounded-GT fields only — `null` when no tiers are configured |
| `freeform_f1_partial_gt_limited` | 0–1 or null | F1 on freeform fields after normalization — labeled GT-limited |
| `field_f1_partial` | 0–1 | F1 across all fields (unweighted mix — see warning below) |
| `field_f1_strict` | 0–1 | Exact-only F1 across all fields |
| `schema_validity_rate` | 0–1 | Fraction of outputs that pass JSON schema validation |
| `hallucination_rate` | 0–1 | Fraction of predicted fields not in the expected output |
| `numeric_precision_rate` | 0–1 | Accuracy on integer/numeric fields only |
| `critical_error_rate` | 0–1 | Fraction of fields with critical-severity errors |

> **Warning on `field_f1_partial`**: This metric mixes all fields equally. On datasets where freeform fields (names, addresses) have noisy GT alongside reliable enum/integer fields, this number is not interpretable as model quality. Use `reliable_f1_partial` instead and configure `field_tiers` — see [Tiered Evaluation](#tiered-evaluation).

---

## Tiered Evaluation

The core insight: **not all fields have the same GT reliability**.

| Field type | GT quality | Scoring approach |
|---|---|---|
| Enum (`tipo_vehiculo`, `trayecto`) | Reliable — bounded set, exact entries in system | Exact match; errors are real model errors |
| Integer (`edad`, `cant_de_traslados`) | Reliable — numeric, clear definition | Exact match with numeric tolerance |
| Date (`cronograma[].fecha`) | Reliable — structured format | Exact match after format normalization |
| Name (`apellido_y_nombre`) | Noisy — manually typed, word-order varies, abbreviations | Freeform normalization + partial match; label as GT-limited |
| Address (`origen.domicilio`) | Noisy — abbreviations, accent inconsistency, GT may differ from what model reads | Freeform normalization + partial match; label as GT-limited |
| Boolean (`es_autovalido`) | Potentially systematically wrong — data-entry defaults | Check for near-0% accuracy; if systematic, add to `ignored_fields` |

### Configuring tiers

In `metrics_config.json`:

```json
{
  "field_tiers": {
    "reliable": ["practica_a_realizar", "tipo_vehiculo", "trayecto", "edad", "cronograma[].fecha"],
    "freeform": ["apellido_y_nombre", "origen.domicilio", "origen.localidad"]
  }
}
```

Fields not listed in either tier are scored normally and tagged `unclassified`. They contribute to `field_f1_partial` but not to either tiered score.

### What you get

After configuring tiers, each run reports:

- `reliable_f1_partial`: F1 computed only over reliable-tier fields. This is your interpretable headline — if it's low, the model is genuinely making errors on well-defined fields.
- `freeform_f1_partial_gt_limited`: F1 computed over freeform-tier fields after normalization. Use for prompt iteration diagnostics, but do not interpret as ground truth.
- `document_extraction_score`: By default still uses the old formula, but you can configure it to be driven by `reliable_f1_partial` — see [Score Formulas](#score-formulas).

### Diagnosing systematic GT errors

If a boolean or enum field shows near-0% accuracy in `field_diagnostics`:

1. Look at the confusion: is the model always predicting one value and GT always the other?
2. Inspect a sample of images to see what the form actually shows.
3. If the model is correct and GT was a data-entry default, add the field to `ignored_fields`:

```json
{
  "ignored_fields": ["es_autovalido"]
}
```

The model still must include the field in its output (schema validation still applies), but it contributes nothing to any score. Document the reason in the dataset's `README.md`.

---

## Field-Level Metrics

Each field gets one of these outcomes:

| Outcome | Meaning | Score |
|---|---|---|
| `exact` | Match after normalization | 1.0 |
| `partial` | Composite similarity ≥ `partial_threshold` (default 0.50) | Composite score (0.5–0.95) |
| `incorrect` | Composite similarity < `partial_threshold` | Composite score (0–0.5) |
| `missed` | Field present in expected, absent in predicted | 0.0 |
| `spurious` | Field present in predicted, absent in expected (hallucination) | 0.0 |

Each outcome also has a severity tag used by `critical_error_rate`:

| Outcome | Severity |
|---|---|
| `exact` | none |
| `partial` (string) | minor |
| `incorrect` (string) | minor |
| `incorrect` (numeric) | critical |
| `missed` | critical |
| `spurious` | critical |
| Boolean mismatch | critical |

---

## String Comparison Pipeline

For every string field, Benchy applies this pipeline:

```
raw predicted string
    ↓
_normalize_string():
    lowercase
    collapse whitespace
    NFKD decompose + strip combining chars  ← removes accents ("Córdoba" → "cordoba")
    ↓
[if field is in freeform tier] _normalize_freeform():
    expand abbreviations ("av." → "avenida")
    NFKD + strip combining chars
    [if word_order_insensitive] sort tokens alphabetically
    ↓
composite score = 0.5 × token_overlap_F1
               + 0.3 × levenshtein_similarity
               + 0.2 × containment
    ↓
classify:
    ≥ exact_threshold (0.95) → exact
    ≥ partial_threshold (0.50) → partial
    < partial_threshold → incorrect
```

The freeform normalization layer handles the GT noise common in manual data entry:

- **Accents**: `Córdoba` vs `CÓRDOBA` vs `CORDOBA` → all normalize to `cordoba`
- **Abbreviations**: `Av. Belgrano` vs `Avenida Belgrano` → both become `avenida belgrano`
- **Word order**: `Marin Dora Antonia` vs `Dora Antonia Marin` → both become `antonia dora marin`

Note: the base `_normalize_string()` (applied to all fields) also strips accents via NFKD. This means even non-tiered string fields benefit from accent-insensitive comparison. Configure `normalization.unicode_normalize: false` to disable this if accent distinctions are semantically meaningful.

---

## Score Formulas

### Field F1 (partial and strict)

Applied across all non-ignored fields:

```
correct_partial = exact_count + partial_credit × partial_count   (partial_credit default: 0.3)

precision_partial = correct_partial / total_predicted_fields
recall_partial    = correct_partial / total_expected_fields
field_f1_partial  = 2 × P × R / (P + R)

field_f1_strict uses only exact_count (no partial credit)
```

### Reliable and Freeform tier F1

Same formula as field F1, but computed only over fields in the respective tier. Fields in neither tier are excluded. The sentinel value `-1.0` (per-sample) or `null` (aggregated) means the tier was not configured.

### Extraction Quality Score (EQS)

```
EQS = w_validity × schema_validity
    + w_f1       × field_f1_partial
    + w_halluc   × (1 - hallucination_rate)
```

Default weights: `schema_validity=0.15`, `field_f1_partial=0.70`, `inverted_hallucination=0.15`.

Configure via:
```json
{ "extraction_quality_score": { "weights": { "schema_validity": 0.15, "field_f1_partial": 0.70, "inverted_hallucination": 0.15 } } }
```

### Document Extraction Score (DES)

The headline metric. Two modes:

**Default** (no `document_extraction_score.weights` configured):
```
DES = 0.50 × numeric_precision_rate
    + 0.35 × field_f1_partial
    + 0.15 × schema_validity
```
Suited for tasks where numeric ID accuracy is the primary concern (invoices, forms with IDs).

**Configured** (`document_extraction_score.weights` present):
Only the listed weight keys contribute — no defaults are added. This prevents double-counting.

```json
{
  "document_extraction_score": {
    "weights": {
      "reliable_f1_partial": 0.70,
      "schema_validity": 0.15,
      "inverted_hallucination": 0.15
    }
  }
}
```
Results in:
```
DES = 0.70 × reliable_f1_partial
    + 0.15 × schema_validity
    + 0.15 × (1 - hallucination_rate)
```

Suited for tasks with tiered fields where you want the headline to reflect only bounded-GT performance.

All valid weight keys for DES:

| Key | Metric used |
|---|---|
| `reliable_f1_partial` | Tier F1 on reliable fields (skipped if tier not configured) |
| `field_f1_partial` | F1 across all fields |
| `numeric_precision_rate` | Accuracy on integer/numeric fields |
| `schema_validity` | 1.0 if schema valid, 0.0 otherwise |
| `inverted_hallucination` | `1 - hallucination_rate` |
| `inverted_critical_error_rate` | `1 - critical_error_rate` |

---

## Configuration Reference

Quick reference — all keys in `metrics_config.json`:

```json
{
  "unordered_arrays": {
    "<field_path>": { "key_fields": ["<key1>", "<key2>"] }
  },

  "ignored_fields": ["<field_path_pattern>"],

  "numeric_string_fields": ["<field_path_pattern>"],
  "critical_string_fields": ["<field_path_pattern>"],

  "field_tiers": {
    "reliable": ["<field_path_pattern>"],
    "freeform":  ["<field_path_pattern>"]
  },

  "freeform_normalization": {
    "strip_accents": true,
    "word_order_insensitive": true,
    "abbreviations": { "<abbr>": "<expansion>" }
  },

  "partial_credit": 0.3,
  "strict": false,

  "partial_matching": {
    "string": {
      "exact_threshold": 0.95,
      "partial_threshold": 0.50,
      "token_overlap_weight": 0.5,
      "levenshtein_weight": 0.3,
      "containment_weight": 0.2
    },
    "number": {
      "relative_tolerance": 0.001,
      "absolute_tolerance": 1e-6
    }
  },

  "normalization": {
    "case_sensitive": false,
    "normalize_whitespace": true,
    "unicode_normalize": true
  },

  "extraction_quality_score": {
    "weights": {
      "schema_validity": 0.15,
      "field_f1_partial": 0.70,
      "inverted_hallucination": 0.15
    }
  },

  "document_extraction_score": {
    "weights": {
      "reliable_f1_partial": 0.70,
      "schema_validity": 0.15,
      "inverted_hallucination": 0.15
    }
  },

  "field_diagnostics": {
    "enabled": true,
    "max_examples_per_field": 20,
    "max_fields_in_report": 100,
    "max_value_chars": 200
  }
}
```

Field-path patterns:
- Exact: `origen.domicilio`
- Array wildcard: `cronograma[].fecha` (matches any index)
- Glob wildcard: `metadata.*` (matches any suffix)

---

## Checklist: Setting Up a New Extraction Dataset

Before running a full evaluation, answer these questions:

**1. What is my ground truth source?**
- System-generated (database, structured form) → GT is reliable for all fields
- Manually typed by operators → GT may contain abbreviations, casing inconsistency, word-order variation for name/address fields

**2. Do I have mixed field types?**
- If the schema has both bounded fields (enum, integer, date) and freeform fields (name, address), configure `field_tiers`.
- Set `document_extraction_score.weights.reliable_f1_partial` as the headline driver.

**3. Are there any fields with near-0% accuracy in the first run?**
- Check `field_diagnostics`. Look at the predicted vs expected examples.
- If a boolean/enum shows >95% of samples with the same wrong prediction, check if the model is actually correct and the GT is wrong.
- Add confirmed GT-error fields to `ignored_fields`.

**4. Do name/address fields need normalization?**
- Accents inconsistent between GT and model output? Enable `normalization.unicode_normalize: true` (default) or `freeform_normalization.strip_accents: true`.
- GT uses abbreviations (`Av.`, `Gral.`, `B°`) that the model expands? Add to `freeform_normalization.abbreviations`.
- GT word order differs from model output for names? Enable `freeform_normalization.word_order_insensitive: true`.

**5. Do I have arrays where order doesn't matter?**
- Scheduling tables, line items, dates in arbitrary order? Add to `unordered_arrays` with `key_fields`.

**Recommended `metrics_config.json` template for document extraction with manual GT:**

```json
{
  "ignored_fields": [],

  "field_tiers": {
    "reliable": [],
    "freeform": []
  },

  "freeform_normalization": {
    "strip_accents": true,
    "word_order_insensitive": true,
    "abbreviations": {}
  },

  "document_extraction_score": {
    "weights": {
      "reliable_f1_partial": 0.70,
      "schema_validity": 0.15,
      "inverted_hallucination": 0.15
    }
  }
}
```

Fill in the field lists, run a smoke test with `--limit 10`, then check `field_diagnostics` to validate.
