# Translation Task Metrics Update

## Summary

Updated the translation evaluation tasks to include **BLEU** metric alongside **chrF**, and ensured proper aggregate scoring across all translation tasks.

## Changes Made

### 1. Main Translation Group (`translation.yaml`)
**Location**: `external/lm-evaluation-harness/lm_eval/tasks/translation/translation.yaml`

**Change**: Added BLEU to aggregate metrics
```yaml
aggregate_metric_list:
  - metric: chrf
    aggregation: mean
    weight_by_size: false
  - metric: bleu         # ← NEW
    aggregation: mean
    weight_by_size: false
```

### 2. FLORES+ Bidirectional Group (`flores_plus_bidirectional.yaml`)
**Location**: `external/lm-evaluation-harness/lm_eval/tasks/translation/flores_plus_latam_v3/flores_plus_bidirectional.yaml`

**Change**: Added BLEU to aggregate metrics for all 15 language pairs
```yaml
aggregate_metric_list:
  - metric: chrf
    aggregation: mean
    weight_by_size: false
  - metric: bleu         # ← NEW
    aggregation: mean
    weight_by_size: false
```

### 3. Individual Language Pair Tasks (15 files)
**Location**: `external/lm-evaluation-harness/lm_eval/tasks/translation/flores_plus_latam_v3/flores_*.yaml`

**Change**: Each individual task now computes both metrics
```yaml
metric_list:
  - metric: chrf
    aggregation: chrf
    higher_is_better: true
  - metric: bleu         # ← NEW
    aggregation: bleu
    higher_is_better: true
```

**Updated files**:
- `flores_por_spa.yaml` (Portuguese ↔ Spanish)
- `flores_eng_spa.yaml` (English ↔ Spanish)
- `flores_fra_spa.yaml` (French ↔ Spanish)
- `flores_ita_spa.yaml` (Italian ↔ Spanish)
- `flores_deu_spa.yaml` (German ↔ Spanish)
- `flores_hin_spa.yaml` (Hindi ↔ Spanish)
- `flores_cmn_spa.yaml` (Chinese ↔ Spanish)
- `flores_arb_spa.yaml` (Arabic ↔ Spanish)
- `flores_eng_por.yaml` (English ↔ Portuguese)
- `flores_fra_por.yaml` (French ↔ Portuguese)
- `flores_ita_por.yaml` (Italian ↔ Portuguese)
- `flores_deu_por.yaml` (German ↔ Portuguese)
- `flores_hin_por.yaml` (Hindi ↔ Portuguese)
- `flores_cmn_por.yaml` (Chinese ↔ Portuguese)
- `flores_arb_por.yaml` (Arabic ↔ Portuguese)

### 4. OPUS-100 Tasks (`_opus_100_common.yaml`)
**Location**: `external/lm-evaluation-harness/lm_eval/tasks/translation/opus_100/_opus_100_common.yaml`

**Change**: Added BLEU metric
```yaml
metric_list:
  - metric: chrf
    aggregation: chrf
    higher_is_better: true
  - metric: bleu         # ← NEW
    aggregation: bleu
    higher_is_better: true
```

## Expected Results Structure

### Before (chrF only)
```json
{
  "results": {
    "translation": {
      "alias": "translation",
      "chrf,none": 45.23
    },
    "flores_por_spa": {
      "alias": "  - spa↔por",
      "chrf,clean_translation": 50.13
    }
  }
}
```

### After (chrF + BLEU)
```json
{
  "results": {
    "translation": {
      "alias": "translation",
      "chrf,none": 45.23,
      "bleu,none": 28.45,         // ← NEW: Overall average
      "chrf_stderr,none": 2.15,
      "bleu_stderr,none": 1.82    // ← NEW
    },
    "flores_por_spa": {
      "alias": "  - spa↔por",
      "chrf,clean_translation": 50.13,
      "bleu,clean_translation": 32.87,    // ← NEW: Per-task
      "chrf_stderr,clean_translation": 3.27,
      "bleu_stderr,clean_translation": 2.14  // ← NEW
    }
  }
}
```

## Comparison with Spanish Task

### Spanish Task Structure
```json
{
  "results": {
    "latam_es": {
      "acc,none": 0.3556,          // ← Overall aggregate
      "alias": "latam_es"
    },
    "spanish": {
      "acc,none": 0.3556,          // ← Group aggregate  
      "alias": " - spanish"
    },
    "copa_es": {
      "acc,none": 0.4,             // ← Individual task
      "alias": "  - copa_es"
    }
  }
}
```

### Translation Task Structure (After Update)
```json
{
  "results": {
    "translation": {
      "chrf,none": 45.23,          // ← Overall aggregate chrF
      "bleu,none": 28.45,          // ← Overall aggregate BLEU
      "alias": "translation"
    },
    "flores_plus_bidirectional": {
      "chrf,none": 46.12,          // ← Group aggregate chrF
      "bleu,none": 29.34,          // ← Group aggregate BLEU
      "alias": " - flores_bidirectional"
    },
    "flores_por_spa": {
      "chrf,clean_translation": 50.13,  // ← Individual task chrF
      "bleu,clean_translation": 32.87,  // ← Individual task BLEU
      "alias": "  - spa↔por"
    }
  }
}
```

## Benefits

1. **BLEU as Backup Metric**: Provides traditional n-gram based evaluation alongside character-level chrF
2. **Comprehensive Evaluation**: Both metrics help identify different aspects of translation quality
3. **Aggregate Scoring**: Like Spanish tasks, translation now has clear overall performance scores
4. **Hierarchical Results**: 
   - Top level: `translation` group (all tasks combined)
   - Middle level: `flores_plus_bidirectional` (15 language pairs) + `opus` (2 tasks)
   - Bottom level: Individual language pairs (e.g., `flores_por_spa`)

## Metric Interpretation

### chrF (Character-level F-score)
- **Range**: 0-100 (higher is better)
- **Strengths**: Better for morphologically rich languages, more lenient with word order
- **Use**: Primary metric for translation quality

### BLEU (Bilingual Evaluation Understudy)
- **Range**: 0-100 (higher is better)  
- **Strengths**: Industry standard, good for comparing models
- **Use**: Backup metric, useful for cross-study comparisons

## Testing

To verify the changes work:

```bash
# Test with a single language pair
cd /home/mauro/dev/benchy
python eval.py -c configs/single_card/Hunyuan-MT-7B.yaml --limit 5

# Check results file includes both metrics
cat outputs/benchmark_outputs/Hunyuan-MT-7B/translation/*/results_*.json | grep -E "chrf|bleu"
```

Expected output should show both `chrf` and `bleu` metrics at all levels.

