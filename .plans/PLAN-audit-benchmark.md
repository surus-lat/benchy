# Audit AI Benchmark — Implementation Plan

## Overview

Two handler-based benchmarks for the 2-stage medical audit pipeline described in `audit-ai-algorithms.md`.

**Status:** Handlers ready, test data in place, pending full dataset.

## Architecture

```
src/tasks/audit/
├── metadata.yaml           ← task group declaration
├── __init__.py             ← class exports
├── policy_eval.py          ← Algorithm 1: Policy Evaluation
└── recommendation.py       ← Algorithm 2: Recommendation Synthesis

.data/audit/
├── policy_eval.jsonl       ← 2 test samples
└── recommendation.jsonl    ← 2 test samples
```

Both use `StructuredHandler` with strict enum-based exact-match scoring, local JSONL data, and batched prompts matching the real auditor code.

## Handler Details

### Algorithm 1 — Policy Evaluation (`audit.policy_eval`)

- **Inherits:** `StructuredHandler`
- **Schema:** Flattened 7-field object — one field per policy ID, each a 4-value enum
  (`compliant`, `non_compliant`, `insufficient_data`, `does_not_apply`)
- **Scoring:** 7 possible points, exact match only, EQS weighted 100% on field F1
- **Data:** `.data/audit/policy_eval.jsonl`
- **Prompt:** Batched policy prompt (matching `auditor/policies.json` template)

### Algorithm 2 — Recommendation Synthesis (`audit.recommendation`)

- **Inherits:** `StructuredHandler`
- **Schema:** 2-field object — `recommendation` (required, 4-value enum) +
  `downgrade_to` (optional, 4-value enum or null)
- **Scoring:** 2 possible points, exact match only
- **Data:** `.data/audit/recommendation.jsonl`
- **Prompt:** Business rules system prompt (matching `_RECOMMENDATION_SYSTEM_PROMPT`)

## Config Registration

```yaml
# configs/config.yaml
task_groups:
  audit:
    description: "Medical transfer audit pipeline benchmarks"
    tasks:
      - "audit.policy_eval"
      - "audit.recommendation"
```

## Usage

```bash
# Smoke test
benchy eval --config configs/models/openai_gpt-4o.yaml \
  --task-group audit --limit 2 --exit-policy smoke

# Single task
benchy eval --config configs/models/openai_gpt-4o.yaml \
  --tasks audit.policy_eval --limit 5 --exit-policy smoke

# Full run
benchy eval --config configs/models/openai_gpt-4o.yaml \
  --task-group audit --exit-policy strict
```

## Data Format

### policy_eval.jsonl

```jsonl
{"id": "0", "text": "<serialized audit_input JSON string>",
 "expected": {"salud_mental": "compliant", "geriatrico": "compliant", ...}}
```

The `text` field is the full audit input document (30+ fields from `backend/schemas/audit_input.json`), serialized as a JSON string. The `expected` field is a flat object with 7 policy IDs mapped to their status values.

### recommendation.jsonl

```jsonl
{"id": "0", "text": "<serialized {transfer_data, policy_results} JSON string>",
 "expected": {"recommendation": "reject", "downgrade_to": null}}
```

The `text` field contains both `transfer_data` (subset of audit input fields) and `policy_results` (output of Algorithm 1). The `expected` field has only `recommendation` and `downgrade_to`.

## Test Samples

| File | Sample | Scenario | Expected |
|------|--------|----------|----------|
| policy_eval.jsonl | 0 | Mendoza, amb sin medico, routine checkup | ugl_mendoza: non_compliant, rest does_not_apply |
| policy_eval.jsonl | 1 | Cordoba, mental health internacion | salud_mental: non_compliant, rest does_not_apply |
| recommendation.jsonl | 0 | ugl_mendoza non_compliant, distance 12.5km | recommend: reject, downgrade_to: null |
| recommendation.jsonl | 1 | All compliant, Salta, distance 45km | recommend: accept, downgrade_to: null |

## Key Decisions

- **No compiler/benchmark.yaml** — developer handler path for full control over prompts and scoring
- **Batched policy eval** — one sample = one LLM call evaluating all 7 policies (matches real system)
- **Flat output schemas** — avoids nested JSON complexity, one field per scored dimension
- **Exact match scoring** — no partial credit for string fields (enums are discrete categories)
- **Local data only** — no HuggingFace dependency