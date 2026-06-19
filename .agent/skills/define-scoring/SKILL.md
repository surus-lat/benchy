---
name: define-scoring
description: Help the user choose how to grade the AI's output and write the scoring: section of benchmark.yaml. Use after define-task. This is Stage 2 of the second-layer workflow.
---
# Define Scoring Skill

Helps the user choose a scoring method and writes the `scoring:` section of `benchmark.yaml`.

This is **Stage 2** of 4. Requires `task:` section to already exist.

---

## Three Options in Plain English

Present these choices. No math required from the user.

| What to say | scoring.type | When it makes sense |
|-------------|-------------|---------------------|
| "Each correct field earns a point (score = correct / total)" | `per_field` | Extraction tasks with multiple fields |
| "All fields must be correct to earn a point (pass/fail)" | `binary` | When partial credit doesn't make sense |
| "Close enough counts (fuzzy match)" | `semantic` | QA, translation, summaries |

For **classification tasks** (`task.output.type: label`), always use `binary` (exact label match). Don't ask — just note it.

---

## Follow-Up Questions

### For `per_field`:
- "Should small numeric differences count as correct? (e.g., 100.00 vs 100.01)" → `numeric_tolerance: 0.01`
- "Should capitalization matter?" → `case_sensitive: true/false`

### For `binary`:
- "Should capitalization matter?" → `case_sensitive: true/false`

### For `semantic`:
No follow-up params needed. F1 score handles variation automatically.

---

## Internal Mapping (never show to user)

| scoring.type | Benchy internal |
|-------------|-----------------|
| `per_field` | `StructuredHandler` + field-level F1 / EQS metrics |
| `binary` | `ExactMatch` |
| `semantic` | `FreeformHandler` + `F1Score` + `ExactMatch` |
| `custom` | user-provided Python function (advanced — only if explicitly asked) |

---

## Output: scoring: section

```yaml
# Per-field (extraction)
scoring:
  type: per_field
  partial_credit: true
  case_sensitive: false
  numeric_tolerance: 0.01     # optional

# Binary (pass/fail)
scoring:
  type: binary
  case_sensitive: false

# Fuzzy (semantic)
scoring:
  type: semantic
```

---

## Writing benchmark.yaml

Read the existing benchmark spec (e.g., `benchmarks/my-benchmark.yaml`). Update or add only the `scoring:` key. Leave all other sections unchanged.

---

## Next Step

After writing the `scoring:` section, tell the user:

> Scoring section written. Next: run `configure-model` to specify which AI system to evaluate, and `setup-data` to provide the test examples. You can run either one first.
