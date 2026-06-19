---
name: read-results
description: Translate benchmark results into plain English for non-developer users. Read run_outcome.json and run_summary.json and produce a human-readable summary with the 2-3 worst-performing samples and one concrete next step.
---
# Read Results Skill

Translates benchmark results into plain English.

This is a **communication tool**, not a failure-diagnosis tool.
- Use `interpret-run` (developer skill) for deep failure diagnosis and debugging.
- Use this skill to explain results to someone who built an AI system and wants to know how it did.

---

## Input: Where to Find Results

```
outputs/benchmark_outputs/<run_id>/<model_name>/
├── run_outcome.json       ← overall status and counts
├── run_summary.json       ← per-task metric summary
└── <task>/
    └── <subtask>/
        ├── *_metrics.json     ← field-level scores
        └── *_samples.json     ← individual predictions
```

If no `run_id` is given, find the most recent run in `outputs/benchmark_outputs/`.

---

## What to Communicate

### 1. Overall result (one sentence)

Translate `run_outcome.status`:
- `passed` → "Your benchmark completed successfully."
- `degraded` → "Your benchmark completed with some issues — most examples worked but some had problems."
- `failed` → "Your benchmark encountered errors and could not complete."

### 2. Score in plain language

For extraction/structured tasks, translate field-level scores:
> "Your system extracted **vendor_name** correctly **89% of the time**.
>  The weakest field was **amount** (61% correct)."

For classification:
> "Your system classified correctly **74% of the time**."

For freeform/QA:
> "Your system matched expected answers **68% of the time** (F1 score: 0.72)."

### 3. Worst-performing samples (2-3 examples)

Read `*_samples.json`. Find the 2-3 samples with the lowest score. For each, show:
```
Input:    [the actual input text]
Expected: [what the correct answer was]
Got:      [what the AI actually returned]
```

Keep this concrete and readable. Truncate long inputs to ~200 characters.

### 4. One concrete next step

Based on the results, give ONE actionable suggestion:

| Pattern | Suggestion |
|---------|-----------|
| Score > 90% | "Excellent results. Consider testing on more diverse examples." |
| Score 70–90% | "Good results with room to improve. The weakest field is [X] — try adding clearer instructions in the system prompt." |
| Score 50–70% | "Moderate results. The biggest issue is [X]. Consider improving your prompt or checking if the model understands the task." |
| Score < 50% | "The system is struggling. Check that the API is returning the right fields and that the expected outputs in your dataset are correct." |
| `connectivity_error` | "The API wasn't reachable. Check the URL and any required authentication." |
| `no_samples` | "No test data was found. Check the data path in your benchmark spec." |

---

## Example Output

```
Results: Invoice Extraction Benchmark
─────────────────────────────���────────

Overall: Completed successfully (89% average score)

Field breakdown:
  ✓ vendor_name   — 94% correct
  ✓ date          — 91% correct
  △ currency      — 78% correct
  ✗ amount        — 61% correct

Weakest examples:
  1. Input: "Factura N° 0001-00045678 - DISTRIBUIDORA OMEGA..."
     Expected amount: 12500.00
     Got:             "12.500,00"  ← number formatting issue

  2. Input: "Invoice for services rendered in March..."
     Expected amount: 875.5
     Got:             null         ← field was missing from response

Next step:
  The amount field has formatting issues (commas vs periods) and missing values.
  Try adding this to your system prompt: "Always return amounts as decimal numbers
  without thousand separators (e.g., 12500.00, not 12.500,00)."
```

---

## What NOT to Do

- Do not show raw JSON, stack traces, or internal field names like `diagnostic_class`
- Do not mention handler classes, task types, or benchy internals
- Do not recommend code changes unless the user is a developer

---

## Output Artifact

Plain-English summary — no files written.
