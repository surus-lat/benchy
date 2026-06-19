---
name: define-task
description: Collect what the AI system does and write the task: section of benchmark.yaml. Use when a user describes their AI in plain English and needs help defining a benchmark task. This is Stage 1 of the second-layer workflow.
---
# Define Task Skill

Collects the task definition from the user and writes the `task:` section of `benchmark.yaml`.

This is **Stage 1** of 4. After completing this skill, run `define-scoring`.

---

## What to Ask

Ask in plain English. Never mention handler classes, task types, or benchy internals.

**Question 1 — What does your AI do?**
"Describe what your AI system does in one or two sentences."

**Question 2 — What goes in?**
- Text (a document, question, or passage)
- Image or PDF
- Text + image

**Question 3 — What comes out?**
- Extracts specific fields (vendor name, amount, date…)
- Classifies into one of several categories
- Answers a question in free text
- Translates text
- Something else

**Question 4 — If extraction:** list the fields.
```
field_name: description (one per line)
vendor_name: Name of the seller
amount: Total invoice amount
date: Invoice date in YYYY-MM-DD format
```

**Question 4 — If classification:** list the possible labels.
```
positive
negative
neutral
```

---

## Internal Mapping (never show to user)

| User says | task.type | output.type |
|-----------|-----------|-------------|
| Extracts fields | extraction | structured |
| Classifies | classification | label |
| Answers questions | qa | text |
| Translates | translation | text |
| Free text output | freeform | text |
| Image or PDF input | (any above) | + input.type: image |

Image input + extraction → `structured` task with `multimodal_input: true` in dataset_config.
This is NOT a separate handler — it is the `structured` type with image enabled.

---

## Output: task: section

```yaml
# Extraction example
task:
  type: extraction
  input:
    type: text                        # text | image | document
    description: "Argentine invoice PDF"
  output:
    type: structured
    fields:
      - name: vendor_name
        type: string
        description: Name of the seller
        required: true
      - name: amount
        type: number
        description: Total invoice amount
        required: true
      - name: date
        type: string
        format: YYYY-MM-DD
        required: false
```

```yaml
# Classification example
task:
  type: classification
  input:
    type: text
    description: "Customer review"
  output:
    type: label
    labels:
      - positive
      - negative
      - neutral
```

```yaml
# QA / Translation / Freeform
task:
  type: qa           # or translation, freeform
  input:
    type: text
    description: "Question about a document"
  output:
    type: text
```

---

## Writing benchmark.yaml

If a benchmark spec already exists (e.g., `benchmarks/my-benchmark.yaml` or `benchmark.yaml` at the root), read it and update only the `task:` section. If it does not exist, create it at `benchmarks/<name>.yaml` (or `benchmark.yaml` for a single-benchmark project) with only the `task:` and `name:` / `description:` fields populated.

```yaml
benchmark:
  name: my-benchmark
  description: One-line description of what the benchmark tests

  task:
    # ... content from above
```

Leave `scoring:`, `data:`, and `target:` empty or absent — the other skills fill them in.

---

## Next Step

After writing the `task:` section, tell the user:

> Task section written. Next: run `define-scoring` to choose how to grade the output.
