---
name: synthesize-data
description: Generate synthetic benchmark examples from a task spec. Invoked from setup-data when there is no data, or directly when the user asks to generate examples. Produces a JSONL file in .data/<benchmark-name>/.
---
# Synthesize Data Skill

Generates synthetic `{text, expected}` pairs from `the benchmark spec`'s task spec.

Can be invoked from `setup-data` (when Path 3 is selected) or directly:
> "Generate 30 examples for my invoice benchmark"

---

## Hard Rule: Image Tasks

If `task.input.type` is `image`, `document`, or `pdf` — **stop immediately**:

> "I can generate expected outputs but not the images themselves. You need real images to run this benchmark. Use setup-data (Path 1 or 2) to supply your own images."

Do not attempt generation. Return to the user.

---

## Steps

1. **Read** the benchmark spec (must already have a valid `task:` section)
2. **Check** input type — block image tasks (see above)
3. **Run** the data generator:

```python
from src.data_generator import generate_data

path = generate_data("<path-to-spec>", count=30)
# → .data/<benchmark_name>/train.jsonl
```

Or via CLI (if running as a subprocess):
```bash
python -c "from src.data_generator import generate_data; generate_data('<path-to-spec>', count=30)"
```

4. **Update** the benchmark spec's `data:` section to point to the generated file:
```yaml
data:
  source: local
  path: .data/<benchmark-name>/train.jsonl
```

5. **Confirm** success:
```
Generated 30 examples → .data/my-benchmark/train.jsonl
```

---

## What the Generator Does

- Reads `task.type`, `task.input.description`, `task.output.fields` from `the benchmark spec`
- Builds a generation prompt from the spec
- Calls the generator model N times in parallel (async)
- Validates each result against the field schema
- Saves to `.data/<benchmark-name>/train.jsonl`

The generator model is an internal engine detail — never mention it to the user.

---

## Counting

- Default count: 30
- User can specify: "Generate 50 examples"
- Minimum useful: 10 (fewer makes metrics unreliable)

---

## Output Artifact

`.data/<benchmark-name>/train.jsonl` — one JSON object per line:
```json
{"id": "0", "text": "Invoice from ...", "expected": {"vendor_name": "ACME Corp", "amount": 1250.00}}
{"id": "1", "text": "Factura de ...", "expected": {"vendor_name": "Distribuidora XYZ", "amount": 875.50}}
```

---

## After Generation

Tell the user:
> "Generated {N} examples → .data/{benchmark-name}/train.jsonl. Your the benchmark spec has been updated. Run `run-benchmark` to evaluate."

If count < 10, warn:
> "Note: {N} examples is a small dataset — results may not be statistically reliable. Consider generating more."
