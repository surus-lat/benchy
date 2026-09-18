---
name: write-adapter
description: Expose any AI-system to benchy — a model, a prompted node, a composed workflow, or a tool-using agent — through the one runtime contract. Use when someone wants to benchmark a system benchy has no built-in provider for.
---
# Write an adapter

benchy knows exactly one interface:

```
named-field input object  →  named-field output object
```

Everything about *how* your system runs lives on your side of that line. That is what
makes a raw model, a model with a prompt, a composed workflow and an agent the same kind
of thing to the engine.

## The whole contract

```python
def my_system(input_object: dict) -> dict:
    return {"invoice_number": "A-001", "total": 121.0}
```

That is a complete adapter. Sync or async both work, and an object with an `invoke`
method works too:

```python
class MySystem:
    async def invoke(self, input_object: dict) -> dict:
        ...
```

Use the class form when you have something to set up once (an HTTP client, a loaded
model, a subprocess) rather than per example.

```bash
benchy run benchmark.yaml --adapter path/to/system.py:my_system
```

Declare it in the benchmark as what is being evaluated:

```yaml
ai-system:
  type: external
  id: invoice-extractor-v7
```

## What you receive

The input object matches the program's input schema, already validated. Artifact fields
(`image`, `audio`, `document`) arrive as **resolved absolute paths** to existing files —
open them, upload them, base64 them, whatever your system needs.

## What you must return

An object matching the output schema exactly: every field, no extras, right types. The
engine validates strictly and does not coerce.

**Do not repair your system's output to make it pass.** If it returns `"121.00"` where
the contract says `float`, let it — that is a true `invalid_output`, and papering over it
makes the benchmark lie about the thing you are trying to measure.

If you cannot produce an output at all, raise. The engine records `execution_error` with
your message, which is more useful than a fabricated answer.

## Worth knowing

- **Return the raw text on a parse failure** rather than raising. The engine stores it as
  `prediction`, so you can see what the system actually said.
- **Retry transient transport failures** inside your adapter if you want to — a rate limit
  is not a quality signal and should not be scored as one. Do *not* retry by changing the
  request (a looser schema, a different model): that changes what is being measured.
- **One adapter, one AI-system.** A run is `R = (B, AI)`. Falling back to a second model
  inside your adapter scores a mixture under one name.

See `examples/invoices/system.py` for a complete, runnable one.
