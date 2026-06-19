---
name: validate
description: Pre-flight check for a benchmark spec before running. Runs benchy validate, reports errors in plain English, and guides the user to the right skill to fix each one. Use after the three definition stages and before run-benchmark.
---
# Validate Skill

Checks that a benchmark spec is complete and ready to run. Use this after finishing the definition stages (define-task, define-scoring, configure-model/setup-data) and before run-benchmark.

---

## When to Use

- User has finished defining a benchmark and wants to check it before running
- User hit an error during `benchy eval` and wants to diagnose the spec
- Routine step between definition and execution

---

## How to Validate

```bash
benchy validate                                           # auto-discovers if one spec exists
benchy validate --benchmark benchmarks/my-benchmark.yaml  # explicit path
```

If the project has multiple specs and no `--benchmark` was specified, list the candidates and ask the user which one to validate.

---

## Interpreting Results

### Clean

```
✓ benchmarks/my-benchmark.yaml is valid and ready to run.
```

Tell the user: "Your benchmark spec is valid. You're ready to run a smoke test."

Suggest next step:
```bash
benchy eval --benchmark <path> --limit 5 --exit-policy smoke
```

### Errors found

```
✗ benchmarks/my-benchmark.yaml has N error(s):
  • <error 1>
  • <error 2>
```

For each error, translate it into plain English and point to the skill that fixes it:

| Error pattern | Plain English | Fix with |
|---|---|---|
| `task.output.fields` required | "You haven't defined what fields to extract." | `define-task` |
| `task.type` invalid | "The task type isn't recognized." | `define-task` |
| `target.url` required | "Your API endpoint URL is missing." | `configure-model` |
| `target.model` required | "No model has been specified." | `configure-model` |
| `data.path` not found | "The data file doesn't exist at the path given." | `setup-data` or `synthesize-data` |
| `data.source` invalid | "The data source type isn't recognized." | `setup-data` |
| `scoring.type` invalid | "The scoring method isn't recognized." | `define-scoring` |

Tell the user which skill to use for each issue. Do not attempt to fix the spec directly unless the user asks — guide them to the right skill.

---

## What NOT to Do

- Do not show raw stack traces or internal field names
- Do not attempt to fix errors silently — always explain what was wrong
- Do not proceed to run-benchmark if validation fails
