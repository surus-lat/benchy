# Architecture Rationale

This document explains *why* Benchy is designed the way it is — the problems each
design decision solves and what was traded off to get there. For *what* each component
does, see `docs/architecture.md`.

## The core problem: evaluating systems, not just models

Most evaluation frameworks are model-centric. They assume you send a prompt, get text
back, and score it. Benchy was built for a different reality: AI products often consist
of entire pipelines — an OCR stage, a structure extraction stage, a classification
stage — that need to be evaluated as systems, not as raw language models.

This meant two requirements from day one:

1. **The same task must run against both a raw model and a task-specific pipeline.**
   A Spanish classification task should produce comparable accuracy numbers whether
   you run it against `gpt-4o-mini` or a Surus-hosted classifier endpoint.

2. **Adding a new AI system to evaluate must not require touching evaluation logic.**
   The person who runs the benchmark should not need to understand how a specific
   API's authentication or payload format works.

These requirements drove the **task/interface decoupling** that defines Benchy's
architecture.

---

## Why tasks and interfaces are separate

The first design decision: tasks define *what to evaluate* (data, prompts, metrics),
and interfaces define *how to send requests* (authentication, payload format, response
parsing).

**The problem this solves:** Before this split, adding a new provider required editing
task code. Task authors had to think about API specifics (token limits, JSON response
wrapping, multimodal content formatting). Provider authors had to understand task-specific
scoring.

**The trade-off:** A compatibility layer (`InterfaceCapabilities` + `TaskCapabilityRequirements`)
is needed to match tasks with the interfaces that can run them. This is a small runtime
overhead for a large reduction in coupling. A task that requires `supports_multimodal`
is simply skipped on text-only providers with a clear log message — no conditional
logic inside the task itself.

**What this means in practice:** Adding a new cloud provider (e.g., a new Together AI
model) requires only a YAML config file, no Python. Adding a new task requires only
Python code in the task directory, no interface changes.

---

## Why the handler system exists

The original task system required 200–400 lines of Python per task: data loading,
prompt building, response parsing, metric calculation, capability checking. A new
task author had to understand and replicate all of it.

**The problem this solves:** A multiple-choice classification task on a Spanish dataset
is structurally identical to one on a Portuguese dataset. The prompts differ; the
mechanics do not. Repeating the mechanics for every task is boilerplate that introduces
bugs when the mechanics change (e.g., a new token limit parameter name).

**The solution:** Four base handlers cover the common patterns:
- `MultipleChoiceHandler` — for classification tasks with discrete labels
- `StructuredHandler` — for JSON extraction tasks with schema validation
- `FreeformHandler` — for open-ended text generation with similarity metrics
- `MultimodalStructuredHandler` — for vision-language extraction tasks

Each handler provides default implementations for data loading, prompt formatting,
metrics, and capability checking. A task author overrides only what differs. Most tasks
need 20–50 lines of code instead of 300.

**The trade-off:** Handlers constrain task shape. A task that doesn't fit any handler
pattern (e.g., a ranking task or a multi-turn dialogue task) requires more work. This
is intentional — the handlers represent the patterns that occur most often. Unusual task
shapes should be rare enough to justify the additional work.

---

## Why convention-based discovery instead of a registry

Tasks are discovered by scanning the `src/tasks/` directory for Python files. A file
`src/tasks/spanish/copa_es.py` is automatically registered as the `spanish.copa_es`
task. No registration call. No import statement in a central registry file.

**The problem this solves:** A central registry creates a merge conflict bottleneck.
When two contributors add tasks simultaneously, both must edit the same file. With
convention-based discovery, they work in separate directories and never conflict.

**The trade-off:** The task's Python class name must match the file's snake-to-PascalCase
conversion (`copa_es.py` → `CopaEs`). This is a naming constraint, not a flexibility
loss. The convention is documented in `docs/contribute_tasks.md`.

---

## Why config files drive everything

Benchy is config-driven: a YAML file selects the model, provider, tasks, and parameters.
You can run a full benchmark without writing Python.

**The problem this solves:** Evaluation reproducibility. If a benchmark run is controlled
by CLI flags that aren't recorded, you can't reproduce it. YAML config files are
committed to the repo, so every run is traceable to a config state.

**The trade-off:** The config format has to cover a wide range of use cases, which makes
it slightly more complex than a pure CLI would be. The `configs/templates/` directory
addresses this by providing documented starting points.

**The providerless CLI exception:** You *can* run without a config file by passing
`--provider`, `--model-name`, and `--base-url` on the CLI. This is intentionally
supported for quick experimentation. The `--save-config` flag lets you capture a
successful CLI invocation as a config file.

---

## Why the probe system exists

Before running an evaluation, Benchy probes the model to detect its actual capabilities.

**The problem this solves:** OpenAI's newer models (`gpt-5`, `o1`, `o3`, `o4 series`)
require `max_completion_tokens` instead of `max_tokens`. If you send the wrong parameter
name, the API silently ignores the limit and you get unpredictable output lengths. The
probe detects which parameter name is required and which structured output format
actually works, so evaluation logic doesn't need per-model conditional branches.

**The trade-off:** Probing adds 30–60 seconds to every run. This is acceptable for
runs that evaluate thousands of samples. For very quick smoke tests (`--limit 2`),
it's a significant fraction of the runtime. Future work could cache probe results.

**Why not use provider documentation instead:** Provider documentation lags behind
actual API behavior and differs between API versions. Probing the real API at runtime
is more reliable than maintaining a static capability table.

---

## Why checkpoint/resume exists

Long evaluation runs (100+ samples, 10+ tasks, multi-GPU local models) can fail
mid-run due to network errors, GPU OOMs, or process kills. Benchy writes a
`task_status.json` per completed subtask. On re-run with the same `--run-id`, completed
tasks are skipped.

**The problem this solves:** Without checkpointing, a 6-hour evaluation run that fails
at hour 5 requires starting over. With checkpointing, re-running with the same
`--run-id` picks up from where it stopped.

**The trade-off:** The checkpoint file only records completion at the subtask level, not
at the sample level. A partially-completed subtask is rerun from the beginning.
Fine-grained sample-level checkpointing would be more efficient but would also make the
checkpoint file significantly more complex to manage.

---

## Why `run_outcome.json` is separate from metrics

Benchy writes two top-level output files:
- `run_outcome.json` — status, exit code, per-task pass/fail, errors, git context
- `run_summary.json` — compact per-task metric values

**The problem this solves:** Automation systems (CI, monitoring, leaderboard pipelines)
need to know whether a run passed or failed before parsing metrics. Combining status
and metrics into one large file would require automation to parse the full file to
extract a single boolean.

**The trade-off:** Two files to maintain consistency between. Benchy writes both
atomically after the pipeline completes, so they're always in sync on successful runs.

---

## Why zero-code CLI evaluation exists

Users can evaluate any HuggingFace dataset or local Parquet file without writing a task:

```bash
benchy eval --task-type structured --dataset-name my-org/invoices \
  --provider openai --model-name gpt-4o-mini --limit 5
```

**The problem this solves:** Not every dataset needs a permanent task definition.
Quick experiments, new dataset validation, one-off evals — all of these would otherwise
require writing a task file and waiting for a code review.

**The trade-off:** Zero-code tasks have less control than full handler-based tasks. You
can't customize the scoring logic, the prompt structure beyond templates, or the data
preprocessing pipeline. For production benchmarks that need full control, the handler
system is still the right path.

---

## Related docs

- [Architecture](architecture.md) — What each component does
- [Handler System Guide](HANDLER_SYSTEM_GUIDE.md) — How the handler system works
- [Contributing Tasks](contribute_tasks.md) — How to add new tasks
- [Contributing Providers](contributing_providers.md) — How to add new providers
