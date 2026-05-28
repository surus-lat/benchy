# Tutorial: Your First Benchy Evaluation

By the end of this tutorial you'll have installed Benchy, configured an API key, and
run a real benchmark evaluation that produces a scored results folder. The whole thing
takes about 10 minutes.

## What you'll need

- Python 3.12 or newer
- An OpenAI API key (or any OpenAI-compatible provider key)
- Internet access (to pull HuggingFace datasets)

## Step 1: Install

Clone the repo and run the setup script:

```bash
git clone https://github.com/surus-lat/benchy.git
cd benchy
bash setup.sh
source .venv/bin/activate
```

The setup script creates a `.venv` virtual environment and installs all cloud-provider
dependencies. You'll see it prompt about an optional dataset download — press Enter to
skip it for now.

Verify the install worked:

```bash
benchy --help
```

You should see the Benchy CLI help text. If `benchy` isn't on your PATH, use
`python -m src.benchy_cli` instead throughout this tutorial.

## Step 2: Set your API key

Copy the environment template and fill in your key:

```bash
cp env.example .env
```

Open `.env` and set `OPENAI_API_KEY`:

```bash
OPENAI_API_KEY=sk-...your-key-here...
```

Benchy reads this file automatically at startup. You can also export it in your shell
instead — either works.

## Step 3: Run your first evaluation

Run a Spanish-language benchmark with a 5-sample limit to keep it fast:

```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks spanish --limit 5
```

You'll see live progress output like this:

```
INFO - Starting benchmark run run_20260528_143012
INFO - Probing model capabilities...
INFO - Probe complete: schema=structured_outputs, multimodal=false
INFO - Running task: spanish (10 subtasks)
INFO - [1/10] copa_es: 5 samples
INFO - [1/10] copa_es: accuracy=0.80
...
INFO - Run complete. Results: outputs/benchmark_outputs/run_20260528_143012/gpt-4o-mini/
```

The probe step (30–60 seconds) detects the model's actual capabilities so Benchy
configures requests correctly. This happens once per run.

## Step 4: Inspect the results

Open the output directory:

```bash
ls outputs/benchmark_outputs/run_20260528_143012/gpt-4o-mini/
```

You'll see folders for each task group and two top-level files:

```
run_outcome.json     # Machine-readable status and metrics summary
run_summary.json     # Compact per-task metric summary
spanish/             # Per-subtask results
  copa_es/
    metrics.json
    samples.jsonl
    task_status.json
  ...
```

Read the run summary:

```bash
cat outputs/benchmark_outputs/run_20260528_143012/gpt-4o-mini/run_summary.json
```

Each task shows its primary metric (accuracy for classification tasks), sample count,
and error rate. A successful run has `"status": "passed"` in `run_outcome.json`.

## What you built

You ran a real benchmark evaluation against the OpenAI API across multiple Spanish-language
tasks (COPA, ESCOLA, MGSM, OpenBookQA, PAWS, and others). Benchy automatically:

1. Probed the model to detect capability support
2. Downloaded the HuggingFace datasets
3. Built prompts, sent requests, and scored responses
4. Wrote reproducible results to a dated run folder

## Next steps

- **Evaluate more tasks**: Add `portuguese`, `translation`, or `structured_extraction`
  to the `--tasks` flag: `--tasks spanish portuguese translation`
- **Run without a limit**: Remove `--limit 5` for a full evaluation
- **Save a config**: Use `--save-config configs/models/my-gpt4o.yaml` to save your
  CLI setup for reuse
- **Try a different provider**: Replace `--provider openai` with `--provider together`
  and set `TOGETHER_API_KEY` in `.env`
- **Evaluate a local model**: See the vLLM section in `docs/evaluating_models.md`

## Troubleshooting

**`benchy: command not found`** — The `.venv/bin` is not on your PATH. Run
`source .venv/bin/activate` first, or use `python -m src.benchy_cli eval ...`.

**`AuthenticationError`** — Your `OPENAI_API_KEY` is missing or wrong. Check your `.env`
file and that you ran `source .venv/bin/activate` after editing it.

**A task is skipped** — The log will say `Skipping task: capability mismatch`. The model
doesn't support a required capability (for example, a multimodal task on a text-only
model). Run only text tasks with `--tasks spanish`.

**Slow first run** — The first run downloads HuggingFace datasets to the local cache.
Subsequent runs are faster.

## Related docs

- [Evaluating Models](evaluating_models.md) — Full options for running evaluations
- [CLI Reference](reference-cli.md) — Every CLI flag explained
- [Task Catalog](reference-tasks.md) — All available tasks and what they measure
