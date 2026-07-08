# Current Benchy → New Benchy — Salvage Audit

**Companion to:** `2026-06-22-new-benchy-design.md`.
**Purpose:** walk the current tree area by area, judge each against the
vision, label it **KEEP / ADAPT / NUKE** with a one-line reason.
**Rule of thumb:** if a piece exists only because we glued
lm-evaluation-harness–shaped frameworks together, it does not survive.

## Summary counts

- ~18k lines under `src/` today.
- ~2k lines survive as-is or lightly reshaped (`KEEP` + `ADAPT`).
- ~16k lines are path-dependent baggage (`NUKE`).

The ratio is not a mistake. The old code solved *running benchmarks
against many models on many providers on many frameworks*. The vision
says benchy is not that project.

## src/tasks/common/

| File | Lines | Verdict | Reason |
|---|---|---|---|
| `metrics.py` | 327 | **KEEP** | `ExactMatch`, `F1Score`, `WordErrorRate`, `CharErrorRate`, `Pearson`, `MSE`, `MultipleChoiceAccuracy` become the atomic primitives of `benchy.scoring.primitives`. |
| `image_metrics.py` | 268 | **KEEP** | Mask/IoU utilities become primitives for image tasks. |
| `dataset_loaders.py` | 211 | **ADAPT** | `CachedDatasetMixin` becomes cache logic inside `benchy.data`. Drop the mixin shape. |
| `dataset_adapters.py` | 870 | **ADAPT** | Extract the source→sample loading paths (HF, JSONL, CSV, TSV) into `benchy.data`. Drop the schema-driven task-config wiring — the new Task owns schema. |
| `base.py` (BaseHandler) | 405 | **NUKE** | Handler god-object; the four modules replace it. Nothing to port. |
| `freeform.py` | 356 | **NUKE** | Rubric baked into a subclass; expressed via `benchy.scoring` now. |
| `structured.py` | 503 | **NUKE** | Rubric baked into a subclass; the useful part (`MetricsCalculator`) lives elsewhere and survives. |
| `multiple_choice.py` | 585 | **NUKE** | Same. |
| `multimodal_structured.py` | 526 | **NUKE** | Same. |
| `multimodal_image_artifact.py` | 128 | **NUKE** | Same. |
| `task_config_schema.py` | 313 | **NUKE** | Task shape now comes from `benchy.task`, not a YAML schema registry. |
| `config_generator.py` | 229 | **NUKE** | Generator for a config format we are dropping. |
| `visualization.py` | 204 | **KEEP** | Small; nice to have for `benchmark.run()` reports. Land under `benchy.benchmark` or `benchy.report`. |
| `utils/dataset_utils.py` | ~200 | **KEEP** | HF download + JSONL cache primitives, still needed by `benchy.data`. |
| `utils/choice_utils.py` | ~150 | **ADAPT** | Choice parsing helpers become primitives for `multiple_choice` scoring. |
| `utils/structured_metrics_calculator.py` | ~500 | **KEEP** | This IS the `field_wise` scorer with partial credit. Port under `benchy.scoring.structural`. |
| `utils/partial_matching.py` | ~300 | **KEEP** | The nuts-and-bolts of partial matching for structured extraction. Same destination. |
| `utils/text_utils.py` | ~100 | **KEEP** | Small text normalization helpers. |

## src/tasks/ (task groups)

Task groups mix "task type" and "language" (e.g. `portuguese/`,
`spanish/`, `classify/`, `transcription/`). This is inconsistent with
the vision's `/<task>/<domain>/<language>` ontology.

| Group | Verdict | Reason |
|---|---|---|
| `transcription/` | **ADAPT** | The FLEURS *data* survives as `benchmarks/transcription/fleurs/{es-419,pt-BR}/`. The Handler code doesn't. |
| `structured_extraction/`, `document_extraction/`, `image_extraction/` | **ADAPT** | Data survives under `benchmarks/`; new benchmarks re-authored via the four peers. |
| `image_manipulation/` (remove_background) | **ADAPT** | Data + scoring metric survive; task re-authored. |
| `classify/`, `portuguese/`, `spanish/`, `structured/`, `translation/`, `audit/`, `_template_handler/` | **NUKE** | Mix of legacy benchmarks glued from third-party frameworks. Any specific benchmark someone still needs can be re-authored in the new shape as a small YAML. |
| `registry.py`, `group_runner.py` | **NUKE** | Old discovery mechanism; replaced by the ontology filesystem layout + `benchy.registry`. |

## src/interfaces/ + src/adapters/

Both become `benchy.system`. The user's own recent adapter work
already telegraphed this direction; the vision names the endpoint.

| File | Verdict | Reason |
|---|---|---|
| `interfaces/openai_interface.py` (1005) | **ADAPT** | Extract only the parts that make OpenAI-compatible endpoints a `System` — request shape, response parsing. Drop the interface layer's task-facing hooks. |
| `interfaces/openai_audio_interface.py` | **ADAPT** | Same; folds into an audio-shaped `System`. |
| `interfaces/transformers_audio_interface.py` | **ADAPT** | Becomes the default `System` for HF audio pipelines. |
| `interfaces/generic_api_interface.py`, `http_interface.py`, `_template.py` | **NUKE** | Old provider-shaped abstraction; the new `System` protocol makes them redundant. |
| `adapters/base.py` | **NUKE** | Replaced by `benchy.system.System`. |
| `adapters/voxtral_chat.py` | **KEEP** | Real, working per-family inference code. Renames from Adapter to System; same body. |
| `adapters/qwen3_asr_chat.py` | **KEEP** | Same. |
| `adapters/canary_nemo.py` | **KEEP** | Same, once NeMo is unblocked. |

## src/engine/

The engine layer is where the old handler+interface contract lives.
Most of it stops making sense once Handler dies.

| File | Verdict | Reason |
|---|---|---|
| `engine/protocols.py` | **NUKE** | Old task+interface protocols. Replaced by `benchy.task`, `benchy.system`, `benchy.scoring` contracts. |
| `engine/benchmark_runner.py` (888) | **NUKE** | Rewritten smaller as `benchy.benchmark.run()`. |
| `engine/connection.py` (525) | **NUKE** | vLLM/provider connection glue; replaced by `System.run`. |
| `engine/checkpoint.py`, `retry.py`, `output_diagnostics.py` | **ADAPT** | Retry semantics and per-sample checkpointing are still useful; port slimmed-down versions into `benchy.benchmark` when a milestone actually needs them. Do not port speculatively. |

## src/inference/

| File | Verdict | Reason |
|---|---|---|
| `inference/vllm_config.py`, `vllm_server.py` | **NUKE** | vLLM is one transport. If a user needs vLLM, they wire it as an `endpoint:` `System`. Not core. |
| `inference/venv_manager.py` | **NUKE** | Per-model venv orchestration was a workaround; the new `System` implementations declare their own dependency envelope. |

## src/probe/

`probe/runner.py` alone is 1631 lines. It's a system-inspection tool
from the pre-vision era.

| File | Verdict | Reason |
|---|---|---|
| `probe/*` | **NUKE** | If a "does this system respond correctly?" utility is needed later, it's a ~50-line `benchy system probe <url>` command, not 1900 lines of orchestration. |

## src/leaderboard/

This is the latamboard publishing pipeline — a downstream project, not
part of authoring benchmarks.

| File | Verdict | Reason |
|---|---|---|
| `leaderboard/*` | **NUKE** (from benchy) | Split to `latamboard-publish` or similar. Benchmarks produced by new benchy feed it via a stable JSON contract. |

## Top-level src/*.py

| File | Verdict | Reason |
|---|---|---|
| `benchy_cli.py`, `benchy_cli_eval.py`, `benchy_cli_probe.py` | **NUKE** | Replaced by a much smaller `benchy.cli` (four verbs: `new`, `run`, `list`, `export-loss`). |
| `pipeline.py` (1056) | **NUKE** | Prefect flow orchestrating the old handler+interface world. No survivors. |
| `config_manager.py`, `config_loader.py` | **NUKE** | Old YAML schema. `benchmark.yaml` in the new tree is 4 fields deep, no manager. |
| `gpu_config.py`, `generation_config.py` | **NUKE** | Tied to vLLM path. |
| `run_id_manager.py`, `signal_utils.py`, `task_completion_checker.py`, `outcome.py`, `logging_utils.py`, `prefect_compat.py` | **NUKE** | Plumbing for the old pipeline. If any of these turns out to be needed, port back in <50 lines when a milestone actually demands it. |

## configs/

| Path | Verdict | Reason |
|---|---|---|
| `configs/models/*.yaml` (55 files) | **NUKE** | System URLs replace per-model YAMLs. If a model has a bespoke loader, it's a `System` implementation, not a config. |
| `configs/providers/*.yaml` | **NUKE** | Same — providers become URL schemes. |
| `configs/systems/*.yaml` | **ADAPT** | These are closest in spirit to the new `benchy.system` — port intent, not schema. |
| `configs/templates/*.yaml`, `configs/tests/`, `configs/config.yaml` | **NUKE** | Templates + registry for the old config format. |

## Ancillary

| Path | Verdict | Reason |
|---|---|---|
| `.plans/*` (16 files) | **NUKE** | Legacy scratch. Anything load-bearing has already been distilled into VISION.md. |
| `.notes/`, `.claude/` | **KEEP as-is** | Local dev metadata. |
| `docs/` | **KEEP** | Design docs live here, including this file. |
| `misc/`, `reference/`, `scripts/` | **AUDIT LATER** | Not on the critical path for the redesign; revisit per file when we get there. |
| `tests/` | **NUKE + REWRITE** | Old tests target Handler/Interface contracts. New tests target the four modules; write them alongside milestones. |
| `Makefile`, `setup.sh` | **ADAPT** | Trim to the verbs new benchy actually needs. |
| `pyproject.toml` | **ADAPT** | Rewrite `[project.scripts]` and dependencies against the new module tree. |
| `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `README.md` | **ADAPT** | Content is stale for the redesign; rewrite once the four modules exist. |

## What this leaves us with

Roughly:

- ~2k lines to port under the new tree (scoring primitives + partial
  matching + system adapters + dataset caching + FLEURS assets).
- ~500 new lines to write for the four modules + `Benchmark`
  composition + CLI.
- Everything else deleted.

The nuke step is deferred behind a user-approval gate in the execution
plan. Nothing is deleted before the reference benchmark passes on the
new tree.
