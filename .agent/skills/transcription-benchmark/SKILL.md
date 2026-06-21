---
name: transcription-benchmark
description: >
  Use when the user wants to run the local multi-architecture transcription
  benchmark on FLEURS Latin-American Spanish + Brazilian Portuguese across
  every supported model — Whisper variants (including the Surus LATAM
  fine-tune), Voxtral, Qwen3-ASR, and Canary. All inference runs on the
  user's own hardware; no cloud APIs or paid endpoints. Covers
  single-model smokes, full-panel runs across the two-venv setup, and
  reading the comparative WER / CER table the runner produces. Triggers
  on: "run the transcription benchmark", "benchmark asr across models",
  "compare voxtral canary qwen whisper", "asr panel", "fleurs panel",
  "latam asr benchmark".
---

# Multi-architecture transcription benchmark (local)

End-to-end recipe for the canonical benchy ASR panel: every model below
runs locally on FLEURS es_419 (and optionally pt_br), printed as a single
comparative WER / CER table. Validated against this repo on 2026-06-20.

**Local-only by design.** No `whisper-1` cloud, no DashScope, no paid
endpoints. Everything below runs on the user's hardware so a colleague
can reproduce results offline once the venvs are built.

## The panel

Eight models across four architectures. Adapter is the one the YAML uses
to load the model.

| Config | HF repo | Adapter / interface | Venv |
|---|---|---|---|
| `whisper-tiny-transformers` | `openai/whisper-tiny` | legacy `transformers_audio` | `.venv` |
| `whisper-base-transformers` | `openai/whisper-base` | legacy `transformers_audio` | `.venv` |
| `whisper-small-transformers` | `openai/whisper-small` | legacy `transformers_audio` | `.venv` |
| `whisper-large-v3-transformers` | `openai/whisper-large-v3` | legacy `transformers_audio` | `.venv` (force CPU / fp16 on Mac) |
| `whisper-large-v3-turbo-transformers` | `openai/whisper-large-v3-turbo` | legacy `transformers_audio` | `.venv` |
| **`surus-whisper-large-v3-turbo-latam-transformers`** | **`surus-ai/whisper-large-v3-turbo-latam`** | legacy `transformers_audio` | `.venv` |
| `canary-1b-flash-transformers` | `nvidia/canary-1b-flash` | `canary_nemo` adapter | either |
| `qwen3-asr-0.6b-transformers` | `Qwen/Qwen3-ASR-0.6B` | `qwen3_asr_chat` adapter | `.venv` |
| `voxtral-mini-4b-transformers` | `mistralai/Voxtral-Mini-4B-Realtime-2602` | `voxtral_chat` adapter | **`.venv-vox`** |

The Surus model is a continued fine-tune of `openai/whisper-large-v3-turbo`
on Common Voice 17 Spanish, with European-Spain accents filtered out. Per
its model card, WER 7.80% on Common Voice ES (vs 15.44% base) and ~1.62x
faster inference. Same architecture as Whisper → loads through the
existing `transformers_audio` provider.

## Prerequisites — build both venvs once

```bash
bash scripts/setup-venvs.sh
# Or piecewise: setup-venvs.sh default | setup-venvs.sh vox
# Or via Makefile: make setup-venvs
```

Two venvs because Voxtral needs `transformers >=5.13` and `qwen-asr` pins
`transformers <5.0`. Each model YAML that needs a specific venv carries
`venv: <path>`; `benchy` pre-flight fails fast with the exact replacement
command if you launch from the wrong one.

No API keys, no `.env` setup needed — everything in this panel loads
weights locally from HuggingFace on first run, then caches at
`~/.cache/huggingface/`.

## Single-model smoke (sanity check before the panel)

Use these whenever you've changed a YAML or want to validate one model in
isolation. All run on 1 sample, exit-policy=smoke, with `--log-samples`
so you can sanity-check the actual transcription.

```bash
# Whisper family + Surus + Qwen + Canary — default venv:
make smoke-whisper          # whisper-tiny via .venv
make smoke-canary           # canary-1b-flash via .venv
make smoke-qwen             # qwen3-asr-0.6b via .venv

# Voxtral — dedicated venv:
make smoke-voxtral          # voxtral-mini-4b via .venv-vox

# Surus LATAM model (no Make target — direct):
.venv/bin/benchy eval -c surus-whisper-large-v3-turbo-latam-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 1 --log-samples \
  --run-id surus_smoke --exit-policy smoke
```

Pass criteria: exit 0, `Run status: passed`, `valid_samples == total_samples`,
and a recognizable Spanish prediction in the `*_samples.json`.

## The full panel

`scripts/run_asr_panel.py` iterates each `(model, locale)` pair through
`benchy eval` and harvests the per-run metrics. It picks the right venv
per model automatically (reads each YAML's `venv:` field).

```bash
# Default panel: every config in DEFAULT_MODELS × locales es + pt
.venv/bin/python scripts/run_asr_panel.py \
  --limit 25 \
  --locales es pt \
  --skip-failures \
  --run-id full_panel
```

`--skip-failures` lets one wedged model not block the others. Without it,
the first failure stops the panel.

To restrict to a subset:

```bash
.venv/bin/python scripts/run_asr_panel.py \
  --limit 25 \
  --locales es \
  --models whisper-large-v3-turbo-transformers \
           surus-whisper-large-v3-turbo-latam-transformers \
           canary-1b-flash-transformers \
           qwen3-asr-0.6b-transformers \
           voxtral-mini-4b-transformers \
  --skip-failures \
  --run-id latest_per_family
```

Output lands in `outputs/benchmark_outputs/<run_id>_LIMITED/<model>/` and
the runner writes a summary at `outputs/asr_panel_summary.json` plus
prints a comparative table to stdout.

## What good output looks like

5-sample smoke on FLEURS es_419 / pt_br, Apple Silicon, the venv setup
shipped with the repo. These are the reference numbers from this session
(real runs, recorded 2026-06-20). Use them as a sanity check on your own
install — small drift is fine, factor-of-2 differences mean something
broke.

| Config | Locale | WER | CER | Notes |
|---|---|---|---|---|
| `whisper-large-v3-turbo-transformers` | es_419 | 0.089 | 0.027 | Best Whisper-family on this sample |
| `surus-whisper-large-v3-turbo-latam-transformers` | es_419 | 0.138 | 0.024 | 1-sample — fine-tune shines on accents, not FLEURS |
| `canary-1b-flash-transformers` | es_419 | **0.069** | 0.024 | Best overall (n=1) |
| `voxtral-mini-4b-transformers` | es_419 | 0.103 | 0.018 | CPU on Mac |
| `qwen3-asr-0.6b-transformers` | es_419 | 0.138 | 0.036 | qwen-asr backend |

Healthy signals:
- WER < 0.30 for any well-supported model with a non-tiny size
- `valid_samples == total_samples` for every (model, locale) row
- Per-sample predictions in `*_samples.json` are recognizable Spanish /
  Portuguese
- Surus model output is grammatical Spanish — if it returns English or
  empty, the YAML lost `supports_language_kwarg: true`

Unhealthy signals:
- WER pinned at 1.0 with `error_rate > 0` — model didn't load, check
  the log under `logs/<run_id>/`
- Canary on pt_br — model only supports en/de/es/fr; the YAML pins
  `source_lang: es`, so a pt_br run will produce nonsense Spanish
- Voxtral hangs > 5 min on Mac — you're on `.venv` (transformers 4.57)
  not `.venv-vox`; pre-flight should have caught this. Check
  `.venv-vox/bin/python -c "import transformers; print(transformers.__version__)"`
  returns `5.13.x.dev0`

## When (model, locale) is unsupported

- **Canary × pt_br** — Canary-1b-flash doesn't support Portuguese. Drop
  pt_br from `--locales` or just live with the high-WER row.
- **Qwen3-ASR × any** — supports 30 languages incl. es and pt. No issue.
- **Voxtral × any** — supports en/fr/es/de/ru/zh/ja/it. pt_br runs but
  quality varies.

## Files this skill touches

| Path | Role |
|---|---|
| `configs/models/whisper-*-transformers.yaml` | Whisper variant configs (5 of them) |
| `configs/models/surus-whisper-large-v3-turbo-latam-transformers.yaml` | Surus LATAM fine-tune |
| `configs/models/{canary-1b-flash,qwen3-asr-0.6b,voxtral-mini-4b}-transformers.yaml` | Adapter-routed models |
| `scripts/run_asr_panel.py` | Panel runner — venv-aware via `_benchy_bin_for_config` |
| `scripts/setup-venvs.sh` | Builds both venvs |
| `Makefile` | Per-model `smoke-*` shortcuts + `setup-venvs` |
| `src/interfaces/transformers_audio_interface.py` | Legacy interface used by every local Whisper config |
| `src/adapters/{canary_nemo,qwen3_asr_chat,voxtral_chat}.py` | The three adapters |
| `src/tasks/transcription/fleurs_{es_latam,pt_br}.py` | Task definitions |

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: model 'mistralai/Voxtral-...' requires venv .venv-vox` | Launched from `.venv` | Use `.venv-vox/bin/benchy` (the error itself prints the right command) |
| `voxtral_realtime not recognized` | `transformers 4.57` in `.venv-vox` | Rebuild: `bash scripts/setup-venvs.sh vox` |
| Predictions empty for Qwen3-ASR | YAML missing `language:` and auto-detect mis-fired | Pin `language: Spanish` in the qwen3_asr_chat block |
| Surus model outputs English | `supports_language_kwarg: true` missing | Restore that line in `surus-whisper-large-v3-turbo-latam-transformers.yaml` |
| Canary `fastconformer` rejected | NeMo not installed | `bash scripts/setup-venvs.sh default` (installs `nemo-toolkit[asr]`) |
| Whisper-large-v3 hangs on Mac | MPS+fp32 wedge | Set `torch_dtype: float16` in YAML, or use `large-v3-turbo` |
| Panel hangs forever on Voxtral | Voxtral on CPU on a 16 GB Mac, 25 samples × 2 locales = 16+ hours | Drop `--locales pt`, drop `--limit` to 5, or run Voxtral standalone |

## Related skills

- [`whisper-benchmark`](../whisper-benchmark/SKILL.md) — Whisper-only walk-through with MPS gotchas
- [`qwen3-asr-howto`](../qwen3-asr-howto/SKILL.md) — Qwen3-ASR specifics (DashScope cloud, vLLM backend)
- [`evaluate`](../evaluate/SKILL.md) — Lower-level `benchy eval` flag reference
