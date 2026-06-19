# How to run a transcription benchmark in benchy

Benchy now supports ASR (transcription) benchmarks against two provider
architectures:

| Provider | Where the model runs | Use when |
|---|---|---|
| `openai_audio` | OpenAI's cloud Whisper (`whisper-1`) | You have an `OPENAI_API_KEY` and want a strong baseline |
| `transformers_audio` | In-process HuggingFace pipeline (CPU / MPS / CUDA) | You want to run any open Whisper / wav2vec2 / Voxtral / Qwen3-ASR / Canary locally — no API key, no HTTP server |

Both use the same `transcription.fleurs_*` tasks and produce the same
WER / CER / exact-match metrics. You can mix them in one run.

**Verified working** on 2026-06-19, Apple Silicon, this repo at `031c039`:

| Config | Provider | WER | CER | Wall time (2 samples) |
|---|---|---|---|---|
| `whisper-tiny-transformers` | `transformers_audio` | 0.265 | 0.085 | 10.2 s |
| `whisper-1` | `openai_audio` | 0.091 | 0.015 | ~12 s |

---

## Setup (one-time)

```bash
# From repo root.
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e '.[transcription]'
# or, without uv:
# pip install -e '.[transcription]'
```

This pulls torch (~700 MB), transformers, librosa, jiwer, soundfile, accelerate.

Verify:

```bash
.venv/bin/python -c "import torch, transformers, librosa, jiwer; print('ok')"
.venv/bin/python -m pytest tests/test_transformers_audio_interface.py -q
```

For the cloud architecture also: `export OPENAI_API_KEY=...` (or put it in
`.env` at repo root — benchy loads it).

---

## Quick smoke (local, no API key)

3 samples of FLEURS Latin-American Spanish against `whisper-tiny` running
locally:

```bash
.venv/bin/benchy eval \
  -c whisper-tiny-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 3 \
  --log-samples \
  --run-id local_smoke \
  --exit-policy smoke
```

Notes:

- **Subtask name is `transcription.fleurs_es_latam` (group.subtask)**, not
  plain `fleurs_es_latam` (which raises *"Task is not handler-based"*).
- `-c whisper-tiny-transformers` resolves to
  `configs/models/whisper-tiny-transformers.yaml`. Don't include the path
  or extension.
- First run downloads FLEURS audio to `.data/transcription/<locale>/` and
  the model weights to `~/.cache/huggingface/`. Subsequent runs reuse both.

Output lands in `outputs/benchmark_outputs/<RUN_ID>/<short_model_name>/`:

- `transcription/<subtask>/*_metrics.json` — aggregated WER / CER / exact_match
- `transcription/<subtask>/*_samples.json` — per-sample predictions + references
- `run_outcome.json` — overall pass/fail summary

**Metrics live nested**: `data["metrics"]["wer"]`, not `data["wer"]`.

---

## Quick smoke (cloud)

Same task, against OpenAI's `whisper-1` — for an apples-to-apples baseline:

```bash
.venv/bin/benchy eval \
  -c whisper-1 \
  --tasks transcription.fleurs_es_latam \
  --limit 3 \
  --log-samples \
  --run-id cloud_smoke \
  --exit-policy smoke
```

Requires `OPENAI_API_KEY`. Costs are tiny (FLEURS samples are ~10 s each;
the OpenAI audio rate is ~$0.006/min).

---

## Compare a panel of models

`scripts/run_asr_panel.py` runs every `(model × locale)` pair through
`benchy eval`, harvests each metrics file, and prints a comparative table.

```bash
.venv/bin/python scripts/run_asr_panel.py \
  --limit 5 \
  --locales es pt \
  --models whisper-tiny-transformers whisper-base-transformers \
           whisper-small-transformers whisper-large-v3-turbo-transformers \
  --skip-failures
```

Output: `outputs/asr_panel_summary.json`. `--skip-failures` keeps the
runner moving if one model wedges (see Mac gotcha below).

Mix architectures in one panel by including `whisper-1` (cloud) alongside
the local `*-transformers` configs.

---

## Mac-specific gotcha: large-v3 wedges on MPS+FP32

**Symptom**: `whisper-large-v3` (1.55B) hangs indefinitely on Apple Silicon.
Process shows ~6% CPU, ~15 MB RSS, and a ~400 GB virtual size in `ps aux`.
SIGTERM is ignored; only `kill -9` unblocks it. Smaller Whisper variants
(tiny / base / small / large-v3-turbo) are fine.

**Cause**: `transformers_audio` defaults to `device: auto` → `mps` on Apple
Silicon. Whisper-large-v3 in `float32` on MPS hits an unsupported op in
chunked attention and stalls.

**Fixes** (pick one):

1. **Force CPU for large-v3 only** — accurate but slow (~30 s/sample):

   ```yaml
   # configs/models/whisper-large-v3-transformers.yaml
   transformers_audio:
     provider_config: transformers_audio
     supports_language_kwarg: true
     device: cpu
   ```

2. **Use float16 on MPS** — MPS handles fp16 well; ~3-8 s/sample:

   ```yaml
   transformers_audio:
     provider_config: transformers_audio
     supports_language_kwarg: true
     torch_dtype: float16
   ```

3. **Use `large-v3-turbo` instead** — 809M, ~8× faster, WER within ~0.01
   of large-v3 on FLEURS. Recommended Mac ceiling.

---

## Interpreting results

Healthy:
- WER < 0.30 for any well-supported language with a non-tiny model.
- `valid_samples == total_samples` (no API/load errors).
- Per-sample predictions in `*_samples.json` look like sensible Spanish /
  Portuguese.

Unhealthy:
- WER pinned at 1.0 with `error_rate > 0` — model didn't load or every
  sample failed. Read the log under `logs/<RUN_ID>/`.
- All predictions look like one language regardless of locale —
  `supports_language_kwarg` is set wrong (Whisper-family REQUIRES it
  `true` to receive the language hint; non-Whisper models need it `false`).

---

## Adding a model

**Another Whisper variant** — drop a YAML following
`configs/models/whisper-small-transformers.yaml`:

```yaml
model:
  name: <hf_repo_id>            # e.g. distil-whisper/distil-large-v3
transformers_audio:
  provider_config: transformers_audio
  supports_language_kwarg: true # Whisper-family only
task_defaults:
  log_samples: true
tasks:
- transcription
metadata:
  provider: huggingface
  model_type: whisper
```

Then pass the YAML name to `--models` in `run_asr_panel.py`.

**Non-Whisper transformers ASR** (wav2vec2, MMS) — same shape but
`supports_language_kwarg: false`. Custom-code models (Qwen3-ASR, Voxtral,
Canary) additionally need `trust_remote_code: true`; these often don't
auto-dispatch through `pipeline("automatic-speech-recognition", ...)` and
may need a per-family interface. The repo has starter YAMLs for these but
smoke runs revealed caveats — treat them as work-in-progress.

---

## File map

What this skill touches in the repo:

| Path | Role |
|---|---|
| `configs/providers/transformers_audio.yaml` | Provider config |
| `configs/providers/openai_audio.yaml` | Cloud provider config |
| `configs/models/whisper-*-transformers.yaml` | Local Whisper variants |
| `configs/models/whisper-1.yaml` | Cloud Whisper |
| `src/interfaces/transformers_audio_interface.py` | Local provider impl |
| `src/interfaces/openai_audio_interface.py` | Cloud provider impl |
| `src/tasks/transcription/_transcription_handler.py` | Shared handler |
| `src/tasks/transcription/fleurs_{es_latam,pt_br}.py` | FLEURS subtasks |
| `src/tasks/common/metrics.py` | `WordErrorRate`, `CharErrorRate` |
| `scripts/run_asr_panel.py` | Multi-model panel runner |

The agent-facing version of this doc lives at
`.agent/skills/whisper-benchmark/SKILL.md`. The two are kept in sync.
