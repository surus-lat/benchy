---
name: whisper-benchmark
description: >
  Run a Whisper-family transcription benchmark on a Mac, end-to-end and locally.
  Use when the user asks to benchmark/evaluate/compare ASR or speech-to-text
  models on FLEURS (Latin American Spanish, Brazilian Portuguese) using the
  `transformers_audio` provider in this repo. Covers setup, the model panel,
  Mac-specific MPS gotchas (the large-v3 wedge), result interpretation, and
  extension. Triggers on: "benchmark whisper", "run ASR panel", "compare STT
  models locally", "FLEURS transcription on Mac".
---

# Whisper-family transcription benchmark on a Mac

Run any subset of the Whisper family (tiny → base → small → large-v3-turbo →
large-v3) against FLEURS es_419 / pt_br locally on a Mac, with no HTTP server
and no API keys. The benchmark exercises the `transformers_audio` provider,
the `TranscriptionHandler` task, and the WER/CER/exact-match aggregator.

---

## When to use this

- "Benchmark Whisper on Spanish/Portuguese audio."
- "Compare tiny vs large-v3 on FLEURS."
- "Run an ASR smoke test locally — no OpenAI key."
- The user references `transformers_audio`, `whisper-*-transformers`,
  `run_asr_panel.py`, or `transcription.fleurs_*`.

**Don't use** for cloud whisper-1 (use `-c whisper-1` with `openai_audio` and
`OPENAI_API_KEY`), for non-FLEURS datasets (write a new subtask first), or for
non-Whisper transformers ASR (Voxtral / Qwen3-ASR / Canary need `trust_remote_code`
and may not work via the standard pipeline auto-dispatch — see *Extending*).

---

## One-time setup

```bash
# From repo root. Pulls torch ~700 MB + transformers + librosa.
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e '.[transcription]'
```

Verify:
```bash
.venv/bin/python -c "import torch, transformers, librosa; print(torch.__version__, transformers.__version__)"
.venv/bin/python -m pytest tests/test_transformers_audio_interface.py -q
```

---

## Quick single-model run

The smallest valid invocation — 3 samples of es_419 against whisper-small:

```bash
.venv/bin/benchy eval \
  -c whisper-small-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 3 \
  --log-samples \
  --exit-policy smoke
```

Important argument shapes:

- **Subtask name is `transcription.fleurs_es_latam` (group.subtask)**, NOT
  `fleurs_es_latam`. Plain `fleurs_es_latam` raises
  *"Task 'fleurs_es_latam' is not handler-based"*.
- `-c whisper-small-transformers` resolves to
  `configs/models/whisper-small-transformers.yaml`. Don't include the path or
  extension.
- First run downloads FLEURS audio to `.data/transcription/<locale>/` and the
  model weights to `~/.cache/huggingface/`. Subsequent runs reuse both.

Output lands in `outputs/benchmark_outputs/<RUN_ID>/<short_model_name>/`. The
key files:

- `transcription/<subtask>/*_metrics.json` — aggregated WER/CER/exact_match.
- `transcription/<subtask>/*_samples.json` — per-sample predictions + refs.
- `run_outcome.json` — overall pass/fail summary.

**Metrics live nested**: `data["metrics"]["wer"]`, not `data["wer"]`. Watch for
this when scripting result extraction.

---

## Run the full panel

```bash
.venv/bin/python scripts/run_asr_panel.py \
  --limit 5 \
  --locales es pt \
  --models whisper-tiny-transformers whisper-base-transformers \
           whisper-small-transformers whisper-large-v3-turbo-transformers \
  --skip-failures
```

The runner iterates `(model × locale)` pairs through `benchy eval`, harvests
each metrics.json, prints a comparative table, and writes
`outputs/asr_panel_summary.json`. `--skip-failures` keeps it moving when one
model wedges (see next section).

---

## Mac-specific gotcha: large-v3 wedges on MPS+FP32

**Symptom**: `whisper-large-v3` (the 1.55B model) hangs indefinitely. The
benchy process shows ~6% CPU, ~15 MB RSS, and a ~400 GB virtual size in
`ps aux`. SIGTERM is ignored (process is in state `U`, uninterruptible
syscall); only `kill -9` unblocks it. The smaller Whisper models (tiny / base
/ small / large-v3-turbo) run fine on the same machine.

**Cause**: the `transformers_audio` provider defaults to `device: auto`,
which on Apple Silicon resolves to `mps`. Whisper-large-v3 in `float32` on
MPS hits an unsupported / pathological op in the chunked attention path and
stalls in I/O wait.

**Fixes** (pick one):

1. **Force CPU for large-v3 only** — accurate but slow (~30 s/sample CPU).
   Override the model YAML's provider knobs:

   ```yaml
   # configs/models/whisper-large-v3-transformers.yaml
   transformers_audio:
     provider_config: transformers_audio
     supports_language_kwarg: true
     device: cpu
   ```

2. **Use float16 on MPS** — MPS handles fp16 well and large-v3 runs at
   reasonable speed (~3–8 s/sample on M-series):

   ```yaml
   transformers_audio:
     provider_config: transformers_audio
     supports_language_kwarg: true
     torch_dtype: float16
   ```

3. **Skip large-v3, prefer large-v3-turbo** — `large-v3-turbo` (809M, ~8×
   faster) is the practical Mac ceiling and gives WER within ~0.01 of
   large-v3 on FLEURS at a fraction of the runtime. This is the default
   recommendation for Mac CPU and MPS.

If you see the wedge live: `kill -9 <pid>` the eval, the panel runner
proceeds to the next model under `--skip-failures`.

---

## What good output looks like

5-sample smoke on FLEURS es_419 (Apple Silicon, CPU/MPS mix):

| Model | Device | WER | CER | Throughput |
|---|---|---|---|---|
| whisper-tiny | mps | 0.18 | 0.05 | 0.53 sps |
| whisper-small | mps | 0.14 | 0.03 | 0.26 sps |
| whisper-large-v3-turbo | mps | 0.11 | 0.02 | 0.17 sps |

Healthy signals to verify:
- WER < 0.30 for any well-supported language with a non-tiny model.
- `valid_samples == total_samples` (no API/load errors).
- Per-sample predictions in `*_samples.json` are recognizable Spanish/Portuguese.

Unhealthy signals:
- WER pinned at 1.0 with `error_rate > 0` → model didn't load or all samples
  failed. Read the log under `logs/<RUN_ID>/`.
- All predictions empty / look like one language regardless of locale →
  `supports_language_kwarg` is set wrong for the model (Whisper models
  REQUIRE it `true` to get the language hint; non-Whisper need it `false`).

---

## Extending the panel

**Add another Whisper variant** — drop a YAML following
`configs/models/whisper-small-transformers.yaml`. The pattern:

```yaml
model:
  name: <hf_repo_id>          # e.g. distil-whisper/distil-large-v3
transformers_audio:
  provider_config: transformers_audio
  supports_language_kwarg: true     # Whisper-family only
task_defaults:
  log_samples: true
tasks:
- transcription
metadata:
  provider: huggingface
  model_type: whisper
```

Then pass the YAML name to `--models` in `run_asr_panel.py`.

**Add a non-Whisper transformers ASR model** (wav2vec2, MMS) — same shape but
set `supports_language_kwarg: false` so the interface doesn't pass Whisper-only
`generate_kwargs`.

**Non-Whisper architectures (Voxtral, Qwen3-ASR, Canary)** now have dedicated
adapters under `src/adapters/`. As of 2026-06-20 (real-model end-to-end smoke):

- `voxtral_chat` (adapter): **PASS** on `transformers >= 5.13` + `mistral-common`.
  Real run: Voxtral-Mini-4B fp16 CPU on FLEURS es_419 → **WER 0.103, CER 0.018**,
  recognizable Spanish transcription. The model loads through
  `AutoModelForSpeechSeq2Seq` with `trust_remote_code=True`, and the processor
  takes a raw audio array (librosa-loaded at 16 kHz mono). Mac MPS wedges
  the same way it does for `whisper-large-v3` — force `device: cpu` in the
  YAML on <=16 GB Macs (already set as default for the shipped config).
- `qwen3_asr_chat` (adapter): **BLOCKED upstream — different root cause**.
  Qwen3-ASR's config declares architecture `Qwen3ASRForConditionalGeneration`,
  which doesn't exist in any transformers release, and the HF repo
  (`Qwen/Qwen3-ASR-0.6B`) ships only weights + tokenizer config — no
  `modeling_qwen3_asr.py` for `trust_remote_code` to consume. Genuine
  upstream gap. Adapter is wired and tested with mocks; will work the
  moment transformers ships the architecture (or the repo adds custom code).
- `canary_nemo` (adapter): **PASS** on `nemo-toolkit[asr] >= 2.7`.
  Real run: nvidia/canary-1b-flash CPU on FLEURS es_419 → **WER 0.069, CER 0.024**
  (best so far on the sample). Loads via NeMo's
  `EncDecMultiTaskModel.from_pretrained` and calls
  `model.transcribe(audio=[path], source_lang='es', target_lang='es',
  pnc='yes')`. NeMo ships a colliding `tests/` package at site-packages
  root — kept under control by `tests/__init__.py` + pytest
  `pythonpath = ["."]`.
  Canary-1b-flash supports en/de/es/fr only; pt_br will produce nonsense.

Use the adapter path by setting `adapter: <name>` in the model YAML and
adding a block named after the adapter for its config knobs. See
`docs/superpowers/specs/2026-06-19-adapter-layer-design.md` for the layer
design and `docs/superpowers/plans/2026-06-19-adapter-layer-implementation.md`
for the empirical wiring story.

### Voxtral install — use the dedicated `.venv-vox`

Voxtral and Qwen3-ASR have incompatible `transformers` pins, so benchy
ships a two-venv setup:

```bash
bash scripts/setup-venvs.sh     # builds .venv (default) AND .venv-vox
# Then run Voxtral specifically from .venv-vox:
.venv-vox/bin/benchy eval -c voxtral-mini-4b-transformers ...
```

The Voxtral YAML declares `venv: .venv-vox` and benchy's CLI pre-flight
check will refuse a Voxtral run launched from the wrong venv. See the
`qwen3-asr-howto` skill for the full conflict matrix.

### Canary install

```bash
# Adds ~2 GB of NeMo deps. Re-install transformers from main afterward
# because NeMo pins an older transformers and will downgrade it.
VIRTUAL_ENV=$(pwd)/.venv uv pip install 'nemo-toolkit[asr]'
VIRTUAL_ENV=$(pwd)/.venv uv pip install --upgrade \
  'git+https://github.com/huggingface/transformers.git'
```

Verified non-regression after both installs: 394 unit tests green;
`whisper-tiny-transformers` smoke still passes through the legacy provider
path; Voxtral + Canary both produce real Spanish transcriptions.

**Add a new dataset** — the FLEURS subtasks at
`src/tasks/transcription/fleurs_es_latam.py` are the template. A new subtask
needs to:
1. Inherit from `TranscriptionHandler`.
2. Set `language` (ISO 639-1) and `locale`.
3. Implement `load()` to populate `self._samples` with dicts containing
   `id`, `audio_path`, `expected`, `language`, `locale`. Cache audio to disk
   with `save_audio_array` so subsequent runs are fast.
4. Register the file in `src/tasks/transcription/__init__.py` and the subtask
   in `src/tasks/transcription/metadata.yaml`.

---

## Verifying the wiring without weights

You don't need to download any model to verify a config change wires up:

```bash
.venv/bin/python -c "
from src.config_manager import ConfigManager
cfg = ConfigManager().load_model_config('configs/models/whisper-small-transformers.yaml')
print('provider_type:', cfg['provider_type'])
print('keys:', sorted(cfg['transformers_audio'].keys()))
"
```

If `provider_type` is `vllm` instead of `transformers_audio`, the model YAML
is missing the `transformers_audio:` block or `ConfigManager.load_model_config`
is missing a branch for it.

---

## Files this skill touches

- Provider: `configs/providers/transformers_audio.yaml`
- Models: `configs/models/whisper-{tiny,base,small,large-v3,large-v3-turbo}-transformers.yaml`
- Runner: `scripts/run_asr_panel.py`
- Interface: `src/interfaces/transformers_audio_interface.py`
- Handler: `src/tasks/transcription/_transcription_handler.py`
- Subtasks: `src/tasks/transcription/fleurs_{es_latam,pt_br}.py`
- Metrics: `WordErrorRate` / `CharErrorRate` in `src/tasks/common/metrics.py`
