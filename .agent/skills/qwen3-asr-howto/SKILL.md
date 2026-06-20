---
name: qwen3-asr-howto
description: >
  Use when the user wants to run Qwen3-ASR (any of the Qwen3-ASR-0.6B,
  Qwen3-ASR-1.7B, or Qwen3-ForcedAligner-0.6B models) locally or via the
  DashScope cloud API. Covers the right install (NOT transformers — the
  qwen-asr PyPI package), local Python usage, benchy integration via the
  qwen3_asr_chat adapter, the DashScope cloud alternative, the vLLM
  backend, and the version conflict with Voxtral. Triggers on: "run
  qwen3 asr", "qwen3-asr inference", "transcribe with qwen", "qwen-asr
  package", "qwen3 asr benchmark".
---

# Qwen3-ASR — how to use it

Qwen3-ASR is the Qwen team's family of multilingual ASR models. The HF
repos (`Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`) declare
`Qwen3ASRForConditionalGeneration` as their architecture, which **does not
exist in any `transformers` release**. The right runtime is the team's
own `qwen-asr` PyPI package — not `transformers` directly.

This skill captures what works as of 2026-06-20, verified end-to-end on
this benchy install.

## Released models

| HF repo | Size | What it does | Languages |
|---|---|---|---|
| `Qwen/Qwen3-ASR-0.6B` | ~600M | Fast ASR + language ID | 30 languages + 22 Chinese dialects |
| `Qwen/Qwen3-ASR-1.7B` | ~1.7B | State-of-the-art open-source ASR | Same set |
| `Qwen/Qwen3-ForcedAligner-0.6B` | ~600M | Word/char-level timestamps | 11 languages |

Supported languages (30): Chinese, English, Cantonese, Arabic, German,
French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai,
Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish,
Finnish, Polish, Czech, Filipino, Persian, Greek, Hungarian, Macedonian,
Romanian. Plus 22 Chinese regional dialects.

## Install

```bash
# Fresh isolated env (recommended — qwen-asr pins transformers<5.0,
# which conflicts with Voxtral's transformers>=5.13 requirement).
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr

# Minimal install (transformers backend):
pip install -U qwen-asr

# Optional: vLLM backend for faster inference + streaming
pip install -U 'qwen-asr[vllm]'

# Optional: FlashAttention 2 (CUDA only, fp16/bf16 only — big speedup on
# long inputs and large batches). Slow on <96GB RAM machines:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

For the benchy install path, see "Using from benchy" below — the
`[transcription]` extras don't currently pull in `qwen-asr` because of
the Voxtral conflict; install it separately into the same venv (or use
a sibling venv).

## Minimal Python usage

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",          # or -1.7B
    dtype=torch.bfloat16,            # bfloat16 recommended; fp16 also OK
    device_map="auto",               # "cuda:0", "cpu", "mps", etc.
    max_inference_batch_size=32,
    max_new_tokens=256,              # bump for long audio
)

results = model.transcribe(
    audio="path/to/clip.wav",        # see "Audio input shapes" below
    language=None,                   # None = auto-detect; "Spanish" to force
)

print(results[0].language)
print(results[0].text)
```

### Audio input shapes

`model.transcribe(audio=...)` accepts:
- Local path: `"/abs/path.wav"`
- URL: `"https://example.com/clip.wav"`
- Base64 data
- `(numpy.ndarray, sampling_rate)` tuple
- A list of any of the above for batched inference

Batch example:
```python
results = model.transcribe(
    audio=["clip1.wav", "clip2.wav"],
    language=["Spanish", "English"],   # one per clip, or None for auto
)
```

### With timestamps (forced alignment)

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
)

results = model.transcribe(
    audio="clip.wav",
    language="Spanish",
    return_time_stamps=True,
)
print(results[0].time_stamps[0])      # (text, start, end) per token/word
```

## vLLM backend (CUDA only)

For fastest inference and streaming support:

```python
import torch
from qwen_asr import Qwen3ASRModel

if __name__ == "__main__":  # required — vLLM spawn issue
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128,
        max_new_tokens=4096,
    )
    results = model.transcribe(audio=["clip.wav"], language=["Spanish"])
```

Standalone server:
```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

Hit it via OpenAI-compatible chat completions:
```python
import requests
r = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={"messages": [{
        "role": "user",
        "content": [{"type": "audio_url", "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
        }}],
    }]},
    timeout=300,
)
content = r.json()["choices"][0]["message"]["content"]

from qwen_asr import parse_asr_output
language, text = parse_asr_output(content)
```

## DashScope cloud (no local GPU needed)

For zero local-runtime usage — Qwen offers Qwen3-ASR via DashScope:

- Real-time API: https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition
- FileTrans API: https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition
- Key: set `DASHSCOPE_API_KEY` in `.env` (this benchy install already has it).

The DashScope path is the recommended next adapter to add to benchy if
you want to compare Qwen3-ASR against Voxtral in the SAME venv (the
local path can't due to the transformers conflict).

## Using from benchy

This benchy install ships a `qwen3_asr_chat` adapter under
`src/adapters/qwen3_asr_chat.py` that wraps `qwen_asr.Qwen3ASRModel`.
The model YAML is `configs/models/qwen3-asr-0.6b-transformers.yaml`:

```yaml
model:
  name: Qwen/Qwen3-ASR-0.6B
adapter: qwen3_asr_chat
qwen3_asr_chat:
  dtype: bfloat16
  device_map: cpu                  # 'auto' on Linux+GPU
  language: Spanish                # or omit for auto-detect
  max_inference_batch_size: 32
  max_new_tokens: 256
tasks: [transcription]
```

Run a smoke:

```bash
.venv/bin/benchy eval \
  -c qwen3-asr-0.6b-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 1 --log-samples \
  --run-id qwen3_smoke \
  --exit-policy smoke
```

Verified result (2026-06-20, Mac CPU, 1 FLEURS es_419 sample):
**WER 0.1379, CER 0.0355**, recognizable Spanish output, ~64 s wall time.

## Sample-language hint

The adapter maps each FLEURS sample's ISO 639-1 code (`es`, `pt`, `en`,
…) to the full language string Qwen3-ASR expects (`Spanish`,
`Portuguese`, `English`, …). If the YAML pins `language:`, that wins. If
neither is set, the model auto-detects.

## Version conflict with Voxtral — and the two-venv solution

`qwen-asr 0.0.6` pins `transformers <5.0`. `Voxtral` needs
`transformers >=5.13`. They can't live in the same Python environment.

| Stack | Whisper | Voxtral | Canary | Qwen3-ASR |
|---|---|---|---|---|
| `transformers >=5.13`, no `qwen-asr` (.venv-vox) | ✅ | ✅ | ✅ | ❌ |
| `transformers <5.0` + `qwen-asr` (.venv default) | ✅ | ❌ | ✅ | ✅ |

**Benchy ships a two-venv setup that handles this automatically.**

```bash
# One-time setup — builds both venvs idempotently.
bash scripts/setup-venvs.sh
```

After setup:

```bash
# Default venv covers Whisper + Canary + Qwen3-ASR:
.venv/bin/benchy eval -c qwen3-asr-0.6b-transformers ...

# Voxtral venv covers Whisper + Canary + Voxtral:
.venv-vox/bin/benchy eval -c voxtral-mini-4b-transformers ...
```

Each model YAML that needs a specific venv declares `venv: <path>`.
If you launch `benchy` from the wrong one, the CLI **pre-flight check
fails fast** with the exact replacement command — no mysterious
transformers import errors during inference.

Convenience targets:

```bash
make setup-venvs    # build both
make smoke-qwen     # .venv/bin/benchy ... qwen3-asr-0.6b
make smoke-voxtral  # .venv-vox/bin/benchy ... voxtral-mini-4b
```

Alternative path (no second venv): write a `qwen3_asr_dashscope` cloud
adapter (~60 LOC). The user already has `DASHSCOPE_API_KEY` set.

## Comparison numbers (FLEURS es_419, 1 sample, Mac)

For sanity-checking your install:

| Model | Adapter / interface | WER | CER | Wall time |
|---|---|---|---|---|
| `whisper-large-v3-turbo` | transformers_audio | 0.089 | 0.027 | ~35 s |
| `whisper-1` (cloud) | openai_audio | 0.091 | 0.015 | ~12 s |
| `canary-1b-flash` | canary_nemo | 0.069 | 0.024 | ~90 s |
| `voxtral-mini-4b` (fp16) | voxtral_chat | 0.103 | 0.018 | ~22 min |
| `qwen3-asr-0.6b` | qwen3_asr_chat | 0.138 | 0.036 | ~64 s |

(All Spanish; the same setup with `whisper-tiny` returns WER ~0.265.)

## Files this skill touches

- `src/adapters/qwen3_asr_chat.py` — adapter implementation
- `configs/models/qwen3-asr-0.6b-transformers.yaml` — model config
- `tests/test_adapters_qwen3_asr_chat.py` — mocked tests
- External: `qwen-asr` PyPI package (https://pypi.org/project/qwen-asr/)
- External: `https://github.com/QwenLM/Qwen3-ASR`

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Qwen3ASRForConditionalGeneration not found in transformers` | You're trying to load via `transformers`, not `qwen-asr` | Use `from qwen_asr import Qwen3ASRModel` |
| `check_model_inputs() missing 1 required positional argument` | `transformers >=5.0` + `qwen-asr` | Downgrade: `pip install 'transformers<5.0'` |
| Predictions empty | Wrong language hint, or auto-detect mis-fires | Pass explicit `language="Spanish"` etc. |
| Slow CPU inference | Running on CPU — Qwen3-ASR is CUDA-first | Use `device_map="cuda:0"` on Linux+GPU, or the vLLM backend |
| Voxtral broke after installing qwen-asr | qwen-asr downgraded transformers | Use separate venvs, or skip Voxtral on the qwen venv |
