# Plan: `transformers_audio` provider — in-process HF ASR backend

## Context

`feat/transcription-support` ships exactly one ASR backend: `openai_audio`, which speaks HTTP to a remote (or local) OpenAI-compatible `/v1/audio/transcriptions` endpoint. That works for Whisper served by faster-whisper-server, whisper.cpp, or vLLM — but it locks the benchmark to the Whisper family. Every other production ASR family (Meta MMS, SeamlessM4T-v2, NVIDIA Parakeet, wav2vec2 fine-tunes, etc.) is shipped as a HuggingFace `transformers` model with no first-party OpenAI-compatible server.

To run a LATAM ASR leaderboard that is interesting *for LATAM* (regional Spanish accents, Brazilian Portuguese varieties, eventually indigenous languages), we need an in-process loader. The most general and interoperable choice — by a wide margin — is `transformers`'s `automatic-speech-recognition` pipeline, which auto-dispatches to the right model class for ~every Whisper variant, MMS, SeamlessM4T, wav2vec2, Parakeet, and so on, and runs on CPU (Mac included) without any HTTP server.

This plan adds `transformers_audio` as a sibling provider to `openai_audio`. It follows the contributor's existing architecture exactly: provider type → provider config YAML → interface class subclassing `OpenAIInterface` semantics → model YAML pointing at the provider. The `TranscriptionHandler`, FLEURS subtasks, WER/CER metrics, and capability gates all stay untouched.

---

## Non-goals

- Replacing `openai_audio`. Both providers coexist; users pick by config.
- Adding a model-launching layer that downloads weights on demand from anywhere other than HuggingFace.
- Supporting streaming ASR or live audio. Same constraints as the existing PR (one file in, one transcript out).
- GPU-specific optimizations (FlashAttention, bettertransformer, etc.). Default device picks CPU; `device` is configurable for users who want CUDA/MPS.

---

## Architecture

Mirror `openai_audio`'s contract. The handler treats every interface the same way (`prepare_request` → `_generate_with_limit` → returns `{output, raw, error, error_type}`), so the only thing that has to change is what happens inside `_transcribe_single`: instead of a network call, we run the `transformers` pipeline.

```
TranscriptionHandler (unchanged)
        ↓ prepare_request
TransformersAudioInterface  ←──── new
        ↓ _transcribe_single
transformers.pipeline("automatic-speech-recognition", model=<hf_id>, device=...)
        ↓
{output: "...", raw: "..."}
```

Concurrency: the existing `_semaphore` stays in place, but in-process inference is single-threaded by default. `max_concurrent` defaults to `1` in the new provider config; users on multi-GPU boxes can crank it up.

---

## What changes

### 1. `pyproject.toml` — 1 change

Extend the `transcription` optional dependency group to include `transformers` and `torch`. We use a CPU-only default `torch` install; users on CUDA Linux can swap to a CUDA wheel.

```toml
transcription = [
    "jiwer>=3.0.0",
    "soundfile>=0.12.0",
    "transformers>=4.45.0",
    "torch>=2.4.0",
    "librosa>=0.10.0",        # transformers ASR pipeline preprocessing
    "accelerate>=0.34.0",     # device_map="auto" + dtype handling
]
```

`librosa` and `accelerate` are pulled in by `transformers`'s ASR pipeline path; declaring them explicitly avoids a flaky import on first run.

### 2. `src/interfaces/transformers_audio_interface.py` — NEW (~120 LOC)

Mirror `src/interfaces/openai_audio_interface.py` line-for-line wherever possible to make the diff obvious to a reviewer. Key differences:

- `__init__` lazy-loads `transformers.pipeline("automatic-speech-recognition", model=model_name, device=device, torch_dtype=dtype, chunk_length_s=chunk_length_s)`. Pipeline construction is expensive (~5–60s depending on model), so do it once at interface init, not per-sample.
- No `AsyncOpenAI` client. We *do* keep the `async def _transcribe_single` signature so the handler runner stays unchanged; the actual call is wrapped in `asyncio.to_thread(self._pipeline, audio_path, ...)`.
- `prepare_request` is identical to `openai_audio`'s — still emits `{sample_id, audio_path, language}`.
- Capability flags: same as `openai_audio` (`supports_audio=True`, everything else `False`).

Sketch:

```python
class TransformersAudioInterface:
    def __init__(self, connection_info, model_name):
        self.model_name = model_name
        self.max_concurrent = connection_info.get("max_concurrent", 1)
        self.timeout = connection_info.get("timeout", 600)  # generous; CPU is slow
        self.max_retries = connection_info.get("max_retries", 1)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._capabilities = parse_interface_capabilities(
            connection_info.get("capabilities"),
            default=InterfaceCapabilities(supports_audio=True, ...),
        )
        # Pipeline kwargs come from provider config
        device = connection_info.get("device", "auto")
        dtype = connection_info.get("torch_dtype", "float32")
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=getattr(torch, dtype),
            chunk_length_s=connection_info.get("chunk_length_s", 30),
        )

    def prepare_request(self, sample, task):
        # identical to OpenAIAudioInterface.prepare_request
        ...

    async def _transcribe_single(self, audio_path, language, sample_id):
        def _run():
            kwargs = {}
            if language:
                kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}
            result = self._pipeline(audio_path, **kwargs)
            return result.get("text", "").strip()

        try:
            text = await asyncio.wait_for(asyncio.to_thread(_run), timeout=self.timeout)
            return {"output": text, "raw": text, "error": None, "error_type": None}
        except Exception as exc:
            return {"output": None, "raw": None, "error": str(exc), "error_type": type(exc).__name__}

    async def _generate_with_limit(self, req):
        async with self._semaphore:
            return await self._transcribe_single(req["audio_path"], req.get("language"), req["sample_id"])
```

Caveats encoded in this sketch:
- `generate_kwargs` with `language`+`task` is the **Whisper-specific** path. For non-Whisper models (MMS, wav2vec2) that don't accept those kwargs, the pipeline will raise. We pass language only if the model config opts in via a `supports_language_kwarg: true` flag in the model YAML. (Document this in the model config docstring.)
- `chunk_length_s=30` matches Whisper's training window; for non-Whisper models we expose it as a provider/model knob.

### 3. `src/benchy_cli_eval.py` — 4 changes

**a.** Add to `PROVIDER_SPECS` (around line 73):

```python
"transformers_audio": {
    "config_key": "transformers_audio",
    "log": "Using transformers in-process audio provider for model: {model_name}",
},
```

**b.** Add to `MODEL_PROVIDER_TYPES` (line 83):

```python
MODEL_PROVIDER_TYPES = {"vllm", "openai", "anthropic", "together", "alibaba", "google", "openai_audio", "transformers_audio"}
```

**c.** Add to `CLI_PROVIDER_DEFAULTS` (line 140):

```python
"transformers_audio": {
    "device": "auto",
    "torch_dtype": "float32",
    "chunk_length_s": 30,
    "timeout": 600,
    "max_retries": 1,
    "max_concurrent": 1,
    "api_endpoint": "audio",
},
```

**d.** Add `"transformers_audio"` to the `--provider` choices list (line 581).

### 4. `src/engine/connection.py` — 1 change

Register the new provider so it's reachable from `benchmark_pipeline`. Mirror the wiring the contributor added for `openai_audio` (the diff is 16 lines so this is mechanical).

### 5. `configs/providers/transformers_audio.yaml` — NEW

```yaml
# HuggingFace transformers in-process ASR provider.
# Loads the model into the benchy process via transformers.pipeline.
# No HTTP server, no auth, runs on CPU by default. Set `device: cuda` or
# `device: mps` to use GPU/Metal acceleration.

provider_type: transformers_audio

device: auto              # auto | cpu | cuda | mps | cuda:0 ...
torch_dtype: float32      # float32 | float16 | bfloat16
chunk_length_s: 30        # audio chunk size; 30s matches Whisper's training window

# In-process pipelines can't parallelize across requests the way HTTP can.
# Default to serial execution; users with multi-GPU boxes can raise this.
max_concurrent: 1
timeout: 600              # CPU inference can be slow on large clips
max_retries: 1

capabilities:
  supports_audio: true
  supports_multimodal: false
  supports_schema: false
  supports_files: false
  supports_logprobs: false
  supports_streaming: false
  request_modes: ["audio"]
```

### 6. `configs/models/*.yaml` — NEW (one per model we want to benchmark)

Start with three to validate the pattern:

`configs/models/whisper-small-transformers.yaml`:

```yaml
model:
  name: openai/whisper-small
transformers_audio:
  provider_config: transformers_audio
  supports_language_kwarg: true     # Whisper accepts generate_kwargs.language
task_defaults:
  log_samples: true
tasks:
- transcription
metadata:
  provider: huggingface
  model_type: whisper
  is_cloud: false
  description: "openai/whisper-small via transformers pipeline (CPU-friendly)"
```

Plus `whisper-large-v3-transformers.yaml` and `whisper-large-v3-turbo-transformers.yaml`. Same structure, swap `model.name`. These three give us a clean speed/accuracy gradient on the same audio.

(Future LATAM-relevant non-Whisper models — MMS, SeamlessM4T, wav2vec2 fine-tunes — set `supports_language_kwarg: false`. Out of scope for this plan but the architecture supports them.)

### 7. `tests/test_transformers_audio_interface.py` — NEW (~80 LOC)

Mirror `tests/test_openai_audio_interface.py`. Tests:

- `test_capabilities_advertise_audio_only` — same checks as the openai_audio version.
- `test_prepare_request_returns_expected_shape` — verify the sample dict shape.
- `test_prepare_request_falls_back_to_task_language` — verify language passthrough.
- `test_prepare_request_raises_when_audio_path_missing` — error shape.
- `test_transcribe_single_calls_pipeline_with_language_kwargs` — patch `transformers.pipeline` to return a stub whose `__call__` returns `{"text": "hola mundo"}`, assert it was invoked with `generate_kwargs={"language": "es", "task": "transcribe"}`.
- `test_transcribe_single_omits_language_when_unsupported` — when `supports_language_kwarg=False`, no `generate_kwargs` is passed.
- `test_transcribe_single_surfaces_runtime_errors` — pipeline raises → result dict has `error` and `error_type` set.

All tests mock the pipeline; no torch / transformers import at test collection time (lazy import inside the interface lets us monkey-patch).

### 8. Capability test — extend `tests/test_transcription_capabilities.py`

Add a parametrized case that builds a `TransformersAudioInterface` (with the pipeline mocked) and asserts it gates correctly against `requires_audio: required` tasks.

### 9. `src/tasks/transcription/metadata.yaml` — 1 edit

Add a `notes` line: "Models with `transformers_audio` provider run in-process; first invocation downloads weights from HuggingFace."

### 10. Documentation — 2 edits

- `coREADME.md`: add a one-paragraph "Running ASR locally with transformers" section pointing at the new model configs.
- `.plans/AGENTS.md`: extend the smoke run example with `whisper-small-transformers.yaml` so automation knows the canonical CPU-only test config.

---

## Acceptance criteria

1. `pip install -e '.[transcription]'` succeeds on a clean Python 3.12 venv (Mac CPU and Linux CPU).
2. All existing 35 tests in the PR still pass.
3. Seven new tests in `tests/test_transformers_audio_interface.py` pass.
4. The capability test passes for both interface types.
5. End-to-end smoke run succeeds on CPU with no HTTP server:

```bash
benchy eval -c whisper-small-transformers \
  --tasks fleurs_es_latam --limit 3 --log-samples \
  --exit-policy smoke
```

6. `run_outcome.json` reports `passed` or `degraded`, all `*_tasks` counts are 0, aggregated `wer ≤ 0.40` on those 3 samples (cf. `TRANSCRIPTION-SMOKE-TEST-PLAN.md` §4).

7. Per-sample log output contains recognizable Spanish transcripts.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `torch` install in `transcription` extra bloats the wheel and slows install for users who only want cloud `openai_audio`. | Split into `transcription` (cloud only) and `transcription-local` (adds torch + transformers). Users opt in explicitly. |
| Pipeline init is slow (~5–60s) and counts against the eval wall-clock. | One-time cost amortized across all samples of a run. Document in the model YAML descriptions. |
| Mac users hit unsupported ops on `mps` for some models (e.g. SeamlessM4T). | Default to `device: auto` which prefers `mps` but falls back to `cpu` on op error; document the gotcha in the provider YAML. |
| `transformers` API churns and breaks our pipeline kwargs across versions. | Pin to `transformers>=4.45,<5.0` and bake a smoke test that fails fast on a kwarg change. |
| Concurrency=1 makes full FLEURS runs slow (~30–60 min on CPU). | Out of scope for this plan; tune later or recommend a CUDA box for full runs. Smoke runs use `--limit`. |
| Different model families need different decoding kwargs (Whisper wants `language`, wav2vec2 doesn't, MMS uses `target_lang`). | The `supports_language_kwarg` flag in the model YAML is a first step. A richer "model adapter" abstraction is deferred until we actually add a non-Whisper model — premature now. |
| `librosa` is large (~50 MB) and pulls in `numba`. | Required by transformers ASR preprocessing; document, don't optimize. |

---

## Out of scope for this plan (but unlocked by it)

- Adding MMS, SeamlessM4T, Parakeet model configs. The architecture supports them once we set `supports_language_kwarg: false` and verify decoding kwargs per family.
- Indigenous-language datasets (Quechua, Guaraní). Datasets are the same shape as FLEURS; what's missing is models that speak those languages — and `transformers_audio` is the door.
- A model-launching layer that pre-downloads weights at config load time (currently lazy on first sample).
- Multi-GPU model sharding (`accelerate`'s `device_map="auto"` for 70B+ ASR models that don't exist yet).

---

## Definition of done

All acceptance criteria above pass. We can run

```bash
benchy eval -c whisper-small-transformers --tasks fleurs_es_latam --limit 3
benchy eval -c whisper-large-v3-transformers --tasks fleurs_es_latam --limit 3
benchy eval -c whisper-large-v3-turbo-transformers --tasks fleurs_es_latam --limit 3
```

on a CPU-only Mac with no Docker, no HTTP server, no API keys, and get three different WER numbers for the same audio. That gradient is the proof that the provider is *general* (works across model sizes) and *interoperable* (the same handler/metrics/aggregator that runs cloud `whisper-1` runs local `whisper-large-v3`). From there, expanding to non-Whisper families and LATAM-specific datasets is a per-model / per-dataset change, not a per-architecture change.
