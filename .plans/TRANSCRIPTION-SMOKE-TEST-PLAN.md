# Plan: Smoke-test the transcription PR with faster-whisper-server

## Context

`feat/transcription-support` (commits `1d0e9c5..2d9aecd`, contributor: franperelman) adds an `openai_audio` provider that hits any OpenAI-compatible `/v1/audio/transcriptions` endpoint. The 35 unit tests in the PR are fully mocked — they never decode real audio or call a real ASR endpoint. Before merging this PR (or before building a LATAM-wide benchmark on top of it), we need to prove the whole pipeline works end-to-end on at least one real audio sample.

The constraint: we want CPU-only, Mac-friendly, **zero new code in benchy**, and we accept a Whisper-only smoke test (Whisper is also the only model the PR ships a config for, so this is consistent with the PR's scope).

The path: stand up `faster-whisper-server` in Docker, which exposes the same `/v1/audio/transcriptions` shape OpenAI does, and point benchy's existing `openai_audio` interface at it via `--base-url`.

---

## Non-goals

- Adding new providers, interfaces, or model loaders.
- Benchmarking non-Whisper ASR models. (See `TRANSFORMERS-AUDIO-PROVIDER-PLAN.md` for that.)
- Producing a publishable WER number. This is a wiring smoke test; meaningful numbers come from the LATAM benchmark plan once we have model breadth.

---

## What we run

### 1. One-time setup

```bash
pip install -e '.[transcription]'   # adds jiwer + soundfile to the existing install
```

Pull the prebuilt server image (CPU variant, no CUDA required):

```bash
docker pull fedirz/faster-whisper-server:latest-cpu
```

### 2. Tier 1 — unit tests (no network, no Docker)

Confirms the contributor's tests pass on this machine before we worry about live audio. Free, ~5s.

```bash
pytest -v \
  tests/test_audio_preprocessing.py \
  tests/test_fleurs_subtasks.py \
  tests/test_openai_audio_interface.py \
  tests/test_transcription_capabilities.py \
  tests/test_transcription_handler.py \
  tests/test_wer_cer_metrics.py
```

**Pass criteria:** 35 passed, 0 failed.

### 3. Tier 2 — live end-to-end on 3 FLEURS samples

**Terminal A** — start the local Whisper server. `Systran/faster-whisper-small` is a 244M-param int8 model (~150 MB download, ~1 GB RAM at runtime, transcribes a 10s clip in a few seconds on CPU):

```bash
docker run --rm -p 8000:8000 \
  -e WHISPER__MODEL=Systran/faster-whisper-small \
  -e WHISPER__COMPUTE_TYPE=int8 \
  fedirz/faster-whisper-server:latest-cpu
```

**Terminal B** — run benchy against it. Downloads ~1–2 GB of FLEURS `es_419` parquet on first run (cached afterward), saves the first 3 audio bytes under `.data/transcription/es_419/`, transcribes them, scores against ground truth.

```bash
benchy eval -c whisper-1 \
  --tasks fleurs_es_latam \
  --base-url http://localhost:8000/v1 \
  --api-key-env DUMMY \
  --model-name Systran/faster-whisper-small \
  --limit 3 -v --log-samples \
  --exit-policy smoke
```

`DUMMY` is a placeholder env var — faster-whisper-server doesn't validate auth headers, but the OpenAI SDK refuses to construct without something in the `Authorization` header.

### 4. Validate the results

Per `.plans/AGENTS.md`, the source of truth is `<base_output_path>/<run_id>/<model_name_segment>/run_outcome.json`.

**Pass criteria:**

- Process exit code `0`.
- `run_outcome.status == "passed"` or `"degraded"`.
- All counts of `failed_tasks`, `error_tasks`, `pending_tasks`, `no_samples_tasks`, `skipped_tasks` equal `0`.
- Aggregated metrics include numeric `wer`, `cer`, `exact_match`, and a `per_locale.es_419` block with `sample_count: 3`.
- WER in a plausible range: `0.05 ≤ wer ≤ 0.40`. Outside this range — especially `wer > 0.8` — means the language hint isn't reaching the server, audio is being decoded wrong, or the ground-truth text isn't aligning.

The `--log-samples` flag writes per-sample `{prediction, expected, wer}` to the run output dir; eyeball at least one sample to confirm the transcript is recognizably Spanish.

---

## What this proves (and what it doesn't)

**Proves:**
- `OpenAIAudioInterface` correctly POSTs WAV bytes and parses `response_format=text`.
- The `language` parameter flows from `TranscriptionHandler` → `prepare_request` → SDK kwargs.
- FLEURS download + `Audio(decode=False)` byte streaming + `save_audio_bytes` cache work end-to-end.
- WER/CER/ExactMatch compute and aggregate correctly with `per_locale` bucketing.
- The CLI override path (`--base-url`, `--model-name`) works against a non-OpenAI endpoint.

**Does NOT prove:**
- Anything about non-Whisper models (MMS, SeamlessM4T, wav2vec2). See `TRANSFORMERS-AUDIO-PROVIDER-PLAN.md`.
- Performance / cost at full FLEURS scale (~647 samples × 2 locales).
- Multi-model comparison (only one model + one dataset is exercised).
- That the `openai_audio` provider auto-launch via benchy's config system works — we used CLI overrides to short-circuit that path.

---

## Failure modes and remediations

| Symptom | Most likely cause | Fix |
|---|---|---|
| `ImportError: jiwer` or `soundfile` | Forgot the `[transcription]` extra. | Re-run install command above. |
| 401 / "missing api key" before any HTTP traffic | OpenAI SDK rejected empty auth. | Confirm `DUMMY` env var is set, or pass `--api-key sk-anything`. |
| Container starts but `curl http://localhost:8000/v1/models` times out | First-time Whisper model download from HF inside the container. | Wait 1–2 min, check `docker logs`, retry. |
| FLEURS download stalls or 403 | HF rate limit or missing acceptance. | `huggingface-cli login`, set `HF_HUB_ENABLE_HF_TRANSFER=1`. |
| All 3 samples error with timeout | CPU is slow + default `timeout=120` is tight for large-v3. | Stick with `small` for the smoke test, or pass `--timeout 300`. |
| WER ≥ 0.8 on 3 samples | Language hint not reaching the server, or audio sample rate mismatch. | Inspect `--log-samples` output. If predictions are English-sounding, language passthrough broke; if predictions are gibberish, suspect the server config. |
| All samples returned but `wer` is `None` in aggregate | `jiwer` import failed silently. | Check `pytest tests/test_wer_cer_metrics.py` — should fail too. Reinstall `jiwer`. |

---

## Time and cost

- Tier 1: ~5 seconds, $0.
- Tier 2 first run: ~3–5 minutes (Docker pull + FLEURS download dominate), $0.
- Tier 2 subsequent runs: ~20–30 seconds, $0.

---

## Definition of done

The `run_outcome.json` validation in §4 passes, AND at least one per-sample transcript in `--log-samples` output is a recognizable Spanish sentence matching the expected text within a sensible edit distance. At that point we have proof that `feat/transcription-support` works end-to-end on a CPU-only machine, and we can confidently move on to building the LATAM benchmark on top of it.
