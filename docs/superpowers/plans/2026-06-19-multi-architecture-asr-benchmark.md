# Multi-Architecture ASR Benchmark Plan

**Goal:** Produce a comparative WER / CER table across four ASR architectures
on FLEURS Latin-American Spanish and Brazilian Portuguese, then codify the
working procedure as a benchy agent skill.

**Architectures under test:**

| Family | Config | HF repo | Params | Architecture style | Notes |
|---|---|---|---|---|---|
| Whisper (latest, Mac-practical) | `whisper-large-v3-turbo-transformers` | `openai/whisper-large-v3-turbo` | 809M | Encoder-decoder, Whisper-style `generate_kwargs.language` | Mac ceiling — `whisper-large-v3` (1.55B) wedges on MPS+fp32 (known) |
| Voxtral | `voxtral-mini-4b-transformers` | `mistralai/Voxtral-Mini-4B-Realtime-2602` | ~4B | Multimodal LLM-style ASR | Custom code (`trust_remote_code: true`); doesn't auto-dispatch through `pipeline("automatic-speech-recognition")` |
| Canary | `canary-1b-flash-transformers` | `nvidia/canary-1b-flash` | 883M | NVIDIA NeMo encoder; auto-language ID | Custom code; languages en/de/es/fr — **pt_br NOT supported, expect skip or high WER** |
| Qwen3-ASR | `qwen3-asr-0.6b-transformers` | `Qwen/Qwen3-ASR-0.6B` | 600M | Prompt-conditioned LLM-style ASR | Custom code; multilingual |

**Tech stack:** benchy `eval` CLI, `transformers_audio` provider, `transcription.fleurs_{es_latam,pt_br}` tasks, `scripts/run_asr_panel.py`.

**Data:** FLEURS Latin-American Spanish (`es_419`) and Brazilian Portuguese (`pt_br`) audio. 25 samples per locale per model is enough for a stable WER signal while keeping wall time bounded.

## Global Constraints

- **Disk budget:** ~12 GB total downloads (whisper-turbo ~1.6 GB, canary ~1.7 GB, voxtral ~8 GB, qwen3-asr ~1.2 GB) cached under `~/.cache/huggingface/`. Verify free space before Phase 1.
- **Memory budget:** Voxtral-4B in fp32 ≈ 16 GB RAM; fp16 ≈ 8 GB. On a 16 GB Mac, force fp16 for Voxtral.
- **No public-surface changes.** This benchmark uses benchy as-is — no edits to schemas, CLI flags, or `run_outcome.json`. Per-model `configs/models/*.yaml` may need tuning (device, dtype, supports_language_kwarg). Per-family interface code may be needed if `transformers.pipeline` can't auto-dispatch a model — flag and scope as a sub-task, do not over-build.
- **One commit per phase outcome.** Per-model fixes land as one commit each so partial progress is preserved if a later model can't be made to work.
- Run from repo root, with `.venv` Python and `OPENAI_API_KEY` (only needed if comparing against `whisper-1` cloud baseline).

## File Map

| Path | Touch type | Why |
|---|---|---|
| `configs/models/{whisper-large-v3-turbo,canary-1b-flash,voxtral-mini-4b,qwen3-asr-0.6b}-transformers.yaml` | Modify (tune device/dtype) | All four exist; tuning may be needed per Phase 1 findings |
| `src/interfaces/transformers_audio_interface.py` | Potentially modify | Pipeline auto-dispatch may not work for Voxtral / Canary / Qwen3-ASR — may need a small dispatcher branch |
| `scripts/run_asr_panel.py` | Use as-is | Already accepts `--models` list and iterates locales |
| `outputs/benchmark_outputs/multi_arch_panel/` | Produced | Run output dir |
| `outputs/asr_panel_summary.json` | Produced | Comparative summary |
| `.agent/skills/multi-architecture-asr-benchmark/SKILL.md` | **Create at end** | The deliverable skill |
| `docs/how-to-multi-architecture-asr-benchmark.md` | **Create at end** | Colleague-shareable mirror of the skill |

---

## Phase 1 — Per-model smoke (4 tasks)

For each model, verify it (a) downloads weights, (b) loads through the
`transformers_audio` interface, (c) produces non-empty predictions, (d)
yields a WER < 1.0 on 3 FLEURS es_419 samples.

The 3-sample smoke is intentionally tiny so a wedged model is detected
fast. The full benchmark uses 25 samples in Phase 2.

### Task 1.1: Whisper-large-v3-turbo (es_419 only)

- [ ] **Disk + memory check**

```bash
df -h ~ | tail -1                       # need ≥ 12 GB free in $HOME
sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB total RAM"}'
```

- [ ] **Smoke run, 3 samples**

```bash
set -a; source .env; set +a
.venv/bin/benchy eval \
  -c whisper-large-v3-turbo-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 3 --log-samples \
  --run-id smoke_whisper_turbo \
  --exit-policy smoke
```

Expected: exit 0, `Run status: passed`, WER < 0.20 (this model is strong on Spanish).

- [ ] **Inspect predictions are real Spanish**

```bash
find outputs/benchmark_outputs/smoke_whisper_turbo_LIMITED -name '*samples.json' -exec head -c 800 {} \;
```

Expected: recognizable Spanish in the `prediction` field of each sample.

- [ ] **Commit** (only if changes were needed to the YAML):

```bash
git add configs/models/whisper-large-v3-turbo-transformers.yaml
git commit -m "asr-bench: tune whisper-large-v3-turbo config for smoke pass"
```

### Task 1.2: Canary-1b-flash (es_419 only — pt_br not in language set)

- [ ] **Smoke run**

```bash
.venv/bin/benchy eval \
  -c canary-1b-flash-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 3 --log-samples \
  --run-id smoke_canary \
  --exit-policy smoke
```

Possible failure modes (handle the first one you hit):

1. **`No module named 'nemo_toolkit'` / model class not found** → Canary's HF integration expects NeMo. Either install `nemo-toolkit[asr]` (heavy, ~2 GB) or add a `canary` branch in `transformers_audio_interface.py` that loads via `AutoModelForSpeechSeq2Seq.from_pretrained(..., trust_remote_code=True)` directly and runs inference manually instead of through `pipeline()`. Try `pipeline` first.

2. **Pipeline auto-dispatch fails** with "Unrecognized model type" → same fallback: per-family loader branch keyed on `metadata.model_type == "canary"`. Scope it to ~30 lines max.

3. **Loads but produces gibberish or empty strings** → check the audio-input contract; Canary expects 16 kHz mono PCM. FLEURS audio is already 16 kHz, so this should not happen.

- [ ] **If a code change was needed**, run the focused interface test before committing:

```bash
.venv/bin/python -m pytest tests/test_transformers_audio_interface.py -q
```

- [ ] **Commit** (config and/or interface):

```bash
git add configs/models/canary-1b-flash-transformers.yaml src/interfaces/transformers_audio_interface.py
git commit -m "asr-bench: get canary-1b-flash loading via transformers_audio"
```

### Task 1.3: Qwen3-ASR-0.6B

- [ ] **Smoke run**

```bash
.venv/bin/benchy eval \
  -c qwen3-asr-0.6b-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 3 --log-samples \
  --run-id smoke_qwen3_asr \
  --exit-policy smoke
```

Likely failure modes:

1. **Custom prompt-conditioned interface** — Qwen3-ASR is an LLM that takes an audio embedding + text prompt and decodes a transcription. `pipeline("automatic-speech-recognition")` may not pass the prompt correctly. If predictions are empty or look like "[Audio]" placeholders, add a `qwen3_asr` branch in `transformers_audio_interface.py` that:
   - Loads the model with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and the matching processor.
   - Builds the input the way the model card shows (typically `messages=[{role:"user", content:[{type:"audio", audio:wav}, {type:"text", text:"Transcribe this audio."}]}]`).
   - Decodes the generated text and returns it as the `output`.

2. **Model card is wrong about HF integration** — check `https://huggingface.co/Qwen/Qwen3-ASR-0.6B`. If it ships its own runner script, the simplest fix is to vendor a thin adapter at `src/interfaces/qwen3_asr_interface.py` and register it as a distinct provider (mirroring how `transformers_audio` itself was added). Treat this as a separate sub-task if it lands here — do not over-engineer it inline.

- [ ] **Test + commit** (same shape as Task 1.2).

### Task 1.4: Voxtral-Mini-4B (force fp16 on Mac)

- [ ] **Force fp16 in the config first** (Mac memory):

```yaml
# configs/models/voxtral-mini-4b-transformers.yaml — under transformers_audio:
  torch_dtype: float16
```

Why: Voxtral-4B fp32 = ~16 GB RAM. On a 16 GB Mac, the OS won't have enough head room and you'll hit swap thrash. fp16 = ~8 GB and runs cleanly on M-series MPS or CPU.

- [ ] **Smoke run** (allow longer timeout — 4B model on Mac will be slow):

```bash
timeout 600 .venv/bin/benchy eval \
  -c voxtral-mini-4b-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 3 --log-samples \
  --run-id smoke_voxtral \
  --exit-policy smoke
```

Voxtral is a multimodal LLM-style ASR — same caveat as Qwen3-ASR. If pipeline auto-dispatch fails, the fix is structurally identical: per-family branch in `transformers_audio_interface.py` keyed on `metadata.model_type == "voxtral"`, using `AutoProcessor` + `AutoModelForCausalLM` with `trust_remote_code: true` and the chat-template the model card prescribes.

- [ ] **Memory-watch during the run** (separate shell):

```bash
top -l 1 -stats pid,command,cpu,rsize,vsize | grep -i 'python\|benchy' | head -3
```

If `rsize` (RSS) climbs past 10 GB and the process stalls, kill -9 and retry with `device: cpu` (slower, no MPS dispatch quirks).

- [ ] **Commit** (config tuning + any interface branch).

### Phase 1 gate

After all four 1.X tasks: each model has a passing 3-sample smoke run with
non-empty, language-plausible predictions, OR has a documented BLOCKED
status with the specific failure mode. A BLOCKED model is **not** treated
as failure of the plan — record it in the eventual skill as a known
limitation. Phase 2 runs only the models that smoked clean.

---

## Phase 2 — Full panel run (1 task)

### Task 2.1: 25-sample × 2-locale × 4-model panel

- [ ] **Disk check** — at this point all weights are cached. Verify:

```bash
du -sh ~/.cache/huggingface/hub 2>/dev/null | head -1
```

- [ ] **Run the panel** (skip-failures lets a wedged model not block the others):

```bash
.venv/bin/python scripts/run_asr_panel.py \
  --limit 25 \
  --locales es pt \
  --models whisper-large-v3-turbo-transformers \
           canary-1b-flash-transformers \
           qwen3-asr-0.6b-transformers \
           voxtral-mini-4b-transformers \
  --skip-failures \
  --run-id multi_arch_panel
```

Expected wall time on Apple Silicon (M2/M3): ~45-90 min. Voxtral dominates.

Expected output:
- `outputs/benchmark_outputs/multi_arch_panel/<model>/transcription/fleurs_*/...metrics.json` per (model, locale)
- `outputs/asr_panel_summary.json` — comparative table

- [ ] **Sanity-check the summary** before drawing conclusions:

```bash
.venv/bin/python -c "
import json
with open('outputs/asr_panel_summary.json') as f:
    data = json.load(f)
for row in data:
    print(f\"{row.get('model'):45s} {row.get('locale'):8s} WER={row.get('wer'):.3f} CER={row.get('cer'):.3f} valid={row.get('valid_samples')}/{row.get('total_samples')}\")
"
```

Healthy: every row has `valid_samples == total_samples`, WER between 0.05 and 0.40 for Spanish, between 0.05 and 0.50 for Portuguese. Canary on pt_br is expected to be high (model doesn't support pt) — note it, don't fix it.

- [ ] **Stash the panel summary as a result artifact:**

```bash
mkdir -p docs/benchmarks
cp outputs/asr_panel_summary.json docs/benchmarks/multi_arch_asr_panel_2026-06-19.json
```

- [ ] **Commit** the result artifact + a short markdown table next to it:

```bash
# Generate a markdown summary table in docs/benchmarks/multi_arch_asr_panel_2026-06-19.md
# (controller produces this inline from the JSON)
git add docs/benchmarks/multi_arch_asr_panel_2026-06-19.{json,md}
git commit -m "asr-bench: results — 4-architecture FLEURS es+pt panel"
```

### Phase 2 gate

A committed result table with at least 6 of 8 (model, locale) cells filled
in (allowing for Canary/pt_br skip + at most one BLOCKED model).

---

## Phase 3 — Codify as a benchy skill

### Task 3.1: Author the skill

Once Phase 2 completes, the working procedure (which models, which configs,
which fixes applied, which gotchas were real) is now empirical fact, not
speculation. The skill writes itself.

- [ ] **Create the skill directory:**

```bash
mkdir -p .agent/skills/multi-architecture-asr-benchmark
```

- [ ] **Write `.agent/skills/multi-architecture-asr-benchmark/SKILL.md`** with this structure:

```markdown
---
name: multi-architecture-asr-benchmark
description: >
  Run a comparative ASR benchmark across architectures (latest Whisper,
  Voxtral, Canary, Qwen3-ASR) on FLEURS es_419 + pt_br using benchy's
  transformers_audio provider. Covers per-model setup, the panel runner,
  per-family caveats discovered during real runs, and result
  interpretation. Triggers on: "compare ASR architectures", "benchmark
  voxtral canary qwen3 whisper", "multi-model speech-to-text panel".
---

# Multi-architecture ASR benchmark

[Body: lift the working content from this plan, with the *actual* fixes that
ended up being needed in Phase 1, and the *actual* numbers from Phase 2's
result table. No speculation; cite the result artifact.]

## When to use this
...

## One-time setup
...

## Per-model setup notes (real findings)
- Whisper-large-v3-turbo: [what actually worked]
- Canary-1b-flash: [what actually worked, including any interface branch]
- Qwen3-ASR-0.6B: [what actually worked]
- Voxtral-Mini-4B: [what actually worked, including dtype/device]

## Running the panel
...

## What good output looks like
[Embed the 2026-06-19 result table verbatim.]

## Files this skill touches
[Per-family interface branches, configs, the runner.]
```

- [ ] **Write `docs/how-to-multi-architecture-asr-benchmark.md`** as the colleague-shareable mirror (no YAML frontmatter, same content). Mirrors the existing `docs/how-to-transcription-benchmark.md` shape.

- [ ] **Cross-link from the existing `whisper-benchmark` SKILL**: add a "Beyond Whisper" section pointing to the new skill.

- [ ] **Smoke-test the skill** by following it cold (read SKILL.md, run the smallest command in it, confirm exit 0):

```bash
.venv/bin/benchy eval -c whisper-large-v3-turbo-transformers \
  --tasks transcription.fleurs_es_latam --limit 1 \
  --run-id skill_dryrun --exit-policy smoke
```

- [ ] **Commit:**

```bash
git add .agent/skills/multi-architecture-asr-benchmark/ \
        docs/how-to-multi-architecture-asr-benchmark.md \
        .agent/skills/whisper-benchmark/SKILL.md
git commit -m "asr-bench: codify multi-architecture procedure as a benchy skill"
```

### Phase 3 gate

A future Claude session asked to "compare ASR architectures on benchy"
loads the skill, runs the panel, and reproduces a result table within
~10% of the 2026-06-19 numbers.

---

## Risks and rollback

- **A model can't be made to work in this repo.** Acceptable outcome — the
  skill documents it as a known limitation. Do not let one stuck model
  block the panel; `--skip-failures` exists for this.
- **Voxtral OOMs even at fp16.** Fallback: drop to `device: cpu` (slow but
  predictable), or skip Voxtral with a "needs ≥ 24 GB RAM" note.
- **Per-family interface branches grow.** Hard cap: ~40 lines of dispatcher
  code in `transformers_audio_interface.py`. Beyond that, each family gets
  its own interface file like `transformers_audio_interface.py` did
  (separate provider in `configs/providers/`). Don't compress logic to
  fit the cap.
- **Result drift over time.** The panel JSON has dates in its filename;
  multiple snapshots are fine, the skill cites the latest.
- **Rollback.** Each commit is independent. `git revert <sha>` undoes one
  model's tuning without touching the others. The Phase 3 skill commit can
  be reverted without affecting the result artifact.

## Open implementation-time questions

- Whether to also include `whisper-1` (OpenAI cloud) in the panel as a
  baseline. Default: yes, add `--include-cloud whisper-1` (or just pass it
  in `--models`). Decide after Phase 1 based on whether the user has
  budget for ~50 cloud calls (~$0.05).
- Whether to use 25 or 50 samples per locale in Phase 2. Default: 25 to
  keep wall time bounded; bump to 50 if the local models finish faster
  than estimated.
- Whether the per-family interface branches stay inside
  `transformers_audio_interface.py` or graduate to their own files. Default:
  start inline behind the soft cap; promote once any branch exceeds ~25
  lines or starts pulling in family-specific imports.
