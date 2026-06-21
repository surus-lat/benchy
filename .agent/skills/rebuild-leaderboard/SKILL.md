---
name: rebuild-leaderboard
description: Re-run all models on the LatamBoard leaderboard from scratch after data loss. Identifies which model configs exist, runs each one with the full latam_board task suite on the cluster, and publishes results to HuggingFace after each model so progress is never lost.
---

# Rebuild Leaderboard from Scratch

Use when raw benchmark outputs are gone (e.g., cluster wipe) and you need to
re-evaluate all models that appear on latamboard.surus.lat.

**Cluster:** `ssh cluster.surus.ddns.net` — 4× RTX 3090 (24 GB each)

---

## How this works

The HuggingFace dataset (`mauroibz/leaderboard-results`) still has the old
processed scores. You're not replacing them — you're re-running each model,
getting fresh scores, and merging them back with `merge_and_publish`. Publish
after **every model**, not at the end. If a run crashes, you keep everything
already published.

---

## Phase 1 — Prep (local or cluster)

**1a. Confirm HF_TOKEN is set**

```bash
grep HF_TOKEN .env || echo "MISSING — add HF_TOKEN=hf_... to .env"
```

**1b. Check the dataset name in config**

```bash
grep "results:" configs/config.yaml
# Should be:  results: "mauroibz/leaderboard-results"
# If it says  results: "LatamBoard/leaderboard-results"  → fix it first
```

**1c. See what's currently on the leaderboard**

```bash
python3 -c "
import json
with open('outputs/publish/summaries/all_model_summaries.json') as f:
    data = json.load(f)
print(f'{len(data)} models on HF:')
for k in sorted(data): print(' ', k)
" 2>/dev/null || echo "(no local summaries — that's fine, they're on HF)"
```

---

## Phase 2 — Models with existing configs (run these first)

These models have a config in `configs/models/` and can be run immediately.
All use `--tasks latam_board` regardless of what the config file says, to
ensure consistent coverage (spanish + portuguese + translation + structured_extraction).

### The run loop

Run **one model at a time** — each takes a full GPU card and finishes in
30–90 minutes depending on model size.

```bash
# Template — repeat for each model below
benchy eval --config configs/models/<CONFIG>.yaml --tasks latam_board
# note the run_id printed at the start, then:
python -m src.leaderboard.merge_and_publish --run-id <RUN_ID>
```

### Model list

| Config file | Model | GPU config |
|---|---|---|
| `zephyr-7b-beta.yaml` | HuggingFaceH4/zephyr-7b-beta | single card |
| `llama3.1.yaml` | meta-llama/Llama-3.1-8B-Instruct | single card |
| `llama3.2.yaml` | meta-llama/Llama-3.2-3B-Instruct | single card |
| `ministral8b.yaml` | mistralai/Ministral-8B-Instruct-2410 | single card |
| `DeepSeek-R1-Distill-Qwen-7B.yaml` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | single card |
| `DeepSeek-R1-Distill-Llama-8B.yaml` | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | single card |
| `Yi-1.5-6B-Chat.yaml` | 01-ai/Yi-1.5-6B-Chat | single card |
| `Yi-1.5-9B-Chat.yaml` | 01-ai/Yi-1.5-9B-Chat | single card |
| `Hermes-3-Llama-3.1-8B.yaml` | NousResearch/Hermes-3-Llama-3.1-8B | single card |
| `gemma3n2.yaml` | google/gemma-3n-E2B-it | single card |
| `gemma3n4.yaml` | google/gemma-3n-E4B-it | single card |
| `hormoz8b.yaml` | Hormoz-8B | single card |
| `phi4mini.yaml` | microsoft/Phi-4-mini-instruct | single card |
| `aya8b.yaml` | CohereLabs/aya-expanse-8b | single card |
| `qwen34b.yaml` | Qwen/Qwen3-4B-Instruct | single card |

With 4× 3090s you can run up to 4 models in parallel by pinning each to a
different card:

```bash
# Parallel example — pin each run to a specific GPU
CUDA_VISIBLE_DEVICES=0 benchy eval --config configs/models/zephyr-7b-beta.yaml --tasks latam_board &
CUDA_VISIBLE_DEVICES=1 benchy eval --config configs/models/llama3.1.yaml --tasks latam_board &
CUDA_VISIBLE_DEVICES=2 benchy eval --config configs/models/ministral8b.yaml --tasks latam_board &
CUDA_VISIBLE_DEVICES=3 benchy eval --config configs/models/DeepSeek-R1-Distill-Qwen-7B.yaml --tasks latam_board &
wait
# Then publish all four:
python -m src.leaderboard.merge_and_publish --run-id <ID1>
python -m src.leaderboard.merge_and_publish --run-id <ID2>
python -m src.leaderboard.merge_and_publish --run-id <ID3>
python -m src.leaderboard.merge_and_publish --run-id <ID4>
```

> **Note:** Two-card configs (`vllm_two_card`) need adjacent GPUs. Check the
> config's `vllm.provider_config` before parallelising — don't put two
> two-card models on the same pair.

---

## Phase 3 — Models without configs (need new configs)

These models are on the leaderboard but have no matching config in
`configs/models/`. Create a config for each using the `configure-model` skill,
then run the same way as Phase 2.

| Model on leaderboard | Likely HF path | Notes |
|---|---|---|
| Qwen3-4B-Instruct-2507 | Qwen/Qwen3-4B-Instruct-2507 | newer Qwen3 variant |

To create a missing config:

```bash
# Minimal config template — save as configs/models/<name>.yaml
cat > configs/models/MyModel.yaml << 'EOF'
model:
  name: "org/ModelName"   # exact HuggingFace repo path

vllm:
  provider_config: "vllm_single_card"
  overrides: {}

tasks:
  - "latam_board"
EOF
```

Then validate before running:

```bash
benchy validate --config configs/models/MyModel.yaml
```

---

## Phase 4 — Verify

After all models are published:

```bash
# Count models now in the dataset
python3 -c "
from huggingface_hub import hf_hub_download
import json
p = hf_hub_download('mauroibz/leaderboard-results', 'leaderboard_table.json', repo_type='dataset')
data = json.load(open(p))
print(f'{len(data)} models on HF:')
for row in data: print(' ', row.get('full_model_name', row.get('model_name')))
"
```

Open latamboard.surus.lat and hard-refresh — all models should appear with
scores. The leaderboard fetches live from HF so no redeploy is needed.

---

## Tracking progress

The fastest way to see where you are mid-rebuild:

```bash
# Which runs have completed?
ls outputs/benchmark_outputs/ | sort

# Which are already published to HF?
ls outputs/publish/summaries/*_summary.json 2>/dev/null | wc -l
```

---

## If a run fails

Don't stop. Skip the failed model, keep going, come back to it:

```bash
# Check what went wrong
cat outputs/benchmark_outputs/<run_id>/<model>/run_outcome.json | python3 -m json.tool | grep -E '"status"|"reason"'

# Common fixes:
# - OOM: switch to two-card config or reduce batch size
# - API error: check HF_TOKEN, model access permissions
# - Task not found: model config may need --tasks override
```

Failed models won't overwrite good scores — `merge_and_publish` only adds/
updates, never deletes existing HF entries.
