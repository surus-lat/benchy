# Submission: latam_asr_es_full

## Models evaluated (4)

- Qwen/Qwen3-ASR-0.6B
- nvidia/canary-1b-flash
- openai/whisper-large-v3-turbo
- surus-ai/whisper-large-v3-turbo-latam

## How to reproduce

```bash
# 1. Run the benchmarks
  benchy eval --config configs/models/qwen3-asr-0.6b-transformers.yaml --tasks latam_board
  benchy eval --config configs/models/canary-1b-flash-transformers.yaml --tasks latam_board
  benchy eval --config configs/models/whisper-large-v3-turbo-transformers.yaml --tasks latam_board
  benchy eval --config configs/models/surus-whisper-large-v3-turbo-latam-transformers.yaml --tasks latam_board

# 2. Package the results (generates this submission directory)
python -m src.leaderboard.process_all --run-id latam_asr_es_full
python -m src.leaderboard.package_submission --run-id latam_asr_es_full --skip-process
```

## Validation checklist

- [ ] Each model's `run_outcome.json` shows `status: passed` (or `degraded`)
- [ ] Model configs in `configs/` are the exact configs used for the run
- [ ] Scores are consistent with what `benchy eval` printed to stdout
