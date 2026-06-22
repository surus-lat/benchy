# Submission: 20260619_013834_LIMITED

## Models evaluated (1)

- openai/whisper-large-v3-turbo

## How to reproduce

```bash
# 1. Run the benchmarks
  benchy eval --config configs/models/whisper-large-v3-turbo-transformers.yaml --tasks latam_board

# 2. Package the results (generates this submission directory)
python -m src.leaderboard.process_all --run-id 20260619_013834_LIMITED
python -m src.leaderboard.package_submission --run-id 20260619_013834_LIMITED --skip-process
```

## Validation checklist

- [ ] Each model's `run_outcome.json` shows `status: passed` (or `degraded`)
- [ ] Model configs in `configs/` are the exact configs used for the run
- [ ] Scores are consistent with what `benchy eval` printed to stdout
