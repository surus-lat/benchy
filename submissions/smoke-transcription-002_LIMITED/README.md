# Submission: smoke-transcription-002_LIMITED

## Models evaluated (1)

- Systran/faster-whisper-small

## How to reproduce

```bash
# 1. Run the benchmarks
  # (model configs not found in configs/models/ — add them)

# 2. Package the results (generates this submission directory)
python -m src.leaderboard.process_all --run-id smoke-transcription-002_LIMITED
python -m src.leaderboard.package_submission --run-id smoke-transcription-002_LIMITED --skip-process
```

## Validation checklist

- [ ] Each model's `run_outcome.json` shows `status: passed` (or `degraded`)
- [ ] Model configs in `configs/` are the exact configs used for the run
- [ ] Scores are consistent with what `benchy eval` printed to stdout
