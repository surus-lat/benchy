# Benchmark submission

## Models submitted

<!-- List every model in this submission -->
- [ ] `org/model-name` — brief description

## How to reproduce

```bash
# Run the benchmark (exact command(s) used)
benchy eval --config configs/models/<your-config>.yaml --tasks latam_board

# Package into this submission
python -m src.leaderboard.process_all --run-id <run_id>
python -m src.leaderboard.package_submission --run-id <run_id> --skip-process
```

## Checklist

- [ ] `run_outcome.json` for each model shows `status: passed` or `degraded`
- [ ] Model config(s) in `submissions/<run_id>/configs/` are the exact configs used
- [ ] All models were evaluated on the full `latam_board` task group (spanish + portuguese + translation + structured_extraction)
- [ ] No cherry-picking: if multiple runs exist for the same model, this PR includes the most recent one

## Notes

<!-- Anything reviewers should know: hardware used, any degraded tasks, known issues -->
