#!/usr/bin/env python3
"""
Package a completed benchy run into submissions/<run_id>/ ready for a PR.

Reads processed summaries from outputs/publish/summaries/ (run process_all first)
and organises them into the canonical submission structure:

  submissions/<run_id>/
    run_manifest.json          — metadata tying the whole batch together
    models/
      <model_name>/
        run_outcome.json       — proof of execution
        model_summary.json     — processed leaderboard entry
    configs/
      <model>.yaml             — model config (reprex)
    README.md                  — exact commands to reproduce

Usage:
  python -m src.leaderboard.package_submission --run-id <run_id>
  python -m src.leaderboard.package_submission --run-id <run_id> --skip-process
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_src_dir = Path(__file__).resolve().parents[1]
_project_root = _src_dir.parent

sys.path.insert(0, str(_src_dir.parent))
from src.config_loader import load_config


def _find_model_config(full_model_name: str) -> Path | None:
    """Find the configs/models/*.yaml whose model.name matches full_model_name."""
    import yaml
    configs_dir = _project_root / "configs" / "models"
    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text())
            if data.get("model", {}).get("name") == full_model_name:
                return yaml_file
        except Exception:
            pass
    return None


def _read_full_model_name(model_dir: Path) -> str | None:
    """Read the full HF model name from run_outcome.json."""
    outcome = model_dir / "run_outcome.json"
    if outcome.exists():
        try:
            return json.loads(outcome.read_text()).get("model")
        except Exception:
            pass
    return None


def _build_manifest(run_id: str, model_entries: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "packaged_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": [e["full_model_name"] for e in model_entries if e.get("full_model_name")],
        "model_count": len(model_entries),
    }


def _write_readme(submission_dir: Path, run_id: str, model_entries: list[dict]) -> None:
    configs = [e["config_file"] for e in model_entries if e.get("config_file")]
    config_lines = "\n".join(
        f"  benchy eval --config configs/models/{Path(c).name} --tasks latam_board"
        for c in configs
    ) or "  # (model configs not found in configs/models/ — add them)"

    process_line = f"  python -m src.leaderboard.process_all --run-id {run_id}"

    readme = f"""# Submission: {run_id}

## Models evaluated ({len(model_entries)})

{chr(10).join(f'- {e.get("full_model_name", e["model_name"])}' for e in model_entries)}

## How to reproduce

```bash
# 1. Run the benchmarks
{config_lines}

# 2. Package the results (generates this submission directory)
python -m src.leaderboard.process_all --run-id {run_id}
python -m src.leaderboard.package_submission --run-id {run_id} --skip-process
```

## Validation checklist

- [ ] Each model's `run_outcome.json` shows `status: passed` (or `degraded`)
- [ ] Model configs in `configs/` are the exact configs used for the run
- [ ] Scores are consistent with what `benchy eval` printed to stdout
"""
    (submission_dir / "README.md").write_text(readme)


def package_submission(run_id: str, skip_process: bool = False) -> Path:
    config = load_config()
    benchmark_dir = Path(config["paths"]["benchmark_outputs"]) / run_id
    summaries_dir = Path(config["paths"]["publish_dir"]) / "summaries"

    if not benchmark_dir.exists():
        print(f"❌ Benchmark output directory not found: {benchmark_dir}")
        sys.exit(1)

    if not skip_process:
        print(f"\n▶ Running process_all for {run_id} ...")
        result = subprocess.run(
            [sys.executable, "-m", "src.leaderboard.process_all", run_id],
            cwd=str(_project_root),
        )
        if result.returncode != 0:
            print("❌ process_all failed — aborting")
            sys.exit(1)

    submission_dir = _project_root / "submissions" / run_id
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "models").mkdir(exist_ok=True)
    (submission_dir / "configs").mkdir(exist_ok=True)

    model_entries = []
    for model_dir in sorted(d for d in benchmark_dir.iterdir() if d.is_dir()):
        model_name = model_dir.name
        full_model_name = _read_full_model_name(model_dir) or model_name

        entry: dict = {"model_name": model_name, "full_model_name": full_model_name}

        # Copy run_outcome.json
        outcome_src = model_dir / "run_outcome.json"
        if outcome_src.exists():
            dest = submission_dir / "models" / model_name
            dest.mkdir(exist_ok=True)
            shutil.copy2(outcome_src, dest / "run_outcome.json")
        else:
            print(f"  ⚠  No run_outcome.json for {model_name}")

        # Copy model_summary.json
        summary_src = summaries_dir / f"{model_name}_summary.json"
        if summary_src.exists():
            dest = submission_dir / "models" / model_name
            dest.mkdir(exist_ok=True)
            shutil.copy2(summary_src, dest / "model_summary.json")
        else:
            print(f"  ⚠  No summary for {model_name} in {summaries_dir}")

        # Copy model config
        config_file = _find_model_config(full_model_name)
        if config_file:
            shutil.copy2(config_file, submission_dir / "configs" / config_file.name)
            entry["config_file"] = config_file.name
        else:
            print(f"  ⚠  No config found for {full_model_name}")

        model_entries.append(entry)
        print(f"  ✓ {full_model_name}")

    # Write manifest and README
    manifest = _build_manifest(run_id, model_entries)
    (submission_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_readme(submission_dir, run_id, model_entries)

    print(f"\n✅ Submission packaged → {submission_dir.relative_to(_project_root)}")
    print(f"   {len(model_entries)} model(s): {', '.join(e['model_name'] for e in model_entries)}")
    print(f"\nNext steps:")
    print(f"  git add submissions/{run_id}")
    print(f"  git commit -m 'submission: {run_id} ({len(model_entries)} models)'")
    print(f"  git push && open a PR")
    return submission_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Package a benchy run into submissions/")
    parser.add_argument("--run-id", required=True, help="Run ID to package")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip process_all (summaries already exist)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    package_submission(args.run_id, skip_process=args.skip_process)
