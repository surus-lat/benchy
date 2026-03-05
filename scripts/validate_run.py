#!/usr/bin/env python3
"""Validate a benchy run against the AGENTS.md smoke gate contract.

Exits 0 if the run passes all smoke gates, non-zero otherwise.
Prints a structured JSON report to stdout.

Usage:
    python scripts/validate_run.py --run-id <id>
    python scripts/validate_run.py --run-id <id> --model <name>
    python scripts/validate_run.py --outcome-path /path/to/run_outcome.json

Can be wired into an agent Stop hook or CI pipeline step:
    benchy eval ... && python scripts/validate_run.py --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Smoke gate logic (mirrors AGENTS.md contract)
# ---------------------------------------------------------------------------

BLOCKING_COUNT_KEYS = [
    "failed_tasks",
    "error_tasks",
    "pending_tasks",
    "no_samples_tasks",
    "skipped_tasks",
]

VALID_STATUSES = {"passed", "degraded"}


def load_outcome(outcome_path: Path) -> dict[str, Any]:
    with open(outcome_path) as f:
        return json.load(f)


def find_outcome_path(run_id: str, model_name: str | None, base_path: Path) -> Path:
    run_dir = base_path / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if model_name:
        model_dir = run_dir / model_name
    else:
        candidates = [d for d in run_dir.iterdir() if d.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No model sub-directories in {run_dir}")
        model_dir = sorted(candidates)[0]

    path = model_dir / "run_outcome.json"
    if not path.exists():
        raise FileNotFoundError(f"run_outcome.json not found: {path}")
    return path


def validate_smoke_gates(outcome: dict[str, Any]) -> dict[str, Any]:
    """Apply AGENTS.md smoke gates and return a structured result."""
    status = outcome.get("status", "unknown")
    counts = outcome.get("counts", {})
    violations: list[str] = []

    if status not in VALID_STATUSES:
        violations.append(f"status is '{status}' (expected: passed or degraded)")

    for key in BLOCKING_COUNT_KEYS:
        val = counts.get(key, 0) or 0
        if val > 0:
            violations.append(f"{key} = {val} (expected: 0)")

    failed_tasks = {
        name: info
        for name, info in (outcome.get("tasks") or {}).items()
        if info.get("status") not in VALID_STATUSES | {"skipped", "no_samples"}
    }

    return {
        "passed": len(violations) == 0,
        "run_id": outcome.get("run_id"),
        "model": outcome.get("model"),
        "status": status,
        "exit_policy": outcome.get("exit_policy"),
        "duration_s": outcome.get("duration_s"),
        "counts": counts,
        "violations": violations,
        "failed_task_details": {
            name: {"status": info.get("status"), "reason": info.get("reason")}
            for name, info in failed_tasks.items()
        },
        "errors": outcome.get("errors", []),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _default_output_path() -> Path:
    try:
        import yaml

        cfg = Path("configs/config.yaml")
        if cfg.exists():
            with open(cfg) as f:
                data = yaml.safe_load(f)
            return Path(data.get("paths", {}).get("benchmark_outputs", "outputs/benchmark_outputs"))
    except Exception:
        pass
    return Path("outputs/benchmark_outputs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a benchy run against the AGENTS.md smoke gate contract."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, help="Run ID to validate")
    group.add_argument("--outcome-path", type=str, help="Direct path to run_outcome.json")

    parser.add_argument("--model", type=str, default=None, help="Model sub-folder name (optional)")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Base output path (default: from configs/config.yaml)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only output JSON, no human summary")
    args = parser.parse_args()

    try:
        if args.outcome_path:
            outcome_path = Path(args.outcome_path)
        else:
            base_path = Path(args.output_path) if args.output_path else _default_output_path()
            outcome_path = find_outcome_path(args.run_id, args.model, base_path)

        outcome = load_outcome(outcome_path)
    except FileNotFoundError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        return 1
    except Exception as exc:
        print(json.dumps({"passed": False, "error": f"Failed to load outcome: {exc}"}))
        return 1

    result = validate_smoke_gates(outcome)
    print(json.dumps(result, indent=2))

    if not args.quiet:
        if result["passed"]:
            print("\n✓ Smoke gates PASSED — safe to proceed with full run.", file=sys.stderr)
        else:
            print("\n✗ Smoke gates FAILED:", file=sys.stderr)
            for v in result["violations"]:
                print(f"  - {v}", file=sys.stderr)
            if result["failed_task_details"]:
                print("  Failed tasks:", file=sys.stderr)
                for name, info in result["failed_task_details"].items():
                    print(f"    {name}: {info['status']} ({info.get('reason')})", file=sys.stderr)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
