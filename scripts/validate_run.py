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
    """
    Load and parse a JSON outcome file from the given filesystem path.
    
    Parameters:
        outcome_path (Path): Path to the JSON file containing the run outcome (e.g., run_outcome.json).
    
    Returns:
        dict[str, Any]: The parsed JSON content as a dictionary.
    """
    with open(outcome_path) as f:
        return json.load(f)


def find_outcome_path(run_id: str, model_name: str | None, base_path: Path) -> Path:
    """
    Locate the run_outcome.json file for a specified run and optional model under a base outputs directory.
    
    Parameters:
        run_id (str): Identifier of the run directory to search.
        model_name (str | None): If provided, the specific model subdirectory to use; if None, the function selects the single model subdirectory when exactly one exists.
        base_path (Path): Base directory containing run subdirectories.
    
    Returns:
        Path: Path to the run_outcome.json file for the resolved model.
    
    Raises:
        FileNotFoundError: If the run directory does not exist; if no model subdirectories exist; if multiple model subdirectories exist and no model_name was given; or if run_outcome.json is missing in the resolved model directory.
    """
    run_dir = base_path / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if model_name:
        model_dir = run_dir / model_name
    else:
        candidates = [d for d in run_dir.iterdir() if d.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No model sub-directories in {run_dir}")
        if len(candidates) == 1:
            model_dir = candidates[0]
        else:
            candidate_names = ", ".join(sorted(d.name for d in candidates))
            raise FileNotFoundError(
                "Multiple model sub-directories found for run "
                f"'{run_id}': {candidate_names}. "
                "Pass the desired model name with --model."
            )

    path = model_dir / "run_outcome.json"
    if not path.exists():
        raise FileNotFoundError(f"run_outcome.json not found: {path}")
    return path


def validate_smoke_gates(outcome: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a benchmark run outcome against the AGENTS.md smoke gate rules.
    
    Checks that run status and exit code meet expectations, verifies that all blocking count
    fields are zero, and collects details for tasks that failed the smoke gates.
    
    Parameters:
        outcome (dict): Parsed run outcome (e.g., contents of run_outcome.json).
    
    Returns:
        result (dict): Structured validation result containing:
            passed (bool): True if no smoke gate violations were found.
            run_id (str|None): The run identifier from the outcome.
            model (str|None): The model name from the outcome.
            status (str): The run status from the outcome.
            exit_code (int): The run exit code from the outcome.
            exit_policy (Any): The exit policy from the outcome, if present.
            duration_s (Any): The run duration in seconds from the outcome, if present.
            counts (dict): The counts object from the outcome.
            violations (list[str]): Human-readable descriptions of any smoke gate violations.
            failed_task_details (dict): Mapping of failing task name to a dict with `status` and `reason`.
            errors (list): The outcome's errors list, if any.
    """
    status = outcome.get("status", "unknown")
    exit_code = outcome.get("exit_code", 0) or 0
    counts = outcome.get("counts", {})
    violations: list[str] = []

    if status not in VALID_STATUSES:
        violations.append(f"status is '{status}' (expected: passed or degraded)")

    if exit_code != 0:
        violations.append(f"exit_code = {exit_code} (expected: 0)")

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
        "exit_code": exit_code,
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
    """
    Determine the default outputs path for benchmark run artifacts.
    
    Attempts to read "paths.benchmark_outputs" from configs/config.yaml and returns it if present; if the config file is missing or cannot be parsed, returns Path("outputs/benchmark_outputs").
    
    Returns:
        Path: The resolved outputs directory path from config or the fallback "outputs/benchmark_outputs".
    """
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
    """
    Parse command-line arguments, load a benchy run outcome, validate it against the AGENTS.md smoke gate contract, emit a structured JSON report to stdout and an optional human-readable summary to stderr, and return an appropriate exit code.
    
    The CLI accepts either --outcome-path or --run-id (with optional --model and --output-path). If the outcome cannot be found or loaded, a JSON error object is printed to stdout and the function exits with a non-zero code.
    
    Returns:
        int: 0 if smoke gates pass; 1 otherwise (including failure to load the outcome or any validation failures).
    """
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
