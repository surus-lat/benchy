"""Benchy MCP Server.

Exposes benchy run outputs and configuration as MCP tools so AI agents
can read results, check smoke gates, and discover available configs and
tasks — without parsing human log output.

Usage:
    benchy-mcp                          # start the MCP server (stdio transport)
    benchy-mcp --output-path /my/path   # override default output base path

Install the optional dep first:
    pip install benchy[mcp]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'mcp' package is required to run the benchy MCP server.\n"
        "Install it with: pip install benchy[mcp]"
    ) from exc


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="benchy",
    instructions=(
        "Benchy MCP server. Provides read access to evaluation run outputs, "
        "configs, and task definitions. Use read_run_outcome and validate_smoke_gates "
        "to check whether a run passed the canonical smoke contract from AGENTS.md."
    ),
)

_DEFAULT_OUTPUT_PATH: str = ""  # resolved at startup from configs/config.yaml


def _get_output_path() -> Path:
    if _DEFAULT_OUTPUT_PATH:
        return Path(_DEFAULT_OUTPUT_PATH)
    # Fall back to project convention
    try:
        import yaml

        cfg_path = Path("configs/config.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            return Path(cfg.get("paths", {}).get("benchmark_outputs", "outputs/benchmark_outputs"))
    except Exception:
        pass
    return Path("outputs/benchmark_outputs")


# ---------------------------------------------------------------------------
# Tool: read_run_outcome
# ---------------------------------------------------------------------------


@mcp.tool()
def read_run_outcome(
    run_id: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Read run_outcome.json for a completed benchy eval run.

    Args:
        run_id: The run ID (folder name under the output base path).
        model_name: Optional model name segment. If omitted, the first model
                    folder found under run_id is used.

    Returns:
        The parsed run_outcome.json dict, or an error dict if not found.
    """
    base = _get_output_path() / run_id

    if not base.exists():
        return {"error": f"Run directory not found: {base}"}

    # Find the model sub-folder
    if model_name:
        model_dir = base / model_name
    else:
        candidates = [d for d in base.iterdir() if d.is_dir()]
        if not candidates:
            return {"error": f"No model sub-directories found in {base}"}
        model_dir = candidates[0]

    outcome_path = model_dir / "run_outcome.json"
    if not outcome_path.exists():
        return {"error": f"run_outcome.json not found: {outcome_path}"}

    with open(outcome_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tool: validate_smoke_gates
# ---------------------------------------------------------------------------


@mcp.tool()
def validate_smoke_gates(
    run_id: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Check whether a run passes the AGENTS.md smoke gates.

    A run passes if:
    - run_outcome.status is 'passed' or 'degraded'
    - counts.failed_tasks == 0
    - counts.error_tasks == 0
    - counts.pending_tasks == 0
    - counts.no_samples_tasks == 0
    - counts.skipped_tasks == 0

    Returns a dict with 'passed' (bool), 'status', 'counts', and 'violations'.
    """
    outcome = read_run_outcome(run_id, model_name)

    if "error" in outcome:
        return {"passed": False, "error": outcome["error"]}

    status = outcome.get("status", "unknown")
    counts = outcome.get("counts", {})

    violations: list[str] = []

    if status not in {"passed", "degraded"}:
        violations.append(f"run status is '{status}' (expected: passed or degraded)")

    blocking_counts = [
        "failed_tasks",
        "error_tasks",
        "pending_tasks",
        "no_samples_tasks",
        "skipped_tasks",
    ]
    for key in blocking_counts:
        val = counts.get(key, 0)
        if val and val > 0:
            violations.append(f"{key} = {val} (expected: 0)")

    return {
        "passed": len(violations) == 0,
        "status": status,
        "counts": counts,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Tool: list_runs
# ---------------------------------------------------------------------------


@mcp.tool()
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """List recent benchy evaluation runs with their status.

    Args:
        limit: Maximum number of runs to return (most recent first).

    Returns:
        List of dicts with run_id, model, status, duration_s, ended_at.
    """
    base = _get_output_path()
    if not base.exists():
        return []

    runs: list[dict[str, Any]] = []

    for run_dir in sorted(base.iterdir(), key=lambda d: d.name, reverse=True):
        if not run_dir.is_dir():
            continue
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            outcome_path = model_dir / "run_outcome.json"
            if outcome_path.exists():
                try:
                    with open(outcome_path) as f:
                        data = json.load(f)
                    runs.append(
                        {
                            "run_id": run_dir.name,
                            "model": data.get("model", model_dir.name),
                            "status": data.get("status"),
                            "duration_s": data.get("duration_s"),
                            "ended_at": data.get("ended_at"),
                            "counts": data.get("counts"),
                        }
                    )
                except Exception as exc:
                    runs.append({"run_id": run_dir.name, "model": model_dir.name, "error": str(exc)})
            if len(runs) >= limit:
                break
        if len(runs) >= limit:
            break

    return runs


# ---------------------------------------------------------------------------
# Tool: list_configs
# ---------------------------------------------------------------------------


@mcp.tool()
def list_configs(kind: str = "all") -> dict[str, list[str]]:
    """List available benchy configuration files.

    Args:
        kind: One of 'models', 'systems', 'tests', 'providers', or 'all'.

    Returns:
        Dict mapping kind -> list of config names (without .yaml).
    """
    dirs = {
        "models": Path("configs/models"),
        "systems": Path("configs/systems"),
        "tests": Path("configs/tests"),
        "providers": Path("configs/providers"),
    }

    if kind != "all" and kind not in dirs:
        return {"error": [f"Unknown kind '{kind}'. Use: models, systems, tests, providers, all"]}

    result: dict[str, list[str]] = {}
    targets = dirs if kind == "all" else {kind: dirs[kind]}

    for category, path in targets.items():
        if path.exists():
            result[category] = sorted(
                p.stem for p in path.glob("*.yaml") if not p.stem.startswith("_")
            )
        else:
            result[category] = []

    return result


# ---------------------------------------------------------------------------
# Tool: list_tasks
# ---------------------------------------------------------------------------


@mcp.tool()
def list_tasks(group: str | None = None) -> dict[str, Any]:
    """List available benchy evaluation tasks.

    Args:
        group: Optional task group name to filter by (e.g. 'latam_board').
               If omitted, all tasks are returned.

    Returns:
        Dict with task names as keys and metadata (group, description) as values.
    """
    ref_path = Path("reference/tasks_list.json")
    if not ref_path.exists():
        return {"error": "reference/tasks_list.json not found"}

    with open(ref_path) as f:
        data = json.load(f)

    tasks: dict[str, Any] = data.get("tasks", {})

    if group:
        tasks = {k: v for k, v in tasks.items() if v.get("group") == group}

    # Return a compact representation
    return {
        name: {
            "group": info.get("group"),
            "description": info.get("description_en", info.get("description")),
        }
        for name, info in tasks.items()
    }


# ---------------------------------------------------------------------------
# Tool: read_run_summary
# ---------------------------------------------------------------------------


@mcp.tool()
def read_run_summary(
    run_id: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Read run_summary.json (compact metric table) for a completed run.

    Args:
        run_id: The run ID.
        model_name: Optional model name segment.

    Returns:
        The parsed run_summary.json dict, or an error dict if not found.
    """
    base = _get_output_path() / run_id

    if not base.exists():
        return {"error": f"Run directory not found: {base}"}

    if model_name:
        model_dir = base / model_name
    else:
        candidates = [d for d in base.iterdir() if d.is_dir()]
        if not candidates:
            return {"error": f"No model sub-directories found in {base}"}
        model_dir = candidates[0]

    summary_path = model_dir / "run_summary.json"
    if not summary_path.exists():
        return {"error": f"run_summary.json not found: {summary_path}"}

    with open(summary_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tool: get_task_errors
# ---------------------------------------------------------------------------


@mcp.tool()
def get_task_errors(
    run_id: str,
    task_name: str,
    model_name: str | None = None,
    max_samples: int = 10,
) -> dict[str, Any]:
    """Get failed/errored samples from a specific task in a run.

    Useful for diagnosing why a task failed or is degraded.

    Args:
        run_id: The run ID.
        task_name: Task or subtask name (e.g. 'spanish' or 'image_extraction.facturas').
        model_name: Optional model name segment.
        max_samples: Maximum number of error samples to return.

    Returns:
        Dict with error_count, samples (failed ones up to max_samples), and task_status.
    """
    base = _get_output_path() / run_id

    if not base.exists():
        return {"error": f"Run directory not found: {base}"}

    if model_name:
        model_dir = base / model_name
    else:
        candidates = [d for d in base.iterdir() if d.is_dir()]
        if not candidates:
            return {"error": "No model sub-directories found"}
        model_dir = candidates[0]

    # Support dotted subtask path (e.g. image_extraction.facturas → image_extraction/facturas)
    task_path = model_dir / task_name.replace(".", "/")

    if not task_path.exists():
        return {"error": f"Task directory not found: {task_path}"}

    # Find samples file
    samples_files = list(task_path.glob("*_samples.json"))
    if not samples_files:
        return {"error": f"No samples file found in {task_path}"}

    with open(samples_files[0]) as f:
        all_samples = json.load(f)

    error_samples = [s for s in all_samples if s.get("error") or s.get("error_type")]
    wrong_samples = [
        s for s in all_samples
        if not s.get("error") and s.get("prediction") != s.get("expected") and "expected" in s
    ]

    return {
        "total_samples": len(all_samples),
        "error_count": len(error_samples),
        "wrong_predictions": len(wrong_samples),
        "error_samples": error_samples[:max_samples],
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchy MCP server")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override benchmark output base path (default: from configs/config.yaml)",
    )
    args, _ = parser.parse_known_args()

    global _DEFAULT_OUTPUT_PATH
    if args.output_path:
        _DEFAULT_OUTPUT_PATH = args.output_path

    mcp.run()


if __name__ == "__main__":
    main()
