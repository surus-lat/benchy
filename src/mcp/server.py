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
    """
    Resolve the base filesystem path for benchmark outputs.
    
    If the module-level _DEFAULT_OUTPUT_PATH is set, that value is used. Otherwise the function attempts to read configs/config.yaml and use the value at `paths.benchmark_outputs`; if that file or key is unavailable it falls back to "outputs/benchmark_outputs".
    
    Returns:
        Path: The resolved benchmark outputs directory path.
    """
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


def _validate_relative_component(value: str, label: str) -> None:
    """
    Validate that a string represents a single, relative path segment suitable for constructing filesystem paths.
    
    Parameters:
        value (str): The path component to validate.
        label (str): A descriptive name used in error messages when validation fails.
    
    Raises:
        ValueError: If `value` is an absolute path, contains more than one path part, or contains the parent-directory segment `".."`.
    """
    path_value = Path(value)
    if path_value.is_absolute():
        raise ValueError(f"{label} must be a relative path segment (absolute paths are not allowed)")
    if len(path_value.parts) != 1:
        raise ValueError(f"{label} must be a single path segment")
    if any(part == ".." for part in path_value.parts):
        raise ValueError(f"{label} must not contain '..'")


def _is_relative_to(path: Path, base: Path) -> bool:
    """
    Determine whether a Path is relative to a given base path.
    
    Returns:
        True if `path` is inside `base`, False otherwise.
    """
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _resolve_run_dir(run_id: str) -> tuple[Path, Path]:
    """
    Resolve and validate the filesystem paths for a given run identifier.
    
    Parameters:
        run_id (str): A single-segment relative run identifier (no nested paths or ".."); it will be validated.
    
    Returns:
        tuple[Path, Path]: A pair (output_root, resolved_run_dir) where `output_root` is the resolved benchmark outputs root and `resolved_run_dir` is the resolved path to the run within that root.
    
    Raises:
        ValueError: If the resolved run directory is outside the benchmark output root.
    """
    _validate_relative_component(run_id, "run_id")
    output_root = _get_output_path().resolve()
    resolved_run_dir = (output_root / run_id).resolve()
    if not _is_relative_to(resolved_run_dir, output_root):
        raise ValueError("run_id resolves outside the benchmark output root")
    return output_root, resolved_run_dir


def _resolve_model_dir(output_root: Path, run_dir: Path, model_name: str | None) -> Path:
    """
    Resolve the filesystem directory for a model within a run, enforcing that it stays under the configured outputs root.
    
    If `model_name` is provided, validate it as a single relative path component and return the corresponding subdirectory under `run_dir`. If `model_name` is None, discover model subdirectories directly under `run_dir`: return the single candidate if exactly one exists, raise if none or if multiple candidates require disambiguation.
    
    Parameters:
        output_root (Path): The configured base outputs directory; used to ensure the resolved model directory is contained under the outputs root.
        run_dir (Path): The resolved run directory under which model subdirectories reside.
        model_name (str | None): Optional single-segment relative model name to select a specific model subdirectory.
    
    Returns:
        Path: The resolved model directory path.
    
    Raises:
        ValueError: If a provided `model_name` resolves outside `run_dir` or `output_root`.
        FileNotFoundError: If no model subdirectories are found under `run_dir`, or if multiple candidates exist and `model_name` was not provided.
    """
    if model_name:
        _validate_relative_component(model_name, "model_name")
        resolved_model_dir = (run_dir / model_name).resolve()
        if not _is_relative_to(resolved_model_dir, run_dir) or not _is_relative_to(
            resolved_model_dir, output_root
        ):
            raise ValueError("model_name resolves outside the run directory")
        return resolved_model_dir

    candidates = [d.resolve() for d in run_dir.iterdir() if d.is_dir()]
    candidates = [
        d
        for d in candidates
        if _is_relative_to(d, run_dir) and _is_relative_to(d, output_root)
    ]
    if not candidates:
        raise FileNotFoundError(f"No model sub-directories found in {run_dir}")
    if len(candidates) == 1:
        return candidates[0]

    candidate_names = ", ".join(sorted(d.name for d in candidates))
    raise FileNotFoundError(
        "Multiple model sub-directories found. "
        f"Candidates: {candidate_names}. "
        "Pass model_name to disambiguate."
    )


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
        model_name: Optional model name segment. If omitted, exactly one model
                    folder must exist under run_id.

    Returns:
        The parsed run_outcome.json dict, or an error dict if not found.
    """
    try:
        output_root, base = _resolve_run_dir(run_id)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    if not base.exists():
        return {"error": f"Run directory not found: {base}"}

    try:
        model_dir = _resolve_model_dir(output_root, base, model_name)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

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
    """
    Determine whether a specified run satisfies the smoke-gate criteria defined for Benchy evaluations.
    
    Parameters:
        run_id (str): Identifier of the run directory to evaluate (a single relative path component).
        model_name (str | None): Optional model subdirectory name to target within the run.
    
    Returns:
        dict: Result object with the following keys:
            - passed (bool): `true` if no smoke-gate violations were found, `false` otherwise.
            - status (str): The run outcome status (e.g., "passed", "degraded", "failed", or "unknown").
            - exit_code (int): The run's exit code (defaults to 0 when not present).
            - counts (dict): Counts object taken from the run outcome (may include keys like `failed_tasks`, `error_tasks`, etc.).
            - violations (list[str]): Human-readable descriptions of any smoke-gate violations.
            - error (str, optional): Present when the run or model could not be resolved or read; in that case `passed` will be `false`.
    """
    outcome = read_run_outcome(run_id, model_name)

    if "error" in outcome:
        return {"passed": False, "error": outcome["error"]}

    status = outcome.get("status", "unknown")
    exit_code = outcome.get("exit_code", 0) or 0
    counts = outcome.get("counts", {})

    violations: list[str] = []

    if status not in {"passed", "degraded"}:
        violations.append(f"run status is '{status}' (expected: passed or degraded)")

    if exit_code != 0:
        violations.append(f"exit_code = {exit_code} (expected: 0)")

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
        "exit_code": exit_code,
        "counts": counts,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Tool: list_runs
# ---------------------------------------------------------------------------


@mcp.tool()
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """
    Return a list of recent runs (most recent first) with basic metadata for each run.
    
    Parameters:
        limit (int): Maximum number of run entries to return.
    
    Returns:
        list[dict]: Ordered list of run summary objects. Each object contains:
            - "run_id": run directory name.
            - "model": model name (from outcome or directory).
            - "status": run status string (if available).
            - "duration_s": run duration in seconds (if available).
            - "ended_at": ISO timestamp when the run ended (if available).
            - "counts": counts summary object from the outcome (if available).
            - "error": error message string if the run's outcome file could not be read.
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
    """
    List available Benchy configuration names grouped by category.
    
    Parameters:
        kind (str): One of "models", "systems", "tests", "providers", or "all". When a specific
            category is provided, only that category is returned. If an unknown kind is given,
            an error mapping is returned under the "error" key.
    
    Returns:
        dict[str, list[str]]: Mapping from category name to a sorted list of configuration
        names (filename stems without the `.yaml` extension). Categories with no configs
        or missing directories yield an empty list.
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
    """
    List available Benchy evaluation tasks.
    
    If `group` is provided, only tasks whose `group` field equals that value are returned.
    
    Parameters:
        group (str | None): Optional task group name to filter by.
    
    Returns:
        dict: Mapping from task name to a dict with keys:
            - "group": the task's group (or None)
            - "description": the English description if present, otherwise the default description.
        If the reference file is missing, returns {"error": "<message>"}.
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
    """
    Retrieve the run_summary.json (compact metric table) for a completed run.
    
    Parameters:
        run_id: Identifier of the run directory to read.
        model_name: Optional model subdirectory name to target a specific model.
    
    Returns:
        dict: Parsed contents of run_summary.json, or a dict with an "error" key describing the failure.
    """
    try:
        output_root, base = _resolve_run_dir(run_id)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    if not base.exists():
        return {"error": f"Run directory not found: {base}"}

    try:
        model_dir = _resolve_model_dir(output_root, base, model_name)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

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
    """
    Retrieve samples for a specific task in a run, highlighting errored and incorrect predictions.
    
    Parameters:
        run_id (str): Run identifier (single path component).
        task_name (str): Task path; dots are treated as directory separators (e.g. "a.b" -> "a/b").
        model_name (str | None): Optional model subdirectory name to target.
        max_samples (int): Maximum number of error samples to include in the response.
    
    Returns:
        dict: On success, a dictionary with:
            - total_samples (int): Number of samples available for the task.
            - error_count (int): Number of samples that contain an error.
            - wrong_predictions (int): Number of samples where `prediction` != `expected` (and `expected` present).
            - error_samples (list): Up to `max_samples` sample objects that contain errors.
        If the run, model, task, or samples file cannot be found or resolved, returns
        {"error": "<message>"} describing the problem.
    """
    try:
        output_root, base = _resolve_run_dir(run_id)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    if not base.exists():
        return {"error": f"Run directory not found: {base}"}

    try:
        model_dir = _resolve_model_dir(output_root, base, model_name)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    # Support dotted subtask path (e.g. document_extraction.facturas_argentinas → document_extraction/facturas_argentinas)
    task_path = model_dir / task_name.replace(".", "/")

    if not task_path.exists():
        return {"error": f"Task directory not found: {task_path}"}

    # Find samples file
    samples_files = list(task_path.glob("*_samples.json"))
    if not samples_files:
        return {"error": f"No samples file found in {task_path}"}

    with open(samples_files[0]) as f:
        raw = json.load(f)

    # Samples file is wrapped: {"model": ..., "samples": [...]}
    if isinstance(raw, dict) and "samples" in raw:
        all_samples = raw["samples"]
    elif isinstance(raw, list):
        all_samples = raw
    else:
        all_samples = []

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
    """
    Start the Benchy MCP server, optionally overriding the module's default output path.
    
    If the `--output-path` command-line argument is provided, set the module-level `_DEFAULT_OUTPUT_PATH` to that value before launching the MCP server via `mcp.run()`.
    """
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
