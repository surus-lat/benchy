"""Main Prefect pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from importlib.metadata import version as package_version, PackageNotFoundError
from .prefect_compat import flow, task, NO_CACHE
from .inference.vllm_server import start_vllm_server, test_vllm_api, stop_vllm_server
from .inference.vllm_config import VLLMServerConfig
from .config_loader import load_config
from .config_manager import ConfigManager
from .generation_config import fetch_generation_config, save_generation_config
from .gpu_config import load_gpu_config
from .outcome import (
    resolve_exit_code,
    resolve_run_status,
)
from .signal_utils import clear_active_server_info, set_active_server_info
from .task_completion_checker import TaskCompletionChecker
from .tasks.registry import (
    is_handler_based_task,
    build_handler_task_config,
    discover_and_run_handler_task,
)
import os
import sys
import logging
import yaml
import json
import traceback
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


SUMMARY_SKIP_KEYS = {
    "total_samples",
    "valid_samples",
    "error_count",
    "throughput",
    "total_duration",
    "samples_with_type_errors",
    "numeric_fields_total",
    "numeric_fields_correct",
    "imperfect_sample_count",
    "match_distribution_counts",
    "subtasks",
}
SUMMARY_INCLUDE_DICT_KEYS = {
    "diagnostic_counts",
    "diagnostic_rates",
}
RUN_OUTCOME_SCHEMA_VERSION = "1.0"


def _normalize_task_key(task_key: str) -> str:
    """Canonicalize task keys so `group.sub-task` and `group.sub_task` align."""
    parts = (task_key or "").split(".", 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1].replace('-', '_')}"
    return task_key or ""


def _get_benchy_version() -> str:
    """Best-effort package version lookup."""
    try:
        return package_version("benchy")
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def _artifact_ref(path: Optional[str]) -> Dict[str, Any]:
    """Build normalized artifact reference payload."""
    if not path:
        return {"path": None, "exists": False}
    candidate = Path(path)
    return {"path": str(candidate), "exists": candidate.exists()}


def _build_artifacts(
    *,
    model_output_path: str,
    log_file_path: Optional[str],
) -> Dict[str, Any]:
    """Build run artifact index for automation clients."""
    model_root = Path(model_output_path)
    return {
        "model_output_dir": _artifact_ref(str(model_root)),
        "run_outcome": _artifact_ref(str(model_root / "run_outcome.json")),
        "run_summary": _artifact_ref(str(model_root / "run_summary.json")),
        "probe_report": _artifact_ref(str(model_root / "probe_report.json")),
        "probe_summary": _artifact_ref(str(model_root / "probe_summary.txt")),
        "log_file": _artifact_ref(log_file_path),
        "task_status_glob": str(model_root / "*/task_status.json"),
    }


def _get_git_metadata() -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""
    try:
        repo = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "available": True,
            "repo": repo,
            "commit": commit,
            "dirty": bool(dirty_status),
        }
    except Exception:
        return {"available": False}


@task(cache_policy=NO_CACHE)
def _run_logged_task(
    *,
    task_name: str,
    model_name: str,
    model_output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Optional[Dict[str, Any]],
    task_config: Dict[str, Any],
    limit: Optional[int],
    cuda_devices: Optional[str],
    provider_type: str,
    provider_config: Optional[Dict[str, Any]],
    api_endpoint: str,
    generation_config: Optional[Dict[str, Any]],
    compatibility_mode: str,
) -> Dict[str, Any]:
    """Wrapper so each benchy task shows up as a Prefect task run."""
    if not is_handler_based_task(task_name):
        raise ValueError(f"Unknown task '{task_name}'. Only handler-based tasks are supported.")

    return discover_and_run_handler_task(
        task_ref=task_name,
        model_name=model_name,
        output_path=model_output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
        compatibility_mode=compatibility_mode,
    )



def _summarize_task_metrics(
    metrics: Dict[str, Any],
    metrics_manifest: Optional[list] = None,
) -> Dict[str, Any]:
    """Filter aggregate metrics to numeric values excluding counts/metadata.
    
    For tasks with multiple subtasks, preserves the 'subtasks' structure.
    """
    # Check if this is a multi-subtask result
    if "subtasks" in metrics and isinstance(metrics["subtasks"], dict):
        # Recursively summarize each subtask
        return {
            "subtasks": {
                subtask_name: _summarize_single_task_metrics(subtask_metrics, metrics_manifest)
                for subtask_name, subtask_metrics in metrics["subtasks"].items()
            }
        }
    
    # Single task or flat metrics
    return _summarize_single_task_metrics(metrics, metrics_manifest)


def _summarize_single_task_metrics(
    metrics: Dict[str, Any],
    metrics_manifest: Optional[list] = None,
) -> Dict[str, Any]:
    """Summarize metrics for a single task/subtask."""
    summarized: Dict[str, Any] = {}
    if metrics_manifest:
        for key in metrics_manifest:
            value = metrics.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                summarized[key] = float(value)
        for key in SUMMARY_INCLUDE_DICT_KEYS:
            value = metrics.get(key)
            if isinstance(value, dict):
                summarized[key] = value
        return summarized

    for key, value in metrics.items():
        if key in SUMMARY_INCLUDE_DICT_KEYS and isinstance(value, dict):
            summarized[key] = value
            continue
        if key in SUMMARY_SKIP_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            summarized[key] = float(value)
    return summarized


def _aggregate_summary_diagnostics(tasks_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate diagnostic counts/rates from run_summary task metrics."""
    counts: Dict[str, int] = {}
    run_samples = 0

    def _walk(node: Any) -> None:
        nonlocal run_samples
        if not isinstance(node, dict):
            return
        diag_counts = node.get("diagnostic_counts")
        if isinstance(diag_counts, dict):
            for cls, value in diag_counts.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    counts[str(cls)] = counts.get(str(cls), 0) + int(value)
        sample_count = node.get("run_samples")
        if isinstance(sample_count, (int, float)) and not isinstance(sample_count, bool):
            run_samples += int(sample_count)

        subtasks = node.get("subtasks")
        if isinstance(subtasks, dict):
            for child in subtasks.values():
                _walk(child)

    for task_data in (tasks_payload or {}).values():
        _walk(task_data)

    rates: Dict[str, float] = {}
    if run_samples > 0:
        rates = {
            cls: round(count / float(run_samples), 6)
            for cls, count in counts.items()
        }

    return {
        "counts": counts,
        "run_samples": run_samples,
        "rates": rates,
    }


def _extract_task_diagnostics(summary_entry: Any) -> Dict[str, Any]:
    """Extract task-level diagnostics (including per-subtask diagnostics) from run_summary task data."""
    if not isinstance(summary_entry, dict):
        return {}

    diagnostics: Dict[str, Any] = {}
    for key in ("diagnostic_counts", "diagnostic_rates", "run_samples"):
        value = summary_entry.get(key)
        if isinstance(value, dict) or isinstance(value, (int, float)):
            diagnostics[key] = value

    subtasks = summary_entry.get("subtasks")
    if isinstance(subtasks, dict):
        subtask_diags: Dict[str, Any] = {}
        for subtask_name, subtask_summary in subtasks.items():
            extracted = _extract_task_diagnostics(subtask_summary)
            if extracted:
                subtask_diags[subtask_name] = extracted
        if subtask_diags:
            diagnostics["subtasks"] = subtask_diags

    return diagnostics


def _attach_per_task_diagnostics(
    task_records: Dict[str, Dict[str, Any]],
    run_summary_payload: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Attach per-task diagnostics from run_summary into run_outcome task records."""
    summary_tasks = {}
    if isinstance(run_summary_payload, dict):
        candidate = run_summary_payload.get("tasks")
        if isinstance(candidate, dict):
            summary_tasks = candidate

    canonical_summary = {
        _normalize_task_key(task_name): payload
        for task_name, payload in summary_tasks.items()
        if isinstance(task_name, str)
    }

    enriched: Dict[str, Dict[str, Any]] = {}
    for task_name, record in (task_records or {}).items():
        record_payload = dict(record or {})
        summary_entry = summary_tasks.get(task_name)
        if summary_entry is None:
            summary_entry = canonical_summary.get(_normalize_task_key(task_name))

        diagnostics = _extract_task_diagnostics(summary_entry)
        if diagnostics:
            record_payload["diagnostics"] = diagnostics
        enriched[task_name] = record_payload
    return enriched


def _write_run_summary(
    model_output_path: str,
    model_name: str,
    run_id: Optional[str],
    tasks: list,
    task_results: Dict[str, Any],
    task_configs_by_name: Dict[str, Any],
) -> Dict[str, Any]:
    def _discover_metrics_summaries(root: str) -> Dict[str, Dict[str, Any]]:
        """Scan the run folder for persisted *_metrics.json files.

        This supports the common case where a user reuses the same run-id across
        multiple invocations: older task results exist on disk, but the in-memory
        `task_results` only includes the tasks from the latest invocation.
        """
        summaries: Dict[str, Dict[str, Any]] = {}
        best_rank: Dict[str, tuple] = {}

        root_path = Path(root)
        if not root_path.exists():
            return summaries

        for metrics_path in root_path.rglob("*_metrics.json"):
            try:
                rel_parts = metrics_path.relative_to(root_path).parts
            except Exception:
                continue

            # Heuristic: if file is under <group>/<subtask>/... treat as group.subtask.
            # Otherwise under <task>/... treat as task.
            if len(rel_parts) >= 3:
                key = f"{rel_parts[0]}.{rel_parts[1]}"
            elif len(rel_parts) >= 2:
                key = rel_parts[0]
            else:
                continue

            try:
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle) or {}
            except Exception:
                continue

            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            if not isinstance(metrics, dict):
                continue

            timestamp = payload.get("timestamp")
            ts_rank = str(timestamp) if isinstance(timestamp, str) else ""
            try:
                mtime = metrics_path.stat().st_mtime
            except Exception:
                mtime = 0.0

            rank = (ts_rank, mtime)
            canonical = _normalize_task_key(key)
            if canonical in best_rank and rank <= best_rank[canonical]:
                continue

            best_rank[canonical] = rank
            summaries[key] = _summarize_task_metrics(metrics)

        return summaries

    summary_path = os.path.join(model_output_path, "run_summary.json")

    existing_tasks: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                existing_payload = json.load(handle) or {}
            if isinstance(existing_payload, dict) and isinstance(existing_payload.get("tasks"), dict):
                existing_tasks = dict(existing_payload.get("tasks") or {})
        except Exception:
            existing_tasks = {}

    # Start from what's already on disk (if any), then fill gaps from a scan, then apply
    # the current invocation's in-memory task results on top.
    merged_tasks: Dict[str, Any] = dict(existing_tasks)
    canonical_to_actual: Dict[str, str] = {
        _normalize_task_key(k): k for k in merged_tasks.keys() if isinstance(k, str)
    }

    # 1) Scan disk for any other task outputs under this run-id/model folder.
    disk_tasks = _discover_metrics_summaries(model_output_path)
    for key, value in disk_tasks.items():
        if not isinstance(key, str):
            continue
        canonical = _normalize_task_key(key)
        if canonical in canonical_to_actual:
            # Preserve the existing key spelling, but refresh its metrics from disk.
            merged_tasks[canonical_to_actual[canonical]] = value
        else:
            merged_tasks[key] = value
            canonical_to_actual[canonical] = key

    # 2) Apply current run's tasks (prefer the current CLI key spelling).
    for task_name in tasks:
        metrics = task_results.get(task_name, {}).get("metrics", {}) or {}
        task_config = task_configs_by_name.get(task_name, {})
        metrics_manifest = task_config.get("metrics_manifest") or None
        summarized = _summarize_task_metrics(metrics, metrics_manifest) if metrics else {}

        canonical = _normalize_task_key(task_name)
        existing_key = canonical_to_actual.get(canonical)
        if existing_key and existing_key != task_name:
            # Rename to the current invocation spelling (e.g. prefer spanish-spam over spanish_spam).
            merged_tasks.pop(existing_key, None)
        merged_tasks[task_name] = summarized
        canonical_to_actual[canonical] = task_name

    summary = {
        "model": model_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "tasks": merged_tasks,
    }
    summary["diagnostics"] = _aggregate_summary_diagnostics(merged_tasks)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote run summary to {summary_path}")
    return summary


def _write_run_outcome(
    *,
    model_output_path: str,
    model_name: str,
    run_id: Optional[str],
    exit_policy: str,
    task_records: Dict[str, Dict[str, Any]],
    started_at: datetime,
    invocation_metadata: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    errors: Optional[list] = None,
    run_summary_payload: Optional[Dict[str, Any]] = None,
    force_status: Optional[str] = None,
    force_exit_code: Optional[int] = None,
) -> Dict[str, Any]:
    """Write concise machine-readable run outcome for agents and automation."""
    finished_at = datetime.now()
    duration_s = round(max((finished_at - started_at).total_seconds(), 0.0), 6)
    task_statuses = [str(record.get("status")) for record in task_records.values()]
    exit_code = resolve_exit_code(task_statuses, policy=exit_policy)
    run_status = resolve_run_status(task_statuses)
    if force_status:
        run_status = force_status
    if force_exit_code is not None:
        exit_code = int(force_exit_code)

    counts = {
        "requested_tasks": len(task_records),
        "completed_tasks": sum(1 for record in task_records.values() if record.get("completed")),
        "passed_tasks": sum(1 for record in task_records.values() if record.get("status") == "passed"),
        "degraded_tasks": sum(1 for record in task_records.values() if record.get("status") == "degraded"),
        "skipped_tasks": sum(1 for record in task_records.values() if record.get("status") == "skipped"),
        "no_samples_tasks": sum(1 for record in task_records.values() if record.get("status") == "no_samples"),
        "pending_tasks": sum(1 for record in task_records.values() if record.get("status") == "pending"),
        "error_tasks": sum(1 for record in task_records.values() if record.get("status") == "error"),
        "failed_tasks": sum(1 for record in task_records.values() if record.get("status") == "failed"),
        "non_passing_tasks": sum(
            1
            for record in task_records.values()
            if record.get("status") in {"failed", "error", "pending", "skipped", "no_samples"}
        ),
    }
    tasks_payload = _attach_per_task_diagnostics(task_records, run_summary_payload)

    outcome = {
        "schema_version": RUN_OUTCOME_SCHEMA_VERSION,
        "benchy_version": _get_benchy_version(),
        "model": model_name,
        "run_id": run_id,
        "timestamp": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "ended_at": finished_at.isoformat(),
        "duration_s": duration_s,
        "status": run_status,
        "exit_policy": exit_policy,
        "exit_code": exit_code,
        "counts": counts,
        "invocation": invocation_metadata or {},
        "git": _get_git_metadata(),
        "artifacts": artifacts or {},
        "errors": errors or [],
        "tasks": tasks_payload,
    }
    diagnostics = {}
    if isinstance(run_summary_payload, dict):
        diagnostics = run_summary_payload.get("diagnostics") or {}
    outcome["diagnostics"] = diagnostics

    outcome_path = Path(model_output_path) / "run_outcome.json"
    with outcome_path.open("w", encoding="utf-8") as f:
        json.dump(outcome, f, indent=2, default=str)
    logger.info(f"Wrote run outcome to {outcome_path}")
    return outcome


def _log_run_outcome_summary(outcome: Dict[str, Any]) -> None:
    """Log a compact end-of-run status summary."""
    logger.info("=" * 60)
    logger.info("RUN OUTCOME SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Run status: {outcome.get('status')}")
    logger.info(f"Exit policy: {outcome.get('exit_policy')}")
    logger.info(f"Suggested exit code: {outcome.get('exit_code')}")
    logger.info(f"Duration (s): {outcome.get('duration_s')}")

    counts = outcome.get("counts", {}) or {}
    logger.info(f"Requested tasks: {counts.get('requested_tasks', 0)}")
    logger.info(f"Completed tasks: {counts.get('completed_tasks', 0)}")
    logger.info(f"Passed tasks: {counts.get('passed_tasks', 0)}")
    logger.info(f"Degraded tasks: {counts.get('degraded_tasks', 0)}")
    logger.info(f"Skipped tasks: {counts.get('skipped_tasks', 0)}")
    logger.info(f"No-samples tasks: {counts.get('no_samples_tasks', 0)}")
    logger.info(f"Failed tasks: {counts.get('failed_tasks', 0)}")
    logger.info(f"Error tasks: {counts.get('error_tasks', 0)}")
    logger.info(f"Pending tasks: {counts.get('pending_tasks', 0)}")
    logger.info(f"Non-passing tasks: {counts.get('non_passing_tasks', 0)}")
    logger.info(f"Errors: {len(outcome.get('errors') or [])}")
    logger.info("")
    logger.info("Per-task statuses:")
    tasks = outcome.get("tasks", {}) or {}
    for task_name in sorted(tasks):
        record = tasks[task_name] or {}
        status = record.get("status", "pending")
        reason = record.get("reason")
        suffix = f" ({reason})" if reason else ""
        logger.info(f"  - {task_name}: {status}{suffix}")
    logger.info("=" * 60)




@flow()
def benchmark_pipeline(
    model_name: str,
    tasks: list,
    output_path: str,
    *,
    model_path: Optional[str] = None,
    limit: Optional[int] = None,
    api_endpoint: str = "completions",
    task_defaults_overrides: Optional[Dict[str, Any]] = None,
    adhoc_task_configs: Optional[Dict[str, Any]] = None,
    log_setup: Optional[Any] = None,
    run_id: Optional[str] = None,
    provider_type: str = "vllm",
    provider_config: Optional[Dict[str, Any]] = None,
    compatibility_mode: str = "skip",
    exit_policy: str = "relaxed",
    invocation_metadata: Optional[Dict[str, Any]] = None,
    log_file_path: Optional[str] = None,
    organization: Optional[str] = None,
    url: Optional[str] = None,
    vllm_config: Optional[VLLMServerConfig] = None,
) -> Dict[str, Any]:
    """
    Complete benchmarking pipeline for vLLM and cloud providers.
    
    The pipeline:
    - For vLLM: Starts server, tests API, runs tasks, stops server
    - For cloud providers (OpenAI/Anthropic): Directly runs tasks using API
    
    Args:
        model_name: Model name used for requests and outputs.
        model_path: Optional local path / HF ID to load in vLLM (defaults to model_name).
        tasks: List of task names to run (e.g., ["spanish", "portuguese"])
        output_path: Base output path for results
        limit: Limit number of examples per task (useful for testing)
        api_endpoint: API endpoint mode ("completions" or "chat")
        task_defaults_overrides: Optional dict to override task default parameters
        log_setup: Logging setup object
        run_id: Optional run ID for organizing outputs
        provider_type: Provider type ('vllm', 'openai', or 'anthropic')
        provider_config: Provider configuration (for cloud providers)
        compatibility_mode: Compatibility handling mode for incompatible tasks
        exit_policy: Exit behavior policy ('relaxed', 'smoke', 'strict')
        invocation_metadata: Redacted CLI invocation context (argv, cwd, args)
        log_file_path: Absolute path to the run log file, if available
        vllm_config: vLLM server configuration (only used if provider_type == 'vllm')
    """
    logger.info(f"Starting benchmark pipeline for model: {model_name}")
    logger.info(f"Provider type: {provider_type}")
    logger.info(f"Tasks to run: {tasks}")
    logger.info(f"Using run_id: {run_id}")
    logger.info(f"Exit policy: {exit_policy}")
    run_started_at = datetime.now()
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load GPU configuration from central config
    central_config = load_config('configs/config.yaml')
    gpu_manager = load_gpu_config(central_config)
    
    # Log GPU configuration
    gpu_summary = gpu_manager.get_config_summary()
    logger.info(f"GPU Configuration: {gpu_summary}")

    if vllm_config is None:
        vllm_config = VLLMServerConfig()
    if provider_type == 'vllm' and vllm_config.cuda_devices is None:
        vllm_config.cuda_devices = gpu_manager.get_vllm_cuda_devices()
    
    # Expand task groups into individual tasks
    expanded_tasks = config_manager.expand_task_groups(tasks, central_config)
    logger.info(f"Task expansion: {tasks} -> {expanded_tasks}")
    
    # Update tasks with expanded list
    tasks = expanded_tasks
    
    # Create run-specific output directory structure: {output_path}/{run_id}/{model_name}
    model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)

    # Check for completed tasks to enable resuming runs
    completion_checker = TaskCompletionChecker(
        output_path=output_path,
        run_id=run_id,
        model_name=model_name
    )
    
    # Check completion status for all requested tasks
    completion_records = completion_checker.get_task_records(tasks)
    completion_checker.log_completion_summary(completion_records)
    
    # Filter out completed tasks
    pending_tasks = [task for task, record in completion_records.items() if not record.get("completed")]
    
    if not pending_tasks:
        logger.info("All tasks are already completed! Nothing to run.")
        run_summary_payload = None
        summary_path = Path(model_output_path) / "run_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    run_summary_payload = json.load(f) or {}
            except Exception:
                run_summary_payload = None
        run_outcome = _write_run_outcome(
            model_output_path=model_output_path,
            model_name=model_name,
            run_id=run_id,
            exit_policy=exit_policy,
            task_records=completion_records,
            started_at=run_started_at,
            invocation_metadata=invocation_metadata,
            artifacts=_build_artifacts(
                model_output_path=model_output_path,
                log_file_path=log_file_path,
            ),
            run_summary_payload=run_summary_payload,
        )
        _log_run_outcome_summary(run_outcome)
        return {
            "model_name": model_name,
            "run_id": run_id,
            "status": "all_tasks_completed",
            "completed_tasks": [task for task, record in completion_records.items() if record.get("completed")],
            "message": "All tasks were already completed in previous run",
            "run_outcome": run_outcome,
            "exit_code": run_outcome.get("exit_code", 0),
        }
    
    logger.info(f"Running {len(pending_tasks)} pending tasks: {pending_tasks}")
    
    # Initialize runtime state
    generation_config = None
    server_info = None
    api_test_result = {"status": "skipped"}
    preflight_error: Optional[Exception] = None
    preflight_errors: list[Dict[str, Any]] = []

    try:
        # Step 0: Fetch generation config from model repository (only for vLLM)
        if provider_type == 'vllm':
            generation_config = fetch_generation_config(
                model_name=model_path or model_name,
                hf_cache=vllm_config.hf_cache,
                hf_token=vllm_config.hf_token,
            )

        # Note: run_config.yaml has been removed as it's redundant with run_outcome.json.
        # All run metadata is now stored in run_outcome.json, the authoritative source.

        # Step 1 & 2: Start and test server (only for vLLM)
        if provider_type == 'vllm':
            logger.info(f"Starting vLLM server with CUDA devices: {vllm_config.cuda_devices}")
            server_info = start_vllm_server(
                model_name=model_name,
                model_path=model_path,
                served_model_name=model_name,
                vllm_config=vllm_config,
            )

            # Store server info globally for signal handler
            set_active_server_info(server_info)

            # Step 2: Test vLLM API
            api_test_result = test_vllm_api(
                server_info=server_info,
                model_name=model_name
            )
        else:
            logger.info(f"Using cloud provider {provider_type}, skipping vLLM server startup")

        # Save generation config to output directory
        if generation_config:
            save_generation_config(generation_config, model_output_path, model_name)
    except Exception as exc:
        preflight_error = exc
        preflight_errors.append(
            {
                "stage": "pipeline_preflight",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        logger.exception("Benchmark pipeline failed before task execution: %s", exc)
    if preflight_error is not None:
        if provider_type == 'vllm' and server_info:
            try:
                stop_vllm_server(server_info=server_info, upload_result={})
            finally:
                clear_active_server_info()

        final_records = completion_checker.get_task_records(tasks)
        run_outcome = _write_run_outcome(
            model_output_path=model_output_path,
            model_name=model_name,
            run_id=run_id,
            exit_policy=exit_policy,
            task_records=final_records,
            started_at=run_started_at,
            invocation_metadata=invocation_metadata,
            artifacts=_build_artifacts(
                model_output_path=model_output_path,
                log_file_path=log_file_path,
            ),
            errors=preflight_errors,
            run_summary_payload=None,
            force_status="failed",
            force_exit_code=1,
        )
        _log_run_outcome_summary(run_outcome)
        raise preflight_error
    
    # Step 3: Run evaluation tasks (only pending ones)
    task_results = {}
    task_configs_by_name: Dict[str, Any] = {}
    
    # Load results from already completed tasks
    completed_tasks = [task_name for task_name, record in completion_records.items() if record.get("completed")]
    for task_name in completed_tasks:
        logger.info(f"Loading results from previously completed task: {task_name}")
        # Create a placeholder result for completed tasks
        task_results[task_name] = {
            "model_name": model_name,
            "task": task_name,
            "status": "previously_completed",
            "output_path": f"{model_output_path}/{task_name.split('.', 1)[0]}",
            "message": f"Task {task_name} was completed in a previous run"
        }
    
    gather_result: Dict[str, Any] = {}
    execution_error: Optional[Exception] = None
    execution_errors: list[Dict[str, Any]] = []
    try:
        for task_name in pending_tasks:
            if not is_handler_based_task(task_name):
                raise ValueError(
                    f"Task '{task_name}' is not handler-based. "
                    "Legacy task.json tasks are no longer supported."
                )

            task_config = build_handler_task_config(
                task_name,
                task_defaults_overrides,
                adhoc_task_configs,
            )

            task_configs_by_name[task_name] = task_config

            display_name = task_config.get("display_name", task_name)
            logger.info(f"Running {display_name} evaluation...")

            if log_setup:
                log_setup.log_task_config(task_name, task_config)

            cloud_provider_config = None
            if provider_config:
                cloud_provider_config = {
                    **provider_config,
                    "provider_type": provider_type,
                }

            run_task_fn = _run_logged_task
            if hasattr(run_task_fn, "with_options"):
                run_task_fn = run_task_fn.with_options(name=f"task:{task_name}")
            task_results[task_name] = run_task_fn(
                task_name=task_name,
                model_name=model_name,
                model_output_path=model_output_path,
                server_info=server_info,
                api_test_result=api_test_result,
                task_config=task_config,
                limit=limit,
                cuda_devices=gpu_manager.get_task_cuda_devices() if provider_type == 'vllm' else None,
                provider_type=provider_type,
                provider_config=cloud_provider_config,
                api_endpoint=api_endpoint,
                generation_config=generation_config,
                compatibility_mode=compatibility_mode,
            )

        # Step 4: Gather results
        gather_result = {
            f"{task_name}_results": result for task_name, result in task_results.items()
        }
    except Exception as exc:
        execution_error = exc
        execution_errors.append(
            {
                "stage": "task_execution",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        logger.exception("Benchmark pipeline failed during task execution: %s", exc)
    finally:
        # Step 5: Stop vLLM server (cleanup) - only for vLLM, even if a task fails.
        if provider_type == 'vllm' and server_info:
            try:
                stop_vllm_server(server_info=server_info, upload_result=gather_result)
            finally:
                clear_active_server_info()

    cleanup_result = gather_result

    if execution_error is not None:
        final_records = completion_checker.get_task_records(tasks)
        run_summary_payload = None
        try:
            run_summary_payload = _write_run_summary(
                model_output_path=model_output_path,
                model_name=model_name,
                run_id=run_id,
                tasks=tasks,
                task_results=task_results,
                task_configs_by_name=task_configs_by_name,
            )
        except Exception as summary_exc:
            logger.warning("Failed to write run_summary after execution error: %s", summary_exc)

        run_outcome = _write_run_outcome(
            model_output_path=model_output_path,
            model_name=model_name,
            run_id=run_id,
            exit_policy=exit_policy,
            task_records=final_records,
            started_at=run_started_at,
            invocation_metadata=invocation_metadata,
            artifacts=_build_artifacts(
                model_output_path=model_output_path,
                log_file_path=log_file_path,
            ),
            errors=execution_errors,
            run_summary_payload=run_summary_payload,
            force_status="failed",
            force_exit_code=1,
        )
        _log_run_outcome_summary(run_outcome)

        cleanup_result["run_outcome"] = run_outcome
        cleanup_result["exit_code"] = 1
        raise execution_error

    final_records = completion_checker.get_task_records(tasks)
    newly_completed = [task for task in pending_tasks if final_records.get(task, {}).get("completed")]
    newly_not_completed = [task for task in pending_tasks if not final_records.get(task, {}).get("completed")]
    total_completed = len(completed_tasks) + len(newly_completed)
    
    logger.info("=" * 60)
    logger.info("BENCHMARK PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tasks requested: {len(tasks)}")
    logger.info(f"Previously completed: {len(completed_tasks)}")
    logger.info(f"Newly completed: {len(newly_completed)}")
    logger.info(f"Still pending/failed: {len(newly_not_completed)}")
    logger.info(f"Total completed: {total_completed}")
    logger.info("=" * 60)

    run_summary_payload = _write_run_summary(
        model_output_path=model_output_path,
        model_name=model_name,
        run_id=run_id,
        tasks=tasks,
        task_results=task_results,
        task_configs_by_name=task_configs_by_name,
    )

    run_outcome = _write_run_outcome(
        model_output_path=model_output_path,
        model_name=model_name,
        run_id=run_id,
        exit_policy=exit_policy,
        task_records=final_records,
        started_at=run_started_at,
        invocation_metadata=invocation_metadata,
        artifacts=_build_artifacts(
            model_output_path=model_output_path,
            log_file_path=log_file_path,
        ),
        run_summary_payload=run_summary_payload,
    )
    _log_run_outcome_summary(run_outcome)

    cleanup_result["run_outcome"] = run_outcome
    cleanup_result["exit_code"] = run_outcome.get("exit_code", 0)

    logger.info("Benchmark pipeline finished")
    return cleanup_result

@flow()
def test_vllm_server(
    model_name: str,
    model_path: Optional[str] = None,
    run_id: Optional[str] = None,
    vllm_config: Optional[VLLMServerConfig] = None,
) -> Dict[str, Any]:
    """
    Test vLLM server functionality without running full evaluation.
    
    Args:
        model_name: Model name used for requests and outputs.
        model_path: Optional local path / HF ID to load in vLLM (defaults to model_name).
        run_id: Optional run ID for organizing outputs. If not provided, auto-generated.
        vllm_config: vLLM server configuration
    """
    logger.info(f"Using run_id for test: {run_id}")
    vllm_config = vllm_config or VLLMServerConfig()
    
    # Load GPU configuration from central config
    central_config = load_config('configs/config.yaml')
    gpu_manager = load_gpu_config(central_config)
    
    # Log GPU configuration
    gpu_summary = gpu_manager.get_config_summary()
    logger.info(f"GPU Configuration for test: {gpu_summary}")
        
    # Step 1: Start vLLM server
    # Use GPU configuration from central config, with command line override
    if vllm_config.cuda_devices is None:
        vllm_config.cuda_devices = gpu_manager.get_vllm_cuda_devices()
    logger.info(f"Starting vLLM test server with CUDA devices: {vllm_config.cuda_devices}")

    # Write test run configuration (simplified version)
    test_config = {
        'run_metadata': {
            'run_id': run_id,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'run_type': 'test'
        },
        'model': {
            'name': model_name,
            'path': model_path,
        },
        'vllm_server': vllm_config.to_dict(),
        'environment': {
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'prefect_api_url': os.environ.get('PREFECT_API_URL', 'http://localhost:4200/api')
        }
    }

    # Create test output directory and write config
    test_output_path = f"/tmp/benchy_test_outputs/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(test_output_path, exist_ok=True)
    test_config_path = f"{test_output_path}/test_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, sort_keys=False, indent=2)
    logger.info(f"Test configuration written to: {test_config_path}")
    
    server_info = start_vllm_server(
        model_name=model_name,
        model_path=model_path,
        served_model_name=model_name,
        vllm_config=vllm_config,
    )
    
    # Store server info globally for signal handler
    set_active_server_info(server_info)
        
    # Step 2: Test vLLM API
    api_test_result = test_vllm_api(
        server_info=server_info,
        model_name=model_name
    )
    
    # Step 3: Stop vLLM server (cleanup) - depends on upload completion
    cleanup_result = stop_vllm_server(server_info=server_info, upload_result=api_test_result)
    
    # Clear global server info
    clear_active_server_info()
    
    logger.info("Test model config completed successfully")
    return cleanup_result
