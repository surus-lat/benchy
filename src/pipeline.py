"""Main Prefect pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from .prefect_compat import flow, task, NO_CACHE
from .inference.vllm_server import start_vllm_server, test_vllm_api, stop_vllm_server
from .inference.vllm_config import VLLMServerConfig
from .config_loader import load_config
from .config_manager import ConfigManager
from .generation_config import fetch_generation_config, save_generation_config
from .gpu_config import load_gpu_config
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
) -> Dict[str, float]:
    """Summarize metrics for a single task/subtask."""
    summarized: Dict[str, float] = {}
    if metrics_manifest:
        for key in metrics_manifest:
            value = metrics.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                summarized[key] = float(value)
        return summarized

    for key, value in metrics.items():
        if key in SUMMARY_SKIP_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            summarized[key] = float(value)
    return summarized


def _write_run_summary(
    model_output_path: str,
    model_name: str,
    run_id: Optional[str],
    tasks: list,
    task_results: Dict[str, Any],
    task_configs_by_name: Dict[str, Any],
) -> None:
    def _canonical_task_key(task_key: str) -> str:
        """Canonicalize task keys so `classify.spanish-spam` == `classify.spanish_spam`."""
        parts = (task_key or "").split(".", 1)
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1].replace('-', '_')}"
        return task_key or ""

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
            canonical = _canonical_task_key(key)
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
        _canonical_task_key(k): k for k in merged_tasks.keys() if isinstance(k, str)
    }

    # 1) Scan disk for any other task outputs under this run-id/model folder.
    disk_tasks = _discover_metrics_summaries(model_output_path)
    for key, value in disk_tasks.items():
        if not isinstance(key, str):
            continue
        canonical = _canonical_task_key(key)
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

        canonical = _canonical_task_key(task_name)
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
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote run summary to {summary_path}")


def write_run_config(
    model_name: str,
    model_path: Optional[str],
    run_id: str,
    output_path: str,
    tasks: list,
    limit: Optional[int],
    api_endpoint: str,
    task_defaults_overrides: Optional[Dict[str, Any]],
    vllm_config: VLLMServerConfig,
    organization: Optional[str] = None,
    url: Optional[str] = None
) -> str:
    """
    Write complete run configuration to a YAML file.
    
    Args:
        model_name: The model being evaluated (request / display name).
        model_path: Optional local path / HF ID used to load the model in vLLM.
        run_id: Run ID for this execution
        output_path: Base output path
        tasks: List of tasks to run
        limit: Limit for examples per task
        api_endpoint: API endpoint mode ("completions" or "chat")
        task_defaults_overrides: Task configuration overrides
        vllm_config: vLLM server configuration
        organization: Optional organization name
        url: Optional URL for the model/organization
        
    Returns:
        Path to the written config file
    """
    # Create the complete configuration dictionary
    model_config = {
        'name': model_name,
        'path': model_path,
        'api_endpoint': api_endpoint
    }
    
    # Add organization and url if provided
    if organization is not None:
        model_config['organization'] = organization
    if url is not None:
        model_config['url'] = url
    
    run_config = {
        'run_metadata': {
            'run_id': run_id,
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'output_structure': f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
        },
        'model': model_config,
        'tasks': tasks,
        'evaluation': {
            'limit': limit,
            'task_defaults_overrides': task_defaults_overrides or {}
        },
        'vllm_server': vllm_config.to_dict(),
        'environment': {
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'prefect_api_url': os.environ.get('PREFECT_API_URL', 'http://localhost:4200/api')
        }
    }
    
    # Create model output directory
    model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)
    
    # Write config file
    config_file_path = f"{model_output_path}/run_config.yaml"
    with open(config_file_path, 'w') as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    logger.info(f"Run configuration written to: {config_file_path}")
    return config_file_path


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
    log_setup: Optional[Any] = None,
    run_id: Optional[str] = None,
    provider_type: str = "vllm",
    provider_config: Optional[Dict[str, Any]] = None,
    compatibility_mode: str = "skip",
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
        vllm_config: vLLM server configuration (only used if provider_type == 'vllm')
    """
    logger.info(f"Starting benchmark pipeline for model: {model_name}")
    logger.info(f"Provider type: {provider_type}")
    logger.info(f"Tasks to run: {tasks}")
    logger.info(f"Using run_id: {run_id}")
    
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
    
    # Check for completed tasks to enable resuming failed runs
    completion_checker = TaskCompletionChecker(
        output_path=output_path,
        run_id=run_id,
        model_name=model_name
    )
    
    # Check completion status for all requested tasks
    completion_status = completion_checker.get_completed_tasks(tasks)
    completion_checker.log_completion_summary(completion_status)
    
    # Filter out completed tasks
    pending_tasks = [task for task, completed in completion_status.items() if not completed]
    
    if not pending_tasks:
        logger.info("All tasks are already completed! Nothing to run.")
        # Return early with a success result
        return {
            "model_name": model_name,
            "run_id": run_id,
            "status": "all_tasks_completed",
            "completed_tasks": tasks,
            "message": "All tasks were already completed in previous run"
        }
    
    logger.info(f"Running {len(pending_tasks)} pending tasks: {pending_tasks}")
    
    # Step 0: Fetch generation config from model repository (only for vLLM)
    generation_config = None
    if provider_type == 'vllm':
        generation_config = fetch_generation_config(
            model_name=model_path or model_name,
            hf_cache=vllm_config.hf_cache,
            hf_token=vllm_config.hf_token,
        )
    
    # Write complete run configuration
    write_run_config(
        model_name=model_name,
        model_path=model_path,
        run_id=run_id,
        output_path=output_path,
        tasks=tasks,
        limit=limit,
        api_endpoint=api_endpoint,
        task_defaults_overrides=task_defaults_overrides,
        vllm_config=vllm_config,
        organization=organization,
        url=url
    )
    
    # Initialize server_info and api_test_result
    server_info = None
    api_test_result = {"status": "skipped"}
    
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
    
    # Create run-specific output directory structure: {output_path}/{run_id}/{model_name}
    model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save generation config to output directory
    if generation_config:
        save_generation_config(generation_config, model_output_path, model_name)
    
    # Step 3: Run evaluation tasks (only pending ones)
    task_results = {}
    task_configs_by_name: Dict[str, Any] = {}
    
    # Load results from already completed tasks
    completed_tasks = [task for task, completed in completion_status.items() if completed]
    for task in completed_tasks:
        logger.info(f"Loading results from previously completed task: {task}")
        # Create a placeholder result for completed tasks
        task_results[task] = {
            "model_name": model_name,
            "task": task,
            "status": "previously_completed",
            "output_path": f"{model_output_path}/{task}",
            "message": f"Task {task} was completed in a previous run"
        }
    
    gather_result: Dict[str, Any] = {}
    try:
        for task_name in pending_tasks:
            if not is_handler_based_task(task_name):
                raise ValueError(
                    f"Task '{task_name}' is not handler-based. "
                    "Legacy task.json tasks are no longer supported."
                )

            task_config = build_handler_task_config(task_name, task_defaults_overrides)

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
    finally:
        # Step 5: Stop vLLM server (cleanup) - only for vLLM, even if a task fails.
        if provider_type == 'vllm' and server_info:
            try:
                stop_vllm_server(server_info=server_info, upload_result=gather_result)
            finally:
                clear_active_server_info()

    cleanup_result = gather_result
    
    # Log final completion summary
    newly_completed = [task for task in pending_tasks if task in task_results]
    total_completed = len(completed_tasks) + len(newly_completed)
    
    logger.info("=" * 60)
    logger.info("BENCHMARK PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tasks requested: {len(tasks)}")
    logger.info(f"Previously completed: {len(completed_tasks)}")
    logger.info(f"Newly completed: {len(newly_completed)}")
    logger.info(f"Total completed: {total_completed}")
    logger.info("=" * 60)

    _write_run_summary(
        model_output_path=model_output_path,
        model_name=model_name,
        run_id=run_id,
        tasks=tasks,
        task_results=task_results,
        task_configs_by_name=task_configs_by_name,
    )
    
    logger.info("Benchmark pipeline completed successfully")
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
