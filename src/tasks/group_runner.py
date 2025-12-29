"""Shared runner for grouped benchmark tasks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..engine import BenchmarkRunner, build_connection_info, get_interface_for_provider, mark_task_complete, save_results
from ..engine.protocols import BaseTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskGroupSpec:
    """Specification for a grouped task runner."""

    name: str
    display_name: str
    output_subdir: Optional[str] = None
    supports_subtasks: bool = True
    default_subtasks: Optional[List[str]] = None
    prepare_task: Optional[Callable[["SubtaskContext"], Optional[BaseTask]]] = None
    run_subtask: Optional[Callable[["SubtaskContext"], Dict[str, Any]]] = None
    aggregate_metrics: Optional[Callable[[Dict[str, Dict[str, Any]], List[str]], Dict[str, Any]]] = None
    write_summary: Optional[
        Callable[[Dict[str, Any], Dict[str, Dict[str, Any]], Path, str, List[str]], None]
    ] = None


@dataclass
class SubtaskContext:
    subtask_name: str
    subtask_config: Dict[str, Any]
    task_config: Dict[str, Any]
    defaults: Dict[str, Any]
    prompts: Dict[str, Any]
    model_name: str
    provider_type: str
    connection_info: Dict[str, Any]
    output_dir: Path
    subtask_output_dir: Path
    limit: Optional[int]


def run_task_group(
    *,
    spec: TaskGroupSpec,
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a grouped task using a shared execution flow."""
    logger.info(f"Starting {spec.display_name} evaluation for model: {model_name}")

    provider_type = "vllm"
    if provider_config:
        provider_type = provider_config.get("provider_type", "vllm")

    connection_info = build_connection_info(
        provider_type=provider_type,
        provider_config=provider_config or {},
        server_info=server_info,
        model_config=task_config.get("defaults", {}),
    )

    logger.info(f"Provider: {provider_type}")
    logger.info(f"Base URL: {connection_info.get('base_url')}")

    output_subdir = task_config.get("output", {}).get("subdirectory", spec.output_subdir or spec.name)
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)

    defaults = task_config.get("defaults", {})
    prompts = task_config.get("prompts", {})
    subtask_configs = task_config.get("task_configs", {})

    if spec.supports_subtasks:
        subtasks_to_run = task_config.get("tasks", None)
        if subtasks_to_run is None:
            subtasks_to_run = spec.default_subtasks or []
    else:
        subtasks_to_run = [spec.name]

    all_metrics: Dict[str, Dict[str, Any]] = {}

    try:
        for subtask_name in subtasks_to_run:
            if spec.supports_subtasks:
                logger.info(f"Running subtask: {subtask_name}")
                subtask_config = subtask_configs.get(subtask_name, {})
                subtask_output_dir = task_output_path / subtask_name
            else:
                subtask_config = task_config
                subtask_output_dir = task_output_path

            context = SubtaskContext(
                subtask_name=subtask_name,
                subtask_config=subtask_config,
                task_config=task_config,
                defaults=defaults,
                prompts=prompts,
                model_name=model_name,
                provider_type=provider_type,
                connection_info=connection_info,
                output_dir=task_output_path,
                subtask_output_dir=subtask_output_dir,
                limit=limit,
            )

            if spec.run_subtask:
                subtask_results = spec.run_subtask(context)
            else:
                if spec.prepare_task is None:
                    raise ValueError(f"No task factory provided for {spec.name}")
                subtask_results = _run_default_subtask(spec, context)

            if subtask_results:
                all_metrics[subtask_name] = subtask_results.get("aggregate_metrics", {})

            logger.info(f"Subtask {subtask_name} completed")

        if spec.supports_subtasks:
            aggregated = {}
            if spec.aggregate_metrics:
                aggregated = spec.aggregate_metrics(all_metrics, subtasks_to_run)

            if spec.write_summary:
                spec.write_summary(
                    aggregated,
                    all_metrics,
                    task_output_path,
                    model_name,
                    subtasks_to_run,
                )

            mark_task_complete(task_output_path)

            logger.info(f"{spec.display_name} evaluation completed successfully")

            return {
                "model_name": model_name,
                "task": spec.name,
                "output_path": str(task_output_path),
                "metrics": aggregated,
                "subtask_metrics": all_metrics,
            }

        metrics = all_metrics.get(spec.name, {})
        logger.info(f"{spec.display_name} evaluation completed successfully")
        return {
            "model_name": model_name,
            "task": spec.name,
            "output_path": str(task_output_path),
            "metrics": metrics,
        }

    except ConnectionError as exc:
        logger.error(f"Connection failed: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Error running {spec.name}: {type(exc).__name__}: {exc}")
        raise


def _run_default_subtask(spec: TaskGroupSpec, context: SubtaskContext) -> Dict[str, Any]:
    task_instance = spec.prepare_task(context)
    if task_instance is None:
        return {}

    interface = get_interface_for_provider(
        provider_type=context.provider_type,
        connection_info=context.connection_info,
        model_name=context.model_name,
    )

    runner_config = {
        "model_name": context.model_name,
        "batch_size": context.defaults.get("batch_size", 20),
        "output_dir": str(context.subtask_output_dir),
        "log_samples": context.defaults.get("log_samples", False),
    }

    runner = BenchmarkRunner(task_instance, interface, runner_config)
    results = asyncio.run(runner.run(limit=context.limit, no_resume=False))

    save_results(
        results=results,
        output_dir=context.subtask_output_dir,
        model_name=context.model_name,
        task_name=task_instance.get_task_name(),
        log_samples=context.defaults.get("log_samples", False),
        mark_complete=not spec.supports_subtasks,
    )

    return results
