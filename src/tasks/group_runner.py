"""Shared runner for grouped benchmark tasks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..engine import BenchmarkRunner, build_connection_info, get_interface_for_provider, mark_task_complete, save_results
from ..engine.protocols import BaseTask, check_compatibility

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskGroupSpec:
    """Configuration for running a task group.

    Use this in each task's run wrapper to declare how subtasks are built and
    how results are aggregated. Provide either:
    - prepare_task (default runner), or
    - run_subtask (custom runner).

    Optional setup/teardown hooks let you load shared resources once and reuse
    them across subtasks via TaskGroupContext.shared.
    """

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
    setup: Optional[Callable[["TaskGroupContext"], Any]] = None
    teardown: Optional[Callable[["TaskGroupContext", Any], None]] = None


@dataclass
class TaskGroupContext:
    """Shared context for a task group run.

    This is passed to setup/teardown hooks to preload shared resources.
    """
    task_config: Dict[str, Any]
    defaults: Dict[str, Any]
    prompts: Dict[str, Any]
    model_name: str
    provider_type: str
    connection_info: Dict[str, Any]
    output_dir: Path
    limit: Optional[int]
    shared: Any


@dataclass
class SubtaskContext:
    """Context for a single subtask execution."""
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
    shared: Any = None


def ensure_task_interface_compatibility(task: BaseTask, interface: Any) -> None:
    """Raise if a task/interface pair is not compatible."""
    compatible, reason = check_compatibility(task, interface)
    if not compatible:
        raise ValueError(f"Incompatible task/interface: {reason}")


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
    """Run a task group using a shared execution flow.

    Returns a consistent shape for all tasks:
    - metrics: aggregated task metrics
    - subtask_metrics: per-subtask aggregate metrics
    """
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

    if not subtasks_to_run:
        raise ValueError(
            f"No subtasks configured for {spec.name}. Set tasks in the task config or default_subtasks in the spec."
        )

    shared_state: Any = {}
    group_context = TaskGroupContext(
        task_config=task_config,
        defaults=defaults,
        prompts=prompts,
        model_name=model_name,
        provider_type=provider_type,
        connection_info=connection_info,
        output_dir=task_output_path,
        limit=limit,
        shared=shared_state,
    )

    if spec.setup:
        setup_result = spec.setup(group_context)
        if setup_result is not None:
            shared_state = setup_result
            group_context.shared = shared_state
        else:
            shared_state = group_context.shared

    all_metrics: Dict[str, Dict[str, Any]] = {}

    async def _run_default_subtasks() -> None:
        """Run default subtask flow with a single event loop."""
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
                shared=shared_state,
            )

            subtask_results = await _run_default_subtask_async(spec, context)

            if subtask_results:
                all_metrics[subtask_name] = subtask_results.get("aggregate_metrics", {})

            logger.info(f"Subtask {subtask_name} completed")

    try:
        if spec.run_subtask:
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
                    shared=shared_state,
                )

                subtask_results = spec.run_subtask(context)

                if subtask_results:
                    all_metrics[subtask_name] = subtask_results.get("aggregate_metrics", {})

                logger.info(f"Subtask {subtask_name} completed")
        else:
            if spec.prepare_task is None:
                raise ValueError(f"No task factory provided for {spec.name}")
            asyncio.run(_run_default_subtasks())

        aggregated: Dict[str, Any] = {}
        if spec.aggregate_metrics:
            aggregated = spec.aggregate_metrics(all_metrics, subtasks_to_run)
        elif len(subtasks_to_run) == 1:
            aggregated = all_metrics.get(subtasks_to_run[0], {})

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

    except ConnectionError as exc:
        logger.error(f"Connection failed: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Error running {spec.name}: {type(exc).__name__}: {exc}")
        raise
    finally:
        if spec.teardown:
            try:
                spec.teardown(group_context, shared_state)
            except Exception as exc:
                logger.warning(f"Teardown failed for {spec.name}: {type(exc).__name__}: {exc}")


async def _run_default_subtask_async(spec: TaskGroupSpec, context: SubtaskContext) -> Dict[str, Any]:
    """Run one subtask using the default BenchmarkRunner flow."""
    task_instance = spec.prepare_task(context)
    if task_instance is None:
        return {}

    interface = get_interface_for_provider(
        provider_type=context.provider_type,
        connection_info=context.connection_info,
        model_name=context.model_name,
    )
    ensure_task_interface_compatibility(task_instance, interface)

    runner_config = {
        "model_name": context.model_name,
        "batch_size": context.defaults.get("batch_size", 20),
        "output_dir": str(context.subtask_output_dir),
        "log_samples": context.defaults.get("log_samples", False),
    }

    runner = BenchmarkRunner(task_instance, interface, runner_config)
    results = await runner.run(limit=context.limit, no_resume=False)

    save_results(
        results=results,
        output_dir=context.subtask_output_dir,
        model_name=context.model_name,
        task_name=task_instance.get_task_name(),
        log_samples=context.defaults.get("log_samples", False),
        mark_complete=False,
    )

    return results
