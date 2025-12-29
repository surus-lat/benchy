"""Portuguese benchmark - Prefect task entry point."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import task

from ...engine import (
    BenchmarkRunner,
    build_connection_info,
    get_interface_for_provider,
    mark_task_complete,
    save_results,
)
from ..summary_reporter import write_group_summary

from .datasets.assin2_rte.task import Assin2RteTask
from .datasets.assin2_sts.task import Assin2StsTask
from .datasets.bluex.task import BluexTask
from .datasets.enem_challenge.task import EnemChallengeTask
from .datasets.oab_exams.task import OabExamsTask

logger = logging.getLogger(__name__)

TASK_CLASSES = {
    "assin2_rte": Assin2RteTask,
    "assin2_sts": Assin2StsTask,
    "bluex": BluexTask,
    "enem_challenge": EnemChallengeTask,
    "oab_exams": OabExamsTask,
}


@task
def run_portuguese(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Portuguese evaluation."""
    logger.info(f"Starting Portuguese evaluation for model: {model_name}")

    provider_type = "vllm"
    if provider_config:
        provider_type = provider_config.get("provider_type", "vllm")

    connection_info = build_connection_info(
        provider_type=provider_type,
        provider_config=provider_config or {},
        server_info=server_info,
        model_config=task_config.get("defaults", {}),
    )

    output_subdir = task_config.get("output", {}).get("subdirectory", "portuguese")
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)

    subtasks_to_run = task_config.get("tasks", [])
    subtask_configs = task_config.get("task_configs", {})
    defaults = task_config.get("defaults", {})
    prompts = task_config.get("prompts", {})

    all_results: Dict[str, Any] = {}
    all_metrics: Dict[str, Dict] = {}

    try:
        for subtask_name in subtasks_to_run:
            logger.info(f"Running subtask: {subtask_name}")

            task_class = TASK_CLASSES.get(subtask_name)
            if task_class is None:
                logger.warning(f"Unknown subtask: {subtask_name}, skipping")
                continue

            subtask_config = subtask_configs.get(subtask_name, {})
            task_instance = task_class({
                "dataset": subtask_config,
                "prompts": prompts,
                **subtask_config,
            })

            interface = get_interface_for_provider(
                provider_type=provider_type,
                connection_info=connection_info,
                model_name=model_name,
            )

            runner_config = {
                "model_name": model_name,
                "batch_size": defaults.get("batch_size", 20),
                "output_dir": str(task_output_path / subtask_name),
                "log_samples": defaults.get("log_samples", False),
            }

            runner = BenchmarkRunner(task_instance, interface, runner_config)
            subtask_results = asyncio.run(runner.run(limit=limit, no_resume=False))

            save_results(
                results=subtask_results,
                output_dir=task_output_path / subtask_name,
                model_name=model_name,
                task_name=task_instance.get_task_name(),
                log_samples=defaults.get("log_samples", False),
                mark_complete=False,
            )

            all_results[subtask_name] = subtask_results
            all_metrics[subtask_name] = subtask_results.get("aggregate_metrics", {})

            logger.info(f"Subtask {subtask_name} completed")

        aggregated = _aggregate_subtask_metrics(all_metrics)

        _save_aggregated_summary(
            aggregated_metrics=aggregated,
            subtask_metrics=all_metrics,
            output_dir=task_output_path,
            model_name=model_name,
            subtasks=subtasks_to_run,
        )

        mark_task_complete(task_output_path)

        logger.info("Portuguese evaluation completed successfully")

        return {
            "model_name": model_name,
            "task": "portuguese",
            "output_path": str(task_output_path),
            "metrics": aggregated,
            "subtask_metrics": all_metrics,
        }

    except ConnectionError as exc:
        logger.error(f"Connection failed: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Error running Portuguese evaluation: {type(exc).__name__}: {exc}")
        raise


def _aggregate_subtask_metrics(subtask_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across subtasks using sample-weighted averages."""
    if not subtask_metrics:
        return {}

    total_samples = sum(m.get("total_samples", 0) for m in subtask_metrics.values())
    valid_samples = sum(m.get("valid_samples", 0) for m in subtask_metrics.values())

    aggregated: Dict[str, Any] = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "error_rate": (total_samples - valid_samples) / total_samples if total_samples else 0.0,
    }

    metric_keys = set()
    for metrics in subtask_metrics.values():
        for key, value in metrics.items():
            if key in {"total_samples", "valid_samples", "error_rate", "throughput", "total_duration"}:
                continue
            if isinstance(value, (int, float)):
                metric_keys.add(key)

    for key in metric_keys:
        weighted_sum = 0.0
        total_weight = 0
        for metrics in subtask_metrics.values():
            if key not in metrics:
                continue
            weight = metrics.get("valid_samples", 0)
            weighted_sum += metrics.get(key, 0.0) * weight
            total_weight += weight
        aggregated[key] = weighted_sum / total_weight if total_weight else 0.0

    return aggregated


def _save_aggregated_summary(
    aggregated_metrics: Dict[str, Any],
    subtask_metrics: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model_name: str,
    subtasks: list,
) -> None:
    """Save aggregated results summary."""
    write_group_summary(
        output_dir=output_dir,
        model_name=model_name,
        subtasks=subtasks,
        aggregated_metrics=aggregated_metrics,
        subtask_metrics=subtask_metrics,
        title="PORTUGUESE BENCHMARK SUMMARY",
        aggregated_fields=[
            ("total_samples", "Total Samples", "d"),
            ("valid_samples", "Valid Samples", "d"),
            ("error_rate", "Error Rate", ".2%"),
            ("acc", "Accuracy", ".4f"),
            ("f1_macro", "F1 Macro", ".4f"),
            ("pearson", "Pearson", ".4f"),
            ("mse", "MSE", ".4f"),
        ],
        per_subtask_fields=[
            ("total_samples", "Samples", "d"),
            ("acc", "Accuracy", ".4f"),
            ("f1_macro", "F1 Macro", ".4f"),
            ("pearson", "Pearson", ".4f"),
            ("mse", "MSE", ".4f"),
        ],
    )
