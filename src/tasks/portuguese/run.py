"""Portuguese benchmark - Prefect task entry point."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
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
    return run_task_group(
        spec=PORTUGUESE_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


def _prepare_portuguese_task(context: SubtaskContext):
    task_class = TASK_CLASSES.get(context.subtask_name)
    if task_class is None:
        logger.warning(f"Unknown subtask: {context.subtask_name}, skipping")
        return None

    return task_class({
        "dataset": context.subtask_config,
        "prompts": context.prompts,
        "capability_requirements": context.task_config.get("capability_requirements", {}),
        **context.subtask_config,
    })


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


PORTUGUESE_SPEC = TaskGroupSpec(
    name="portuguese",
    display_name="Portuguese",
    output_subdir="portuguese",
    prepare_task=_prepare_portuguese_task,
    aggregate_metrics=lambda metrics, _: _aggregate_subtask_metrics(metrics),
    write_summary=_save_aggregated_summary,
)
