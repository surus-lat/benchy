"""Classification task - Prefect entry point."""

import logging
from typing import Any, Dict, Optional

from ...prefect_compat import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
from .task import ClassifyTask

logger = logging.getLogger(__name__)


@task
def run_classify(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run classification evaluation."""
    return run_task_group(
        spec=CLASSIFY_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


def _prepare_classify_task(context: SubtaskContext) -> ClassifyTask:
    prompts = context.subtask_config.get("prompts", context.prompts)
    dataset_config = context.subtask_config.get("dataset", {})
    return ClassifyTask(
        {
            "dataset": dataset_config,
            "prompts": prompts,
            "subtask_name": context.subtask_name,
        }
    )


def _aggregate_classify_metrics(subtask_metrics: Dict[str, Dict[str, Any]], subtask_names: list) -> Dict[str, Any]:
    if not subtask_metrics:
        return {}

    total_samples = sum(m.get("total_samples", 0) for m in subtask_metrics.values())
    valid_samples = sum(m.get("valid_samples", 0) for m in subtask_metrics.values())

    weighted_accuracy = 0.0
    if valid_samples > 0:
        weighted_accuracy = sum(
            m.get("accuracy", 0.0) * m.get("valid_samples", 0) for m in subtask_metrics.values()
        ) / valid_samples

    error_rate = (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0

    return {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "accuracy": weighted_accuracy,
        "error_rate": error_rate,
    }


CLASSIFY_SPEC = TaskGroupSpec(
    name="classify",
    display_name="Classification",
    output_subdir="classify",
    supports_subtasks=True,
    prepare_task=_prepare_classify_task,
    aggregate_metrics=_aggregate_classify_metrics,
)
