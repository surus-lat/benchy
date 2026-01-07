"""Template task - grouped entry point for multiple formats.

The task class (TemplateTask) handles:
- Auto-downloading data from HuggingFace if needed
- Preprocessing samples to eval format
- Metrics calculation

This file just wires everything together.
"""

import logging
from typing import Dict, Any, Optional
from ...prefect_compat import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
from .task import (
    TemplateFreeformTask,
    TemplateMultipleChoiceTask,
    TemplateStructuredTask,
)


@task
def run_template_task(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run grouped template task evaluation.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info from vLLM (None for cloud providers)
        api_test_result: API test result (unused, kept for interface compatibility)
        task_config: Task configuration from src/tasks/<task>/task.json
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    return run_task_group(
        spec=TEMPLATE_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=task_config,
        limit=limit,
        cuda_devices=cuda_devices,
        provider_config=provider_config,
    )


def _prepare_template_task(context: SubtaskContext):
    """Instantiate the correct task class based on subtask name."""
    task_map = {
        "multiple_choice_example": TemplateMultipleChoiceTask,
        "structured_example": TemplateStructuredTask,
        "freeform_example": TemplateFreeformTask,
    }
    task_cls = task_map.get(context.subtask_name)
    if task_cls is None:
        return None

    # Merge prompts/dataset config so each subtask can override as needed.
    return task_cls(
        {
            "dataset": context.subtask_config.get("dataset", {}),
            "prompts": context.subtask_config.get("prompts", context.prompts),
        }
    )


TEMPLATE_SPEC = TaskGroupSpec(
    name="template_task",
    display_name="Template task",
    output_subdir="template_task",
    prepare_task=_prepare_template_task,
)
