"""Image extraction benchmark - Prefect task entry point.

This is the main entry point for the image extraction task.
It uses the generic benchmark engine to run evaluation.
"""

import logging
from typing import Dict, Any, Optional
from prefect import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
from .task import ImageExtractionTask

logger = logging.getLogger(__name__)


@task
def run_image_extraction(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run image extraction evaluation.
    
    This is a Prefect task that wraps the generic benchmark runner.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info from vLLM (None for cloud providers)
        api_test_result: API test result (unused, for interface compatibility)
        task_config: Task configuration from src/tasks/image_extraction/task.json
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused for this task)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    return run_task_group(
        spec=IMAGE_EXTRACTION_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


def _prepare_image_extraction_task(context: SubtaskContext):
    return ImageExtractionTask({
        "source_dir": context.task_config.get("source_dir", ""),
        "prompts": context.task_config.get("prompts", {}),
        "metrics": context.task_config.get("metrics", {}),
        "capability_requirements": context.task_config.get("capability_requirements", {}),
    })


IMAGE_EXTRACTION_SPEC = TaskGroupSpec(
    name="image_extraction",
    display_name="Image extraction",
    output_subdir="image_extraction",
    supports_subtasks=False,
    prepare_task=_prepare_image_extraction_task,
)
