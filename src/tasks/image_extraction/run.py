"""Image extraction benchmark - Prefect task entry point.

This is the main entry point for the image extraction task.
It uses the generic benchmark engine to run evaluation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from prefect import task

from ...engine import (
    BenchmarkRunner,
    save_results,
    build_connection_info,
    get_interface_for_provider,
    mark_task_complete,
)
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
    logger.info(f"Starting image extraction evaluation for model: {model_name}")
    
    # Determine provider type
    provider_type = "vllm"
    if provider_config:
        provider_type = provider_config.get('provider_type', 'vllm')
    
    # Build connection info from provider config
    connection_info = build_connection_info(
        provider_type=provider_type,
        provider_config=provider_config or {},
        server_info=server_info,
        model_config=task_config.get('defaults', {}),
    )
    
    logger.info(f"Provider: {provider_type}")
    logger.info(f"Base URL: {connection_info.get('base_url')}")
    
    # Create output directory
    output_subdir = task_config.get('output', {}).get('subdirectory', 'image_extraction')
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get defaults
    defaults = task_config.get('defaults', {})
    
    # Create task instance
    task_instance = ImageExtractionTask({
        'source_dir': task_config.get('source_dir', ''),
        'prompts': task_config.get('prompts', {}),
        'metrics': task_config.get('metrics', {}),
    })
    
    # Create interface
    interface = get_interface_for_provider(
        provider_type=provider_type,
        connection_info=connection_info,
        model_name=model_name,
    )
    
    # Verify multimodal support
    if hasattr(interface, 'supports_multimodal') and not interface.supports_multimodal:
        raise ValueError(
            f"Provider {provider_type} does not support multimodal (vision) inputs. "
            "Image extraction requires a vision-capable model."
        )
    
    # Create runner config
    runner_config = {
        "model_name": model_name,
        "batch_size": defaults.get('batch_size', 5),  # Lower default for vision
        "output_dir": str(task_output_path),
        "log_samples": defaults.get('log_samples', False),
    }
    
    # Run benchmark using generic engine
    runner = BenchmarkRunner(task_instance, interface, runner_config)
    results = asyncio.run(runner.run(limit=limit, no_resume=False))
    
    # Save results
    save_results(
        results=results,
        output_dir=task_output_path,
        model_name=model_name,
        task_name=task_instance.get_task_name(),
        log_samples=defaults.get('log_samples', False),
    )
    
    logger.info("Image extraction evaluation completed successfully")
    
    return {
        "model_name": model_name,
        "task": "image_extraction",
        "output_path": str(task_output_path),
        "metrics": results.get('aggregate_metrics', {}),
    }
