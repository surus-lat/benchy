"""Template task - Prefect entry point.

This is the main entry point called by the pipeline.
It creates the task instance, interface, and runs the benchmark.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from prefect import task

from ...engine import (
    BenchmarkRunner,
    save_results,
    build_connection_info,
    get_interface_for_provider,
)
from .task import TemplateTask

logger = logging.getLogger(__name__)

# Data directory relative to this module
DATA_DIR = Path(__file__).parent / '.data'


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
    """Run template task evaluation.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info from vLLM (None for cloud providers)
        api_test_result: API test result (unused)
        task_config: Task configuration from configs/tasks/
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    logger.info(f"Starting template task for model: {model_name}")
    
    # Get provider type and build connection info
    provider_type = provider_config.get('provider_type', 'vllm') if provider_config else 'vllm'
    
    connection_info = build_connection_info(
        provider_type=provider_type,
        provider_config=provider_config or {},
        server_info=server_info,
        model_config=task_config.get('defaults', {}),
    )
    
    # Create output directory
    output_subdir = task_config.get('output', {}).get('subdirectory', 'template_task')
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get config defaults
    defaults = task_config.get('defaults', {})
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create task instance
    data_file = DATA_DIR / task_config.get('dataset_file', 'data.jsonl')
    
    task_instance = TemplateTask({
        'dataset': {'data_file': str(data_file)},
        'prompts': task_config.get('prompts', {}),
    })
    
    # Create interface
    interface = get_interface_for_provider(
        provider_type=provider_type,
        connection_info=connection_info,
        model_name=model_name,
    )
    
    # Create runner config
    runner_config = {
        "model_name": model_name,
        "batch_size": defaults.get('batch_size', 20),
        "output_dir": str(task_output_path),
        "log_samples": defaults.get('log_samples', False),
    }
    
    # Run benchmark
    runner = BenchmarkRunner(task_instance, interface, runner_config)
    results = asyncio.run(runner.run(limit=limit, no_resume=False))
    
    # Save results (automatically marks task complete)
    save_results(
        results=results,
        output_dir=task_output_path,
        model_name=model_name,
        task_name=task_instance.get_task_name(),
        log_samples=defaults.get('log_samples', False),
    )
    
    logger.info("Template task completed successfully")
    
    return {
        "model_name": model_name,
        "task": "template_task",
        "output_path": str(task_output_path),
        "metrics": results.get('aggregate_metrics', {}),
    }

