"""Structured data extraction benchmark task for ML model evaluation."""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from prefect import task

logger = logging.getLogger(__name__)


@task
def run_structured_extraction(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run structured data extraction evaluation.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Dictionary containing server info from start_vllm_server
        api_test_result: API test result (unused but kept for compatibility)
        task_config: Task configuration dictionary
        limit: Limit number of examples per task (useful for testing)
        cuda_devices: CUDA devices to use (unused for this task)
        
    Returns:
        Dictionary with execution results and metadata
    """
    # Import here to avoid circular dependencies
    from .structured.benchmark_runner import BenchmarkRunner, save_results
    from .structured.llm import VLLMInterface
    from .structured.tasks import ParaloqTask
    from .structured.metrics import MetricsCalculator
    from ..task_completion_checker import write_task_done_file
    
    logger.info(f"Starting structured extraction evaluation for model: {model_name}")
    
    file_logger = logging.getLogger('benchy.structured')
    try:
        file_logger.info("=== Starting Structured Extraction Evaluation ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Server URL: {server_info['url']}")
    except (RuntimeError, OSError):
        pass
    
    # Create task-specific output path
    output_subdir = task_config.get('output', {}).get('subdirectory', 'structured_extraction')
    task_output_path = f"{output_path}/{output_subdir}"
    Path(task_output_path).mkdir(parents=True, exist_ok=True)
    
    # Build config for the benchmark runner
    # Extract defaults from task config
    defaults = task_config.get('defaults', {})
    
    # Setup dataset path
    dataset_dir = Path(__file__).parent / 'structured' / '.data'
    dataset_file = dataset_dir / task_config.get('dataset_file', 'paraloq_data.jsonl')
    
    # Auto-download dataset if not present
    if not dataset_file.exists():
        logger.info(f"Dataset not found at {dataset_file}, downloading...")
        try:
            from .structured.utils.dataset_download import download_and_preprocess_dataset
            
            cache_dir = Path(__file__).parent / 'structured' / 'cache'
            download_and_preprocess_dataset(
                dataset_name=task_config.get('dataset_name', 'paraloq/json_data_extraction'),
                output_file=dataset_file,
                cache_dir=str(cache_dir),
                split="train",
                max_input_chars=20000,
            )
            logger.info(f"Dataset downloaded successfully to {dataset_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    
    # Prepare the config in the format expected by BenchmarkRunner
    benchmark_config = {
        'model': {
            'base_url': server_info['url'] + '/v1',
            'api_key': 'EMPTY',
            'temperature': defaults.get('temperature', 0.0),
            'max_tokens': defaults.get('max_tokens', 2048),
            'timeout': defaults.get('timeout', 120),
            'max_retries': defaults.get('max_retries', 3),
        },
        'dataset': {
            'data_file': str(dataset_file),
        },
        'prompts': task_config.get('prompts', {}),
        'output': {
            'results_dir': task_output_path,
            'log_samples': defaults.get('log_samples', False),
        },
        'performance': {
            'batch_size': defaults.get('batch_size', 20),
        },
        'metrics': task_config.get('metrics', {}),
    }
    
    # Log configuration
    try:
        file_logger.info(f"Batch size: {benchmark_config['performance']['batch_size']}")
        file_logger.info(f"Temperature: {benchmark_config['model']['temperature']}")
        if limit:
            file_logger.info(f"Limit: {limit} (testing mode)")
    except (RuntimeError, OSError):
        pass
    
    try:
        # Create and run benchmark
        runner = BenchmarkRunner(model_name, benchmark_config)
        
        # Run async benchmark in sync context
        results = asyncio.run(runner.run(
            limit=limit,
            log_samples=benchmark_config['output']['log_samples'],
            no_resume=False,
        ))
        
        # Save results
        save_results(
            results=results,
            output_dir=Path(task_output_path),
            model_name=model_name,
            log_samples=benchmark_config['output']['log_samples'],
            config=benchmark_config
        )
        
        logger.info("Structured extraction evaluation completed successfully")
        try:
            file_logger.info("=== Structured Extraction Evaluation COMPLETED SUCCESSFULLY ===")
            file_logger.info(f"Output saved to: {task_output_path}")
        except (RuntimeError, OSError):
            pass
        
        # Write done file to mark task completion
        write_task_done_file(task_output_path)
        
        return {
            "model_name": model_name,
            "task": "structured_extraction",
            "output_path": task_output_path,
            "metrics": results.get('aggregate_metrics', {}),
        }
        
    except Exception as e:
        error_msg = f"Error running structured extraction: {str(e)}"
        logger.error(error_msg)
        try:
            file_logger.error(error_msg)
            file_logger.error("=== Structured Extraction Evaluation FAILED ===")
        except (RuntimeError, OSError):
            pass
        raise

