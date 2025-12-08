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
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run structured data extraction evaluation.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Dictionary containing server info from start_vllm_server (None for cloud providers)
        api_test_result: API test result (unused but kept for compatibility)
        task_config: Task configuration dictionary
        limit: Limit number of examples per task (useful for testing)
        cuda_devices: CUDA devices to use (unused for this task)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metadata
    """
    # Import here to avoid circular dependencies
    from .structured.benchmark_runner import BenchmarkRunner, save_results
    from .structured.tasks import ParaloqTask, ChatExtractTask
    from .structured.utils.dataset_download import (
        download_and_preprocess_dataset,
        download_and_preprocess_chat_extraction,
    )
    from ..task_completion_checker import write_task_done_file
    import os
    
    logger.info(f"Starting structured extraction evaluation for model: {model_name}")
    
    # Determine provider type and base URL
    provider_type = "vllm"  # Default
    base_url = None
    api_key = "EMPTY"  # For vLLM only
    api_key_env = None
    
    if provider_config:
        provider_type = provider_config.get('provider_type', 'vllm')
        if provider_type in ['openai', 'anthropic', 'surus']:
            base_url = provider_config.get('base_url') or provider_config.get('endpoint')
            api_key_env = provider_config.get('api_key_env')
            api_key = None  # Let interface load from env
            logger.info(f"Using cloud/HTTP provider: {provider_type}")
            logger.info(f"Endpoint: {base_url}")
    
    if base_url is None and server_info:
        base_url = server_info['url'] + '/v1'
        logger.info(f"Using vLLM server URL: {base_url}")
    
    file_logger = logging.getLogger('benchy.structured')
    try:
        file_logger.info("=== Starting Structured Extraction Evaluation ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Provider: {provider_type}")
        if server_info:
            file_logger.info(f"Server URL: {server_info['url']}")
        else:
            file_logger.info(f"Base URL: {base_url}")
    except (RuntimeError, OSError):
        pass
    
    # Create task-specific output path
    output_subdir = task_config.get('output', {}).get('subdirectory', 'structured_extraction')
    task_output_path = f"{output_path}/{output_subdir}"
    Path(task_output_path).mkdir(parents=True, exist_ok=True)
    
    # Get list of tasks to run (default to paraloq for backward compatibility)
    tasks_to_run = task_config.get('tasks', ['paraloq'])
    task_configs = task_config.get('task_configs', {})
    
    # Extract defaults from task config
    defaults = task_config.get('defaults', {})
    
    # Setup paths
    dataset_dir = Path(__file__).parent / 'structured' / '.data'
    cache_dir = Path(__file__).parent / 'structured' / 'cache'
    
    # Store results for each task
    all_task_results = {}
    all_metrics = {}
    
    try:
        # Run each task
        for task_name in tasks_to_run:
            logger.info(f"Running task: {task_name}")
            try:
                file_logger.info(f"--- Starting {task_name} task ---")
            except (RuntimeError, OSError):
                pass
            
            # Get task-specific config (with fallbacks)
            task_cfg = task_configs.get(task_name, {})
            
            # Create task instance
            if task_name == 'paraloq':
                dataset_file = dataset_dir / task_cfg.get('dataset_file', 'paraloq_data.jsonl')
                
                # Auto-download dataset if not present
                if not dataset_file.exists():
                    logger.info(f"Dataset not found at {dataset_file}, downloading...")
                    download_and_preprocess_dataset(
                        dataset_name=task_cfg.get('dataset_name', 'paraloq/json_data_extraction'),
                        output_file=dataset_file,
                        cache_dir=str(cache_dir),
                        split="train",
                        max_input_chars=20000,
                    )
                    logger.info(f"Dataset downloaded successfully to {dataset_file}")
                
                task_instance = ParaloqTask({
                    'dataset': {'data_file': str(dataset_file)},
                    'prompts': task_config.get('prompts', {}),
                })
                
            elif task_name == 'chat_extract':
                dataset_file = dataset_dir / task_cfg.get('dataset_file', 'chat_extract_data.jsonl')
                schema_file = dataset_dir / task_cfg.get('schema_file', 'schema_expected_lead_data.json')
                
                # Auto-download dataset if not present
                if not dataset_file.exists():
                    logger.info(f"Dataset not found at {dataset_file}, downloading...")
                    download_and_preprocess_chat_extraction(
                        dataset_name=task_cfg.get('dataset_name', 'mauroibz/chat_structured_extraction'),
                        output_file=dataset_file,
                        schema_file=schema_file,
                        cache_dir=str(cache_dir),
                        split="train",
                        max_input_chars=20000,
                    )
                    logger.info(f"Dataset downloaded successfully to {dataset_file}")
                
                task_instance = ChatExtractTask({
                    'dataset': {
                        'data_file': str(dataset_file),
                        'schema_file': str(schema_file),
                    },
                    'prompts': task_config.get('prompts', {}),
                })
                
            else:
                raise ValueError(f"Unknown task: {task_name}")
            
            # Prepare the config for BenchmarkRunner
            benchmark_config = {
                'model': {
                    'base_url': base_url,
                    'api_key': api_key,
                    'api_key_env': api_key_env,
                    'temperature': defaults.get('temperature', 0.0),
                    'max_tokens': defaults.get('max_tokens', 2048),
                    'timeout': defaults.get('timeout', 120),
                    'max_retries': defaults.get('max_retries', 3),
                },
                'dataset': task_instance.config.get('dataset', {}),
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
            
            # Add provider-specific config
            if provider_config:
                if provider_type == 'surus':
                    # HTTPInterface expects config[provider_type] structure
                    benchmark_config['surus'] = {
                        'endpoint': provider_config.get('endpoint'),
                        'api_key_env': provider_config.get('api_key_env'),
                        'timeout': provider_config.get('timeout', 30),
                        'max_retries': provider_config.get('max_retries', 3),
                    }
                elif provider_type in ['openai', 'anthropic']:
                    # Merge provider-specific settings for LLM providers
                    for key in ['temperature', 'max_tokens', 'timeout', 'max_retries']:
                        if key in provider_config:
                            benchmark_config['model'][key] = provider_config[key]
            
            # Create and run benchmark for this task
            runner = BenchmarkRunner(model_name, benchmark_config, task=task_instance, provider_type=provider_type)
            
            # Run async benchmark in sync context
            task_results = asyncio.run(runner.run(
                limit=limit,
                log_samples=benchmark_config['output']['log_samples'],
                no_resume=False,
            ))
            
            # Save individual task results
            task_output_subdir = Path(task_output_path) / task_name
            save_results(
                results=task_results,
                output_dir=task_output_subdir,
                model_name=model_name,
                log_samples=benchmark_config['output']['log_samples'],
                config=benchmark_config
            )
            
            # Move example_message file to task-specific subdirectory if it exists
            # (BenchmarkRunner saves it in the top-level output dir, we move it to task subdir)
            example_file_top = Path(task_output_path) / f"example_message_{task_name}.txt"
            example_file_task = task_output_subdir / f"example_message_{task_name}.txt"
            if example_file_top.exists():
                if example_file_task.exists():
                    example_file_top.unlink()  # Remove duplicate if exists
                else:
                    example_file_top.rename(example_file_task)
                    logger.debug(f"Moved example message to task subdirectory: {example_file_task}")
            
            # Store results with task prefix
            all_task_results[task_name] = task_results
            all_metrics[task_name] = task_results.get('aggregate_metrics', {})
            
            logger.info(f"Task {task_name} completed successfully")
            try:
                file_logger.info(f"--- {task_name} task COMPLETED ---")
            except (RuntimeError, OSError):
                pass
        
        # Aggregate metrics across all tasks
        aggregated_metrics = _aggregate_task_metrics(all_metrics, tasks_to_run)
        
        # Save combined summary
        _save_aggregated_results(
            aggregated_metrics=aggregated_metrics,
            task_metrics=all_metrics,
            output_dir=Path(task_output_path),
            model_name=model_name,
            tasks=tasks_to_run,
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
            "metrics": aggregated_metrics,
            "task_metrics": all_metrics,
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


def _aggregate_task_metrics(task_metrics: Dict[str, Dict], task_names: list) -> Dict:
    """Aggregate metrics across multiple tasks.
    
    Args:
        task_metrics: Dictionary mapping task names to their metrics
        task_names: List of task names
        
    Returns:
        Aggregated metrics dictionary
    """
    if not task_metrics:
        return {}
    
    # Aggregate common metrics
    total_samples = sum(m.get('total_samples', 0) for m in task_metrics.values())
    valid_samples = sum(m.get('valid_samples', 0) for m in task_metrics.values())
    error_count = sum(m.get('error_count', 0) for m in task_metrics.values())
    
    # Weighted averages for score metrics
    valid_metrics = {k: v for k, v in task_metrics.items() if v.get('valid_samples', 0) > 0}
    
    aggregated = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'error_count': error_count,
        'error_rate': error_count / total_samples if total_samples > 0 else 0.0,
        
        # Per-task breakdown
        'tasks': task_metrics,
    }
    
    if valid_metrics:
        # Weight by number of valid samples
        total_valid = sum(m.get('valid_samples', 0) for m in valid_metrics.values())
        
        for metric_name in ['extraction_quality_score', 'field_f1_partial', 'field_f1_strict',
                           'schema_validity_rate', 'exact_match_rate', 'hallucination_rate']:
            weighted_sum = sum(
                m.get(metric_name, 0) * m.get('valid_samples', 0)
                for m in valid_metrics.values()
            )
            aggregated[metric_name] = weighted_sum / total_valid if total_valid > 0 else 0.0
    else:
        # All zeros if no valid samples
        for metric_name in ['extraction_quality_score', 'field_f1_partial', 'field_f1_strict',
                           'schema_validity_rate', 'exact_match_rate', 'hallucination_rate']:
            aggregated[metric_name] = 0.0
    
    return aggregated


def _save_aggregated_results(
    aggregated_metrics: Dict,
    task_metrics: Dict[str, Dict],
    output_dir: Path,
    model_name: str,
    tasks: list,
):
    """Save aggregated results summary."""
    import json
    from datetime import datetime
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_")
    
    # Save aggregated metrics
    summary_file = output_dir / f"{safe_name}_{timestamp}_aggregated_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "tasks": tasks,
            "aggregated_metrics": aggregated_metrics,
            "per_task_metrics": task_metrics,
        }, f, indent=2)
    
    logger.info(f"Saved aggregated summary to {summary_file}")
    
    # Save text summary
    summary_text_file = output_dir / f"{safe_name}_{timestamp}_aggregated_summary.txt"
    with open(summary_text_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("STRUCTURED EXTRACTION BENCHMARK - AGGREGATED RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Tasks: {', '.join(tasks)}\n")
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("AGGREGATED METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples: {aggregated_metrics.get('total_samples', 0)}\n")
        f.write(f"Valid Samples: {aggregated_metrics.get('valid_samples', 0)}\n")
        f.write(f"Error Rate: {aggregated_metrics.get('error_rate', 0.0):.2%}\n")
        f.write(f"\n")
        f.write(f"Extraction Quality Score: {aggregated_metrics.get('extraction_quality_score', 0.0):.3f}\n")
        f.write(f"Schema Validity Rate: {aggregated_metrics.get('schema_validity_rate', 0.0):.2%}\n")
        f.write(f"Exact Match Rate: {aggregated_metrics.get('exact_match_rate', 0.0):.2%}\n")
        f.write(f"F1 (Partial): {aggregated_metrics.get('field_f1_partial', 0.0):.3f}\n")
        f.write(f"F1 (Strict): {aggregated_metrics.get('field_f1_strict', 0.0):.3f}\n")
        f.write(f"Hallucination Rate: {aggregated_metrics.get('hallucination_rate', 0.0):.2%}\n")
        f.write("\n")
        
        # Per-task breakdown
        f.write("-" * 80 + "\n")
        f.write("PER-TASK BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        for task_name, metrics in task_metrics.items():
            f.write(f"\n{task_name.upper()}:\n")
            f.write(f"  Samples: {metrics.get('total_samples', 0)} (valid: {metrics.get('valid_samples', 0)})\n")
            f.write(f"  EQS: {metrics.get('extraction_quality_score', 0.0):.3f}\n")
            f.write(f"  F1 (Partial): {metrics.get('field_f1_partial', 0.0):.3f}\n")
            f.write(f"  Schema Validity: {metrics.get('schema_validity_rate', 0.0):.2%}\n")
            f.write(f"  Exact Match: {metrics.get('exact_match_rate', 0.0):.2%}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Saved aggregated text summary to {summary_text_file}")

