"""Structured data extraction benchmark - Prefect task entry point.

This is the main entry point for the structured extraction task.
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
from ..summary_reporter import write_group_summary
from .tasks import ParaloqTask, ChatExtractTask
from .utils.dataset_download import (
    download_and_preprocess_dataset,
    download_and_preprocess_chat_extraction,
)

logger = logging.getLogger(__name__)

# Data and cache directories relative to this module
DATA_DIR = Path(__file__).parent / '.data'
CACHE_DIR = Path(__file__).parent / 'cache'


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
    """Run structured data extraction evaluation.
    
    This is a Prefect task that wraps the generic benchmark runner.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info from vLLM (None for cloud providers)
        api_test_result: API test result (unused, for interface compatibility)
        task_config: Task configuration from src/tasks/structured/task.json
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused for this task)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    logger.info(f"Starting structured extraction evaluation for model: {model_name}")
    
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
    output_subdir = task_config.get('output', {}).get('subdirectory', 'structured_extraction')
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of subtasks to run
    subtasks_to_run = task_config.get('tasks', ['paraloq'])
    subtask_configs = task_config.get('task_configs', {})
    defaults = task_config.get('defaults', {})
    
    # Store results for each subtask
    all_results = {}
    all_metrics = {}
    
    try:
        for subtask_name in subtasks_to_run:
            logger.info(f"Running subtask: {subtask_name}")
            
            # Create task instance
            task_instance = _create_task_instance(
                subtask_name=subtask_name,
                subtask_config=subtask_configs.get(subtask_name, {}),
                task_config=task_config,
            )
            
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
                "output_dir": str(task_output_path / subtask_name),
                "log_samples": defaults.get('log_samples', False),
            }
            
            # Run benchmark using generic engine
            runner = BenchmarkRunner(task_instance, interface, runner_config)
            subtask_results = asyncio.run(runner.run(limit=limit, no_resume=False))
            
            # Save results (don't mark subtask complete - parent handles that)
            save_results(
                results=subtask_results,
                output_dir=task_output_path / subtask_name,
                model_name=model_name,
                task_name=subtask_name,
                log_samples=defaults.get('log_samples', False),
                mark_complete=False,  # Parent task handles completion
            )
            
            all_results[subtask_name] = subtask_results
            all_metrics[subtask_name] = subtask_results.get('aggregate_metrics', {})
            
            logger.info(f"Subtask {subtask_name} completed")
        
        # Aggregate metrics across all subtasks
        aggregated = _aggregate_subtask_metrics(all_metrics, subtasks_to_run)
        
        # Save aggregated summary
        _save_aggregated_summary(
            aggregated_metrics=aggregated,
            subtask_metrics=all_metrics,
            output_dir=task_output_path,
            model_name=model_name,
            subtasks=subtasks_to_run,
        )
        
        # Mark parent task complete (subtasks marked by save_results)
        mark_task_complete(task_output_path)
        
        logger.info("Structured extraction evaluation completed successfully")
        
        return {
            "model_name": model_name,
            "task": "structured_extraction",
            "output_path": str(task_output_path),
            "metrics": aggregated,
            "subtask_metrics": all_metrics,
        }
        
    except ConnectionError as e:
        # Clean error for connection failures (no traceback needed)
        logger.error(f"Connection failed: {e}")
        logger.error(f"Check that the endpoint is accessible and responding")
        raise
    except Exception as e:
        logger.error(f"Error running structured extraction: {type(e).__name__}: {e}")
        raise


def _create_task_instance(
    subtask_name: str,
    subtask_config: Dict,
    task_config: Dict,
):
    """Create a task instance for a subtask."""
    if subtask_name == 'paraloq':
        dataset_file = DATA_DIR / subtask_config.get('dataset_file', 'paraloq_data.jsonl')
        
        # Auto-download if needed
        if not dataset_file.exists():
            logger.info(f"Downloading paraloq dataset to {dataset_file}")
            download_and_preprocess_dataset(
                dataset_name=subtask_config.get('dataset_name', 'paraloq/json_data_extraction'),
                output_file=dataset_file,
                cache_dir=str(CACHE_DIR),
                split="train",
                max_input_chars=20000,
            )
        
        return ParaloqTask({
            'dataset': {'data_file': str(dataset_file)},
            'prompts': task_config.get('prompts', {}),
            'metrics': task_config.get('metrics', {}),
        })
        
    elif subtask_name == 'chat_extract':
        dataset_file = DATA_DIR / subtask_config.get('dataset_file', 'chat_extract_data.jsonl')
        schema_file = DATA_DIR / subtask_config.get('schema_file', 'schema_expected_lead_data.json')
        
        # Auto-download if needed
        if not dataset_file.exists():
            logger.info(f"Downloading chat_extract dataset to {dataset_file}")
            download_and_preprocess_chat_extraction(
                dataset_name=subtask_config.get('dataset_name', 'mauroibz/chat_structured_extraction'),
                output_file=dataset_file,
                schema_file=schema_file,
                cache_dir=str(CACHE_DIR),
                split="train",
                max_input_chars=20000,
            )
        
        return ChatExtractTask({
            'dataset': {
                'data_file': str(dataset_file),
                'schema_file': str(schema_file),
            },
            'prompts': task_config.get('prompts', {}),
            'metrics': task_config.get('metrics', {}),
        })
        
    else:
        raise ValueError(f"Unknown subtask: {subtask_name}")


def _aggregate_subtask_metrics(subtask_metrics: Dict[str, Dict], subtask_names: list) -> Dict:
    """Aggregate metrics across subtasks."""
    if not subtask_metrics:
        return {}
    
    total_samples = sum(m.get('total_samples', 0) for m in subtask_metrics.values())
    valid_samples = sum(m.get('valid_samples', 0) for m in subtask_metrics.values())
    error_count = sum(m.get('error_count', 0) for m in subtask_metrics.values())
    
    aggregated = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'error_count': error_count,
        'error_rate': error_count / total_samples if total_samples > 0 else 0.0,
        'subtasks': subtask_metrics,
    }
    
    # Weighted average for key metrics
    valid_subtasks = {k: v for k, v in subtask_metrics.items() if v.get('valid_samples', 0) > 0}
    
    if valid_subtasks:
        total_valid = sum(m.get('valid_samples', 0) for m in valid_subtasks.values())
        
        for metric in ['extraction_quality_score', 'field_f1_partial', 'field_f1_strict',
                       'schema_validity_rate', 'exact_match_rate', 'hallucination_rate']:
            weighted_sum = sum(
                m.get(metric, 0) * m.get('valid_samples', 0)
                for m in valid_subtasks.values()
            )
            aggregated[metric] = weighted_sum / total_valid if total_valid > 0 else 0.0
    else:
        for metric in ['extraction_quality_score', 'field_f1_partial', 'field_f1_strict',
                       'schema_validity_rate', 'exact_match_rate', 'hallucination_rate']:
            aggregated[metric] = 0.0
    
    return aggregated


def _save_aggregated_summary(
    aggregated_metrics: Dict,
    subtask_metrics: Dict[str, Dict],
    output_dir: Path,
    model_name: str,
    subtasks: list,
):
    """Save aggregated results summary."""
    summary_metrics = dict(aggregated_metrics)
    if "overall_extraction_quality_score" not in summary_metrics:
        summary_metrics["overall_extraction_quality_score"] = summary_metrics.get(
            "extraction_quality_score",
            0,
        )

    write_group_summary(
        output_dir=output_dir,
        model_name=model_name,
        subtasks=subtasks,
        aggregated_metrics=summary_metrics,
        subtask_metrics=subtask_metrics,
        title="STRUCTURED EXTRACTION BENCHMARK SUMMARY",
        aggregated_fields=[
            ("total_samples", "Total Samples", "d"),
            ("valid_samples", "Valid Samples", "d"),
            ("error_rate", "Error Rate", ".2%"),
            ("extraction_quality_score", "EQS", ".3f"),
            ("overall_extraction_quality_score", "Overall EQS (accounts for invalid responses)", ".3f"),
            ("field_f1_partial", "F1 (Partial)", ".3f"),
            ("schema_validity_rate", "Schema Validity", ".2%"),
            ("exact_match_rate", "Exact Match", ".2%"),
            ("hallucination_rate", "Hallucination Rate", ".2%"),
        ],
        per_subtask_fields=[
            ("total_samples", "Samples", "d"),
            ("extraction_quality_score", "EQS", ".3f"),
            ("field_f1_partial", "F1", ".3f"),
        ],
    )
    
    # Log a concise, auditable view of the aggregated metrics in the main run log
    logger.info("Structured extraction - aggregated metrics across subtasks:")
    logger.info(
        "  total_samples=%d, valid_samples=%d, error_rate=%.4f",
        aggregated_metrics.get("total_samples", 0),
        aggregated_metrics.get("valid_samples", 0),
        aggregated_metrics.get("error_count", 0)
        / aggregated_metrics.get("total_samples", 1)
        if aggregated_metrics.get("total_samples", 0) > 0
        else 0.0,
    )
    eqs = aggregated_metrics.get("extraction_quality_score", 0.0)
    overall_eqs = aggregated_metrics.get("overall_extraction_quality_score", eqs)
    logger.info(
        "  EQS=%.4f, Overall_EQS=%.4f, F1_partial=%.4f, schema_validity=%.4f, exact_match=%.4f, hallucination_rate=%.4f",
        eqs,
        overall_eqs,
        aggregated_metrics.get("field_f1_partial", 0.0),
        aggregated_metrics.get("schema_validity_rate", 0.0),
        aggregated_metrics.get("exact_match_rate", 0.0),
        aggregated_metrics.get("hallucination_rate", 0.0),
    )
    for name, metrics in subtask_metrics.items():
        logger.info(
            "Subtask %s -> samples=%d, EQS=%.4f, F1_partial=%.4f",
            name,
            metrics.get("total_samples", 0),
            metrics.get("extraction_quality_score", 0.0),
            metrics.get("field_f1_partial", 0.0),
        )
