"""Spanish benchmark - Prefect task entry point.

This is the main entry point for the Spanish task.
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

logger = logging.getLogger(__name__)

# Import task classes
from .datasets.copa_es.task import CopaEsTask
from .datasets.escola.task import EscolaTask
from .datasets.mgsm.task import MgsmTask
from .datasets.openbookqa_es.task import OpenBookQaEsTask
from .datasets.paws_es.task import PawsEsTask
from .datasets.teleia.pce import TeleiaPceTask
from .datasets.teleia.cervantes_ave import TeleiaCervantesAveTask
from .datasets.teleia.siele import TeleiaSieleTask
from .datasets.wnli_es.task import WnliEsTask
from .datasets.xnli_es.task import XnliEsTask

# Task class mappings
TASK_CLASSES = {
    "copa_es": CopaEsTask,
    "escola": EscolaTask,
    "mgsm_direct_es_spanish_bench": MgsmTask,
    "openbookqa_es": OpenBookQaEsTask,
    "paws_es_spanish_bench": PawsEsTask,
    "teleia_pce": TeleiaPceTask,
    "teleia_cervantes_ave": TeleiaCervantesAveTask,
    "teleia_siele": TeleiaSieleTask,
    "wnli_es": WnliEsTask,
    "xnli_es_spanish_bench": XnliEsTask,
}


@task
def run_spanish(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run Spanish evaluation.
    
    This is a Prefect task that wraps the generic benchmark runner.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info from vLLM (None for cloud providers)
        api_test_result: API test result (unused, for interface compatibility)
        task_config: Task configuration from src/tasks/spanish/task.json
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused for this task)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    logger.info(f"Starting Spanish evaluation for model: {model_name}")
    
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
    output_subdir = task_config.get('output', {}).get('subdirectory', 'spanish')
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of subtasks to run
    subtasks_to_run = task_config.get('tasks', [])
    subtask_configs = task_config.get('task_configs', {})
    defaults = task_config.get('defaults', {})
    prompts = task_config.get('prompts', {})
    
    # Store results for each subtask
    all_results = {}
    all_metrics = {}
    
    try:
        for subtask_name in subtasks_to_run:
            logger.info(f"Running subtask: {subtask_name}")
            
            # Get task class
            if subtask_name not in TASK_CLASSES:
                logger.warning(f"Unknown subtask: {subtask_name}, skipping")
                continue
            
            task_class = TASK_CLASSES[subtask_name]
            
            # Create task instance
            subtask_config = subtask_configs.get(subtask_name, {})
            task_instance = task_class({
                'dataset': subtask_config,
                'prompts': prompts,
                **subtask_config,
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
                "output_dir": str(task_output_path / subtask_name),
                "log_samples": defaults.get('log_samples', False),
            }
            
            # Run benchmark
            runner = BenchmarkRunner(task_instance, interface, runner_config)
            subtask_results = asyncio.run(runner.run(limit=limit, no_resume=False))
            
            # Save results
            save_results(
                results=subtask_results,
                output_dir=task_output_path / subtask_name,
                model_name=model_name,
                task_name=task_instance.get_task_name(),
                log_samples=defaults.get('log_samples', False),
                mark_complete=False,
            )
            
            all_results[subtask_name] = subtask_results
            all_metrics[subtask_name] = subtask_results.get('aggregate_metrics', {})
            
            logger.info(f"Subtask {subtask_name} completed")
        
        # Aggregate metrics across all subtasks (weighted by size)
        aggregated = _aggregate_subtask_metrics(all_metrics, subtasks_to_run)
        
        # Save aggregated summary
        _save_aggregated_summary(
            aggregated_metrics=aggregated,
            subtask_metrics=all_metrics,
            output_dir=task_output_path,
            model_name=model_name,
            subtasks=subtasks_to_run,
        )
        
        # Mark parent task complete
        mark_task_complete(task_output_path)
        
        logger.info("Spanish evaluation completed successfully")
        
        return {
            "model_name": model_name,
            "task": "spanish",
            "output_path": str(task_output_path),
            "metrics": aggregated,
            "subtask_metrics": all_metrics,
        }
        
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        logger.error(f"Check that the endpoint is accessible and responding")
        raise
    except Exception as e:
        logger.error(f"Error running Spanish evaluation: {type(e).__name__}: {e}")
        raise


def _aggregate_subtask_metrics(subtask_metrics: Dict[str, Dict], subtask_names: list) -> Dict:
    """Aggregate metrics across subtasks (weighted by sample size).
    
    Args:
        subtask_metrics: Dictionary mapping subtask names to their metrics
        subtask_names: List of subtask names
        
    Returns:
        Aggregated metrics dictionary
    """
    if not subtask_metrics:
        return {}
    
    total_samples = sum(m.get('total_samples', 0) for m in subtask_metrics.values())
    valid_samples = sum(m.get('valid_samples', 0) for m in subtask_metrics.values())
    
    aggregated = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'error_rate': (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
    }
    
    # Weighted average for accuracy (weighted by sample size)
    valid_subtasks = {k: v for k, v in subtask_metrics.items() if v.get('valid_samples', 0) > 0}
    
    if valid_subtasks:
        total_valid = sum(m.get('valid_samples', 0) for m in valid_subtasks.values())
        
        # Aggregate accuracy
        if any('acc' in m for m in valid_subtasks.values()):
            weighted_sum = sum(
                m.get('acc', 0) * m.get('valid_samples', 0)
                for m in valid_subtasks.values()
                if 'acc' in m
            )
            aggregated['acc'] = weighted_sum / total_valid if total_valid > 0 else 0.0
        
        # Aggregate exact_match (for mgsm)
        if any('exact_match' in m for m in valid_subtasks.values()):
            weighted_sum = sum(
                m.get('exact_match', 0) * m.get('valid_samples', 0)
                for m in valid_subtasks.values()
                if 'exact_match' in m
            )
            aggregated['exact_match'] = weighted_sum / total_valid if total_valid > 0 else 0.0
    else:
        aggregated['acc'] = 0.0
        aggregated['exact_match'] = 0.0
    
    return aggregated


def _save_aggregated_summary(
    aggregated_metrics: Dict,
    subtask_metrics: Dict[str, Dict],
    output_dir: Path,
    model_name: str,
    subtasks: list,
):
    """Save aggregated results summary."""
    write_group_summary(
        output_dir=output_dir,
        model_name=model_name,
        subtasks=subtasks,
        aggregated_metrics=aggregated_metrics,
        subtask_metrics=subtask_metrics,
        title="SPANISH BENCHMARK SUMMARY",
        aggregated_fields=[
            ("total_samples", "Total Samples", "d"),
            ("valid_samples", "Valid Samples", "d"),
            ("error_rate", "Error Rate", ".2%"),
            ("acc", "Accuracy", ".4f"),
            ("exact_match", "Exact Match", ".4f"),
        ],
        per_subtask_fields=[
            ("total_samples", "Samples", "d"),
            ("acc", "Accuracy", ".4f"),
            ("exact_match", "Exact Match", ".4f"),
        ],
    )
