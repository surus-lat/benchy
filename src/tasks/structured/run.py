"""Structured data extraction benchmark - Prefect task entry point.

This is the main entry point for the structured extraction task.
It uses the generic benchmark engine to run evaluation.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from prefect import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
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
    return run_task_group(
        spec=STRUCTURED_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


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
            'capability_requirements': task_config.get('capability_requirements', {}),
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
            'capability_requirements': task_config.get('capability_requirements', {}),
        })
        
    else:
        raise ValueError(f"Unknown subtask: {subtask_name}")


def _prepare_structured_task(context: SubtaskContext):
    return _create_task_instance(
        subtask_name=context.subtask_name,
        subtask_config=context.subtask_config,
        task_config=context.task_config,
    )


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


STRUCTURED_SPEC = TaskGroupSpec(
    name="structured_extraction",
    display_name="Structured extraction",
    output_subdir="structured_extraction",
    default_subtasks=["paraloq"],
    prepare_task=_prepare_structured_task,
    aggregate_metrics=_aggregate_subtask_metrics,
    write_summary=_save_aggregated_summary,
)
