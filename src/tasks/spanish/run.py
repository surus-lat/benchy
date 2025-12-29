"""Spanish benchmark - Prefect task entry point.

This is the main entry point for the Spanish task.
It uses the generic benchmark engine to run evaluation.
"""

import logging
from typing import Dict, Any, Optional
from prefect import task

from ..group_runner import TaskGroupSpec, SubtaskContext, run_task_group
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
    return run_task_group(
        spec=SPANISH_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


def _prepare_spanish_task(context: SubtaskContext):
    task_class = TASK_CLASSES.get(context.subtask_name)
    if task_class is None:
        logger.warning(f"Unknown subtask: {context.subtask_name}, skipping")
        return None
    return task_class({
        "dataset": context.subtask_config,
        "prompts": context.prompts,
        **context.subtask_config,
    })


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


def _write_summary(
    aggregated_metrics: Dict,
    subtask_metrics: Dict[str, Dict],
    output_dir,
    model_name: str,
    subtasks: list,
):
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


SPANISH_SPEC = TaskGroupSpec(
    name="spanish",
    display_name="Spanish",
    output_subdir="spanish",
    prepare_task=_prepare_spanish_task,
    aggregate_metrics=_aggregate_subtask_metrics,
    write_summary=_write_summary,
)
