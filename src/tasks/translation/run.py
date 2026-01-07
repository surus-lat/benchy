"""Translation benchmark - Prefect task entry point.

This is the main entry point for the translation task.
It uses the generic benchmark engine to run evaluation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from ...prefect_compat import task

from ...engine import (
    BenchmarkRunner,
    save_results,
    get_interface_for_provider,
)
from ..group_runner import (
    TaskGroupSpec,
    SubtaskContext,
    TaskGroupContext,
    ensure_task_interface_compatibility,
    run_task_group,
)
from ..summary_reporter import write_group_summary
from .metrics import load_comet_model

logger = logging.getLogger(__name__)

# Data and cache directories relative to this module
DATA_DIR = Path(__file__).parent / '.data'
CACHE_DIR = Path(__file__).parent / 'cache'


@task
def run_translation(
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run translation evaluation.
    
    This is a Prefect task that wraps the generic benchmark runner.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Server info from vLLM (None for cloud providers)
        api_test_result: API test result (unused, for interface compatibility)
        task_config: Task configuration from src/tasks/translation/task.json
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused for this task)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    return run_task_group(
        spec=TRANSLATION_SPEC,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


def _setup_translation(context: TaskGroupContext) -> Dict[str, Any]:
    comet_model = load_comet_model()
    if comet_model is None:
        return {}
    return {"comet_model": comet_model}


def _run_subtask(
    subtask_name: str,
    subtask_config: Dict,
    task_config: Dict,
    model_name: str,
    connection_info: Dict,
    provider_type: str,
    task_output_path: Path,
    limit: Optional[int],
    defaults: Dict,
    prompts: Dict,
    shared: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single subtask (opus or flores).
    
    Args:
        subtask_name: Name of subtask ("opus" or "flores")
        subtask_config: Configuration for this subtask
        task_config: Full task configuration
        model_name: Model name
        connection_info: Connection info for interface
        provider_type: Provider type
        task_output_path: Base output path
        limit: Sample limit
        defaults: Default settings
        prompts: Prompt templates
        
    Returns:
        Results dictionary
    """
    shared = shared or {}
    comet_model = shared.get("comet_model")

    capability_requirements = task_config.get("capability_requirements", {})

    if subtask_name == "opus":
        return _run_opus_subtask(
            subtask_config=subtask_config,
            model_name=model_name,
            connection_info=connection_info,
            provider_type=provider_type,
            task_output_path=task_output_path,
            limit=limit,
            defaults=defaults,
            prompts=prompts,
            comet_model=comet_model,
            capability_requirements=capability_requirements,
        )
    elif subtask_name == "flores":
        return _run_flores_subtask(
            subtask_config=subtask_config,
            model_name=model_name,
            connection_info=connection_info,
            provider_type=provider_type,
            task_output_path=task_output_path,
            limit=limit,
            defaults=defaults,
            prompts=prompts,
            comet_model=comet_model,
            capability_requirements=capability_requirements,
        )
    else:
        raise ValueError(f"Unknown subtask: {subtask_name}")


def _run_translation_subtask(context: SubtaskContext) -> Dict[str, Any]:
    return _run_subtask(
        subtask_name=context.subtask_name,
        subtask_config=context.subtask_config,
        task_config=context.task_config,
        model_name=context.model_name,
        connection_info=context.connection_info,
        provider_type=context.provider_type,
        task_output_path=context.output_dir,
        limit=context.limit,
        defaults=context.defaults,
        prompts=context.prompts,
        shared=context.shared,
    )


def _run_opus_subtask(
    subtask_config: Dict,
    model_name: str,
    connection_info: Dict,
    provider_type: str,
    task_output_path: Path,
    limit: Optional[int],
    defaults: Dict,
    prompts: Dict,
    comet_model: Optional[Any] = None,
    capability_requirements: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run OPUS subtask for all language pairs."""
    from .datasets.opus.task import OpusTask
    from .datasets.opus.download import download_and_preprocess_opus
    
    language_pairs = subtask_config.get('language_pairs', ['en-es', 'en-pt'])
    dataset_name = subtask_config.get('dataset_name', 'Helsinki-NLP/opus-100')
    
    # Ensure data is downloaded
    opus_data_dir = DATA_DIR / 'opus'
    opus_data_dir.mkdir(parents=True, exist_ok=True)
    
    for pair in language_pairs:
        pair_file = opus_data_dir / f"{pair}.jsonl"
        if not pair_file.exists():
            logger.info(f"Downloading OPUS data for {pair}...")
            download_and_preprocess_opus(
                dataset_name=dataset_name,
                language_pairs=[pair],
                output_dir=opus_data_dir,
                cache_dir=str(CACHE_DIR),
                split='test',  # Use test split for evaluation
            )
    
    # Aggregate results across all language pairs
    all_pair_results = {}
    all_pair_metrics = {}
    
    for pair in language_pairs:
        pair_file = opus_data_dir / f"{pair}.jsonl"
        if not pair_file.exists():
            logger.warning(f"Data file not found for {pair}, skipping")
            continue
        
        logger.info(f"Running OPUS evaluation for {pair}")
        
        # Create task instance
        task_instance = OpusTask({
            'dataset': {'data_file': str(pair_file)},
            'prompts': prompts,
            'language_pair': pair,
            'comet_model': comet_model,
            'capability_requirements': capability_requirements or {},
        })
        
        # Create interface
        interface = get_interface_for_provider(
            provider_type=provider_type,
            connection_info=connection_info,
            model_name=model_name,
        )
        report = ensure_task_interface_compatibility(task_instance, interface)
        if not report.compatible:
            reason = ", ".join(report.errors) if report.errors else "incompatible capabilities"
            logger.warning(f"Skipping OPUS subtask due to incompatibility: {reason}")
            return {
                "subtask": "opus",
                "language_pairs": language_pairs,
                "per_pair_results": {},
                "aggregate_metrics": {
                    "total_samples": 0,
                    "valid_samples": 0,
                    "bleu": 0.0,
                    "chrf": 0.0,
                    "comet": 0.0,
                    "error_rate": 0.0,
                },
                "skipped": True,
                "skip_reason": reason,
            }
        
        # Create runner config
        runner_config = {
            "model_name": model_name,
            "batch_size": defaults.get('batch_size', 20),
            "output_dir": str(task_output_path / 'opus' / pair),
            "log_samples": defaults.get('log_samples', False),
        }
        
        # Run benchmark
        runner = BenchmarkRunner(task_instance, interface, runner_config)
        pair_results = asyncio.run(runner.run(limit=limit, no_resume=False))
        
        # Save results
        save_results(
            results=pair_results,
            output_dir=task_output_path / 'opus' / pair,
            model_name=model_name,
            task_name=task_instance.get_task_name(),
            log_samples=defaults.get('log_samples', False),
            mark_complete=False,
        )
        
        all_pair_results[pair] = pair_results
        all_pair_metrics[pair] = pair_results.get('aggregate_metrics', {})
    
    # Aggregate across pairs
    aggregated = _aggregate_pair_metrics(all_pair_metrics, language_pairs)
    
    return {
        "subtask": "opus",
        "language_pairs": language_pairs,
        "per_pair_results": all_pair_results,
        "aggregate_metrics": aggregated,
    }


def _run_flores_subtask(
    subtask_config: Dict,
    model_name: str,
    connection_info: Dict,
    provider_type: str,
    task_output_path: Path,
    limit: Optional[int],
    defaults: Dict,
    prompts: Dict,
    comet_model: Optional[Any] = None,
    capability_requirements: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run FLORES subtask for all language pairs."""
    from .datasets.flores.task import FloresTask
    from .datasets.flores.download import download_and_preprocess_flores
    
    # Get language pairs - either from config or auto-detect
    language_pairs = subtask_config.get('language_pairs', [])
    dataset_name = subtask_config.get('dataset_name', 'openlanguagedata/flores_plus')
    split = subtask_config.get('split', 'devtest')
    
    # Ensure data is downloaded
    flores_data_dir = DATA_DIR / 'flores'
    flores_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect language pairs if not specified
    if not language_pairs:
        # Look for existing data directories
        if flores_data_dir.exists():
            language_pairs = [d.name for d in flores_data_dir.iterdir() if d.is_dir()]
            # Filter to only directories that have the required split file
            language_pairs = [
                pair for pair in language_pairs
                if (flores_data_dir / pair / f"{split}.jsonl").exists()
            ]
        
        # If no data found, automatically download/preprocess
        if not language_pairs:
            logger.info("No FLORES data found. Downloading and preprocessing dataset...")
            logger.info("This may take a few minutes on first run...")
            try:
                # Download all language pairs (None = all pairs)
                counts = download_and_preprocess_flores(
                    dataset_name=dataset_name,
                    language_pairs=None,  # None = generate all pairs
                    output_dir=flores_data_dir,
                    cache_dir=str(CACHE_DIR),
                )
                
                if counts:
                    logger.info(f"Successfully downloaded {len(counts)} language pairs")
                    # Auto-detect the pairs that were created
                    language_pairs = list(counts.keys())
                else:
                    logger.warning("Download completed but no language pairs were created")
                    logger.info("Skipping FLORES subtask - no data available")
                    return {
                        "subtask": "flores",
                        "language_pairs": [],
                        "split": split,
                        "per_pair_results": {},
                        "aggregate_metrics": {
                            "total_samples": 0,
                            "valid_samples": 0,
                            "bleu": 0.0,
                            "chrf": 0.0,
                            "comet": 0.0,
                            "error_rate": 0.0,
                        },
                        "skipped": True,
                        "skip_reason": "Download completed but no language pairs available",
                    }
            except Exception as e:
                logger.error(f"Error downloading FLORES dataset: {e}")
                logger.info("Skipping FLORES subtask - download failed")
                return {
                    "subtask": "flores",
                    "language_pairs": [],
                    "split": split,
                    "per_pair_results": {},
                    "aggregate_metrics": {
                        "total_samples": 0,
                        "valid_samples": 0,
                        "bleu": 0.0,
                        "chrf": 0.0,
                        "comet": 0.0,
                        "error_rate": 0.0,
                    },
                    "skipped": True,
                    "skip_reason": f"Download failed: {e}",
                }
    
    # Download/preprocess missing pairs if needed
    for pair in language_pairs:
        pair_dir = flores_data_dir / pair
        data_file = pair_dir / f"{split}.jsonl"
        if not data_file.exists():
            logger.info(f"Downloading FLORES data for {pair}...")
            try:
                download_and_preprocess_flores(
                    dataset_name=dataset_name,
                    language_pairs=[pair],
                    output_dir=flores_data_dir,
                    cache_dir=str(CACHE_DIR),
                )
            except Exception as e:
                logger.warning(f"Failed to download {pair}: {e}")
                continue
    
    # Aggregate results across all language pairs
    all_pair_results = {}
    all_pair_metrics = {}
    
    for pair in language_pairs:
        pair_dir = flores_data_dir / pair
        data_file = pair_dir / f"{split}.jsonl"
        if not data_file.exists():
            logger.warning(f"Data file not found for {pair}, skipping")
            continue
        
        logger.info(f"Running FLORES evaluation for {pair} ({split})")
        
        # Create task instance
        task_instance = FloresTask({
            'dataset': {'data_dir': str(flores_data_dir), 'language_pair': pair},
            'prompts': prompts,
            'language_pair': pair,
            'split': split,
            'comet_model': comet_model,
            'capability_requirements': capability_requirements or {},
        })
        
        # Create interface
        interface = get_interface_for_provider(
            provider_type=provider_type,
            connection_info=connection_info,
            model_name=model_name,
        )
        report = ensure_task_interface_compatibility(task_instance, interface)
        if not report.compatible:
            reason = ", ".join(report.errors) if report.errors else "incompatible capabilities"
            logger.warning(f"Skipping FLORES subtask due to incompatibility: {reason}")
            return {
                "subtask": "flores",
                "language_pairs": language_pairs,
                "split": split,
                "per_pair_results": {},
                "aggregate_metrics": {
                    "total_samples": 0,
                    "valid_samples": 0,
                    "bleu": 0.0,
                    "chrf": 0.0,
                    "comet": 0.0,
                    "error_rate": 0.0,
                },
                "skipped": True,
                "skip_reason": reason,
            }
        
        # Create runner config
        runner_config = {
            "model_name": model_name,
            "batch_size": defaults.get('batch_size', 20),
            "output_dir": str(task_output_path / 'flores' / pair),
            "log_samples": defaults.get('log_samples', False),
        }
        
        # Run benchmark
        runner = BenchmarkRunner(task_instance, interface, runner_config)
        pair_results = asyncio.run(runner.run(limit=limit, no_resume=False))
        
        # Save results
        save_results(
            results=pair_results,
            output_dir=task_output_path / 'flores' / pair,
            model_name=model_name,
            task_name=task_instance.get_task_name(),
            log_samples=defaults.get('log_samples', False),
            mark_complete=False,
        )
        
        all_pair_results[pair] = pair_results
        all_pair_metrics[pair] = pair_results.get('aggregate_metrics', {})
    
    # Aggregate across pairs
    aggregated = _aggregate_pair_metrics(all_pair_metrics, language_pairs)
    
    return {
        "subtask": "flores",
        "language_pairs": language_pairs,
        "split": split,
        "per_pair_results": all_pair_results,
        "aggregate_metrics": aggregated,
    }


def _aggregate_pair_metrics(pair_metrics: Dict[str, Dict], pairs: list) -> Dict:
    """Aggregate metrics across language pairs."""
    if not pair_metrics:
        return {}
    
    total_samples = sum(m.get('total_samples', 0) for m in pair_metrics.values())
    valid_samples = sum(m.get('valid_samples', 0) for m in pair_metrics.values())
    
    aggregated = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'error_rate': (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
    }
    
    # Weighted average for metrics
    valid_pairs = {k: v for k, v in pair_metrics.items() if v.get('valid_samples', 0) > 0}
    
    if valid_pairs:
        total_valid = sum(m.get('valid_samples', 0) for m in valid_pairs.values())
        
        for metric in ['bleu', 'chrf', 'comet']:
            weighted_sum = sum(
                m.get(metric, 0) * m.get('valid_samples', 0)
                for m in valid_pairs.values()
            )
            aggregated[metric] = weighted_sum / total_valid if total_valid > 0 else 0.0
    else:
        for metric in ['bleu', 'chrf', 'comet']:
            aggregated[metric] = 0.0
    
    return aggregated


def _aggregate_subtask_metrics(subtask_metrics: Dict[str, Dict], subtask_names: list) -> Dict:
    """Aggregate metrics across subtasks (opus, flores)."""
    if not subtask_metrics:
        return {}
    
    total_samples = sum(m.get('total_samples', 0) for m in subtask_metrics.values())
    valid_samples = sum(m.get('valid_samples', 0) for m in subtask_metrics.values())
    
    aggregated = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'error_rate': (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
    }
    
    # Weighted average for metrics
    valid_subtasks = {k: v for k, v in subtask_metrics.items() if v.get('valid_samples', 0) > 0}
    
    if valid_subtasks:
        total_valid = sum(m.get('valid_samples', 0) for m in valid_subtasks.values())
        
        for metric in ['bleu', 'chrf', 'comet']:
            weighted_sum = sum(
                m.get(metric, 0) * m.get('valid_samples', 0)
                for m in valid_subtasks.values()
            )
            aggregated[metric] = weighted_sum / total_valid if total_valid > 0 else 0.0
    else:
        for metric in ['bleu', 'chrf', 'comet']:
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
    write_group_summary(
        output_dir=output_dir,
        model_name=model_name,
        subtasks=subtasks,
        aggregated_metrics=aggregated_metrics,
        subtask_metrics=subtask_metrics,
        title="TRANSLATION BENCHMARK SUMMARY",
        aggregated_fields=[
            ("total_samples", "Total Samples", "d"),
            ("valid_samples", "Valid Samples", "d"),
            ("error_rate", "Error Rate", ".2%"),
            ("bleu", "BLEU", ".4f"),
            ("chrf", "chrF", ".4f"),
            ("comet", "COMET", ".4f"),
        ],
        per_subtask_fields=[
            ("total_samples", "Samples", "d"),
            ("bleu", "BLEU", ".4f"),
            ("chrf", "chrF", ".4f"),
            ("comet", "COMET", ".4f"),
        ],
    )


TRANSLATION_SPEC = TaskGroupSpec(
    name="translation",
    display_name="Translation",
    output_subdir="translation",
    default_subtasks=["opus"],
    run_subtask=_run_translation_subtask,
    aggregate_metrics=_aggregate_subtask_metrics,
    write_summary=_save_aggregated_summary,
    setup=_setup_translation,
)
