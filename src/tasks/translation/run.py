"""Translation benchmark - Prefect task entry point.

This is the main entry point for the translation task.
It uses the generic benchmark engine to run evaluation.
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
    mark_task_complete,
)

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
        task_config: Task configuration from configs/tasks/translation.yaml
        limit: Limit number of examples (for testing)
        cuda_devices: CUDA devices (unused for this task)
        provider_config: Provider configuration (for cloud providers)
        
    Returns:
        Dictionary with execution results and metrics
    """
    logger.info(f"Starting translation evaluation for model: {model_name}")
    
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
    output_subdir = task_config.get('output', {}).get('subdirectory', 'translation')
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of subtasks to run
    subtasks_to_run = task_config.get('tasks', ['opus'])
    subtask_configs = task_config.get('task_configs', {})
    defaults = task_config.get('defaults', {})
    prompts = task_config.get('prompts', {})
    
    # Store results for each subtask
    all_results = {}
    all_metrics = {}
    
    try:
        for subtask_name in subtasks_to_run:
            logger.info(f"Running subtask: {subtask_name}")
            
            # Create task instances for this subtask
            subtask_results = _run_subtask(
                subtask_name=subtask_name,
                subtask_config=subtask_configs.get(subtask_name, {}),
                task_config=task_config,
                model_name=model_name,
                connection_info=connection_info,
                provider_type=provider_type,
                task_output_path=task_output_path,
                limit=limit,
                defaults=defaults,
                prompts=prompts,
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
        
        # Mark parent task complete
        mark_task_complete(task_output_path)
        
        logger.info("Translation evaluation completed successfully")
        
        return {
            "model_name": model_name,
            "task": "translation",
            "output_path": str(task_output_path),
            "metrics": aggregated,
            "subtask_metrics": all_metrics,
        }
        
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        logger.error(f"Check that the endpoint is accessible and responding")
        raise
    except Exception as e:
        logger.error(f"Error running translation: {type(e).__name__}: {e}")
        raise


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
        )
    else:
        raise ValueError(f"Unknown subtask: {subtask_name}")


def _run_opus_subtask(
    subtask_config: Dict,
    model_name: str,
    connection_info: Dict,
    provider_type: str,
    task_output_path: Path,
    limit: Optional[int],
    defaults: Dict,
    prompts: Dict,
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
                        "reason": "Download completed but no language pairs available",
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
                    "reason": f"Download failed: {e}",
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
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_")
    
    # JSON summary
    summary_file = output_dir / f"{safe_name}_{timestamp}_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "subtasks": subtasks,
            "aggregated_metrics": aggregated_metrics,
            "per_subtask_metrics": subtask_metrics,
        }, f, indent=2)
    
    logger.info(f"Saved summary to {summary_file}")
    
    # Text summary
    text_file = output_dir / f"{safe_name}_{timestamp}_summary.txt"
    with open(text_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TRANSLATION BENCHMARK SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Subtasks: {', '.join(subtasks)}\n\n")
        
        f.write("AGGREGATED METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {aggregated_metrics.get('total_samples', 0)}\n")
        f.write(f"Valid Samples: {aggregated_metrics.get('valid_samples', 0)}\n")
        f.write(f"Error Rate: {aggregated_metrics.get('error_rate', 0):.2%}\n\n")
        f.write(f"BLEU: {aggregated_metrics.get('bleu', 0):.4f}\n")
        f.write(f"chrF: {aggregated_metrics.get('chrf', 0):.4f}\n")
        f.write(f"COMET: {aggregated_metrics.get('comet', 0):.4f}\n\n")
        
        f.write("PER-SUBTASK BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for name, metrics in subtask_metrics.items():
            f.write(f"\n{name.upper()}:\n")
            f.write(f"  Samples: {metrics.get('total_samples', 0)}\n")
            f.write(f"  BLEU: {metrics.get('bleu', 0):.4f}\n")
            f.write(f"  chrF: {metrics.get('chrf', 0):.4f}\n")
            f.write(f"  COMET: {metrics.get('comet', 0):.4f}\n")
        f.write("=" * 60 + "\n")
    
    logger.info(f"Saved text summary to {text_file}")

