"""ZenML pipeline for batch evaluation of multiple models."""

from typing import List, Dict, Any
from zenml import pipeline, step
from zenml.logger import get_logger
from .steps import run_lm_evaluation, upload_results

logger = get_logger(__name__)


@step
def prepare_model_configs(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare model configurations for batch processing.
    
    Args:
        models: List of model configuration dictionaries
        
    Returns:
        List of prepared model configurations
    """
    logger.info(f"Preparing {len(models)} model configurations")
    
    prepared_configs = []
    for i, model_config in enumerate(models):
        # Add index and validate required fields
        config = model_config.copy()
        config['batch_index'] = i
        config['total_models'] = len(models)
        
        # Validate required fields
        required_fields = ['name', 'dtype', 'tasks', 'device', 'batch_size', 'output_path']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Model config {i} missing required field: {field}")
        
        # Optional fields with defaults
        config.setdefault('limit', None)
                
        prepared_configs.append(config)
        logger.info(f"Prepared config {i+1}/{len(models)}: {config['name']}")
        
    return prepared_configs


@step
def collect_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collect and summarize results from multiple model evaluations.
    
    Args:
        results: List of results from each model evaluation
        
    Returns:
        Summary of all results
    """
    logger.info(f"Collecting results from {len(results)} model evaluations")
    
    summary = {
        'total_models': len(results),
        'successful': 0,
        'failed': 0,
        'models': {},
        'errors': []
    }
    
    for result in results:
        model_name = result.get('model_name', 'unknown')
        if result.get('upload_return_code') == 0:
            summary['successful'] += 1
            summary['models'][model_name] = 'success'
        else:
            summary['failed'] += 1
            summary['models'][model_name] = 'failed'
            summary['errors'].append({
                'model': model_name,
                'error': result.get('upload_stderr', 'Unknown error')
            })
    
    logger.info(f"Batch evaluation complete: {summary['successful']} successful, {summary['failed']} failed")
    return summary


@pipeline
def batch_benchmark_pipeline(
    models: List[Dict[str, Any]],
    wandb_args: str = "",
    log_samples: bool = True,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    upload_script_path: str = "/home/mauro/dev/leaderboard",
    upload_script_name: str = "run_pipeline.py"
):
    """
    Batch benchmarking pipeline for multiple models.
    
    Args:
        models: List of model configurations
        wandb_args: Weights & Biases arguments
        log_samples: Whether to log samples
        lm_eval_path: Path to lm-evaluation-harness installation
        upload_script_path: Path to upload script directory
        upload_script_name: Name of upload script
    """
    logger.info(f"Starting batch benchmark pipeline for {len(models)} models")
    
    # Prepare configurations
    prepared_configs = prepare_model_configs(models)
    
    # Run evaluation for each model
    all_results = []
    for config in prepared_configs:
        logger.info(f"Processing model {config['batch_index']+1}/{config['total_models']}: {config['name']}")
        
        # Build model args
        model_args = f"pretrained={config['name']}"
        if 'dtype' in config:
            model_args += f",dtype={config['dtype']}"
        if 'max_length' in config:
            model_args += f",max_length={config['max_length']}"
        
        # Step 1: Run evaluation
        eval_result = run_lm_evaluation(
            model_name=config['name'],
            model_args=model_args,
            tasks=config['tasks'],
            device=config['device'],
            batch_size=config['batch_size'],
            output_path=config['output_path'],
            wandb_args=wandb_args,
            log_samples=log_samples,
            limit=config.get('limit'),
            lm_eval_path=lm_eval_path
        )
        
        # Step 2: Upload results
        upload_result = upload_results(
            eval_results=eval_result,
            script_path=upload_script_path,
            script_name=upload_script_name
        )
        
        all_results.append(upload_result)
    
    # Collect and summarize results
    summary = collect_results(all_results)
    
    logger.info("Batch benchmark pipeline completed")
    return summary
