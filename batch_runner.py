#!/usr/bin/env python3
"""Advanced batch runner using ZenML multi-model pipeline."""

import os
import yaml
from dotenv import load_dotenv
from zenml.logger import get_logger
from src.batch_pipeline import batch_benchmark_pipeline
from src.pipeline import create_run_name

logger = get_logger(__name__)

def load_batch_config(config_path: str = "batch_config.yaml") -> dict:
    """Load batch configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Run batch evaluation using ZenML multi-model pipeline."""
    logger.info("Starting advanced batch benchmarking with ZenML")
    
    # Load environment variables
    if os.path.exists('.env'):
        load_dotenv('.env')
        logger.info("Loaded environment variables from .env")
    
    # Load batch configuration
    try:
        batch_config = load_batch_config()
        logger.info("Loaded batch configuration from batch_config.yaml")
    except FileNotFoundError:
        logger.error("batch_config.yaml not found. Please create it.")
        return
    except Exception as e:
        logger.error(f"Error loading batch_config.yaml: {e}")
        return
    
    # Extract configuration
    models = batch_config['models']
    common_config = batch_config.get('common', {})
    
    logger.info(f"Planning to evaluate {len(models)} models")
    for i, model in enumerate(models, 1):
        logger.info(f"  {i}. {model['name']}")
    
    # Build wandb args if present
    wandb_args = ""
    if 'wandb' in common_config:
        wandb = common_config['wandb']
        wandb_parts = []
        if 'entity' in wandb:
            wandb_parts.append(f"entity={wandb['entity']}")
        if 'project' in wandb:
            wandb_parts.append(f"project={wandb['project']}")
        wandb_args = ",".join(wandb_parts)
    
    try:
        # Create custom run name for batch (use first model name as representative)
        first_model_name = models[0]['name'] if models else "batch"
        # Check if any model has a limit set (indicating test mode)
        has_limit = any(model.get('limit') is not None for model in models)
        batch_run_name = f"batch_{create_run_name(first_model_name, limit=1 if has_limit else None).replace(first_model_name.split('/')[-1], f'{len(models)}_models')}"
        logger.info(f"Running batch pipeline with custom name: {batch_run_name}")
        if has_limit:
            logger.info("TEST batch detected - one or more models have limit set")
        
        # Run the batch pipeline with custom run name
        result = batch_benchmark_pipeline.with_options(
            run_name=batch_run_name
        )(
            models=models,
            wandb_args=wandb_args,
            log_samples=common_config.get('log_samples', True),
            lm_eval_path=common_config.get('lm_eval_path', '/home/mauro/dev/lm-evaluation-harness'),
            upload_script_path=common_config.get('upload_script_path', '/home/mauro/dev/leaderboard'),
            upload_script_name=common_config.get('upload_script_name', 'run_pipeline.py')
        )
        
        logger.info("Batch pipeline completed successfully!")
        logger.info(f"Summary: {result}")
        
    except Exception as e:
        logger.error(f"Batch pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
