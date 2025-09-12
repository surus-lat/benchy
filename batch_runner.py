#!/usr/bin/env python3
"""Advanced batch runner using ZenML multi-model pipeline."""

import os
import yaml
from dotenv import load_dotenv
from zenml.logger import get_logger
from src.batch_pipeline import batch_benchmark_pipeline

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
        # Run the batch pipeline
        result = batch_benchmark_pipeline(
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
