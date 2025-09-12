"""Main entry point for benchy - ZenML-powered ML benchmarking."""

import os
import yaml
from dotenv import load_dotenv
from zenml.logger import get_logger
from src.pipeline import benchmark_pipeline

logger = get_logger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    # Check for environment variable first (used by batch runner)
    if config_path is None:
        config_path = os.environ.get('BENCHY_CONFIG', 'config.yaml')
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model_args(config: dict) -> str:
    """Build model arguments string from config."""
    model = config['model']
    args = [f"pretrained={model['name']}"]
    
    if 'dtype' in model:
        args.append(f"dtype={model['dtype']}")
    if 'max_length' in model:
        args.append(f"max_length={model['max_length']}")
        
    return ",".join(args)


def build_wandb_args(config: dict) -> str:
    """Build wandb arguments string from config."""
    if 'wandb' not in config:
        return ""
    
    wandb = config['wandb']
    args = []
    
    if 'entity' in wandb:
        args.append(f"entity={wandb['entity']}")
    if 'project' in wandb:
        args.append(f"project={wandb['project']}")
        
    return ",".join(args)


def main():
    """Run the benchmark pipeline with configuration from config.yaml."""
    logger.info("Starting benchy - ML model benchmarking with ZenML")
    
    # Load environment variables
    if os.path.exists('.env'):
        load_dotenv('.env')
        logger.info("Loaded environment variables from .env")
    
    # Load configuration
    try:
        config_path = os.environ.get('BENCHY_CONFIG', 'config.yaml')
        config = load_config()
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.error("config.yaml not found. Please create it based on the template.")
        return
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        return
    
    # Build arguments from config
    model_args = build_model_args(config)
    wandb_args = build_wandb_args(config)
    
    # Extract other config values
    eval_config = config['evaluation']
    upload_config = config.get('upload', {})
    venv_config = config.get('venvs', {})
    
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Tasks: {eval_config['tasks']}")
    logger.info(f"Device: {eval_config['device']}")
    
    try:
        # Run the pipeline
        result = benchmark_pipeline(
            model_name=config['model']['name'],
            model_args=model_args,
            tasks=eval_config['tasks'],
            device=eval_config['device'],
            batch_size=eval_config['batch_size'],
            output_path=eval_config['output_path'],
            wandb_args=wandb_args,
            log_samples=eval_config.get('log_samples', True),
            limit=eval_config.get('limit'),
            lm_eval_path=venv_config.get('lm_eval', '/home/mauro/dev/lm-evaluation-harness'),
            upload_script_path=upload_config.get('script_path', '/home/mauro/dev/leaderboard'),
            upload_script_name=upload_config.get('script_name', 'run_pipeline.py')
        )
        
        logger.info("Benchmark pipeline completed successfully!")
        logger.info(f"Results: {result}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
