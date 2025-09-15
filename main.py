"""Main entry point for benchy - ZenML-powered ML benchmarking."""

import os
import sys
import argparse
import yaml
import subprocess
import time
import requests
from dotenv import load_dotenv
from zenml.logger import get_logger
from src.pipeline import benchmark_pipeline, create_run_name
from src.logging_utils import setup_file_logging

logger = get_logger(__name__)


def ensure_zenml_server():
    """Ensure ZenML server is running."""
    # Check if ZenML server is already running
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ ZenML server is already running")
            return
    except requests.exceptions.RequestException:
        pass
    
    logger.info("🚀 Starting ZenML server...")
    
    # Check if we need sudo for docker commands
    needs_sudo = False
    try:
        subprocess.run("docker ps", shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        needs_sudo = True
        logger.info("🔐 Docker requires sudo permissions")
    
    # Try docker-compose first, then docker compose
    compose_commands = ["docker-compose", "docker compose"]
    
    for cmd in compose_commands:
        try:
            # Check if command exists
            test_cmd = f"sudo {cmd}" if needs_sudo else cmd
            subprocess.run(f"{test_cmd} --version", shell=True, check=True, 
                         capture_output=True, text=True)
            
            # Remove existing container and start the service
            subprocess.run(f"{test_cmd} rm -f zenml", shell=True, capture_output=True)
            subprocess.run(f"{test_cmd} up -d zenml", shell=True, check=True)
            
            # Wait and verify
            time.sleep(15)
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ ZenML server started successfully")
                return
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    # Fallback to manual docker command
    try:
        docker_cmd = "sudo docker" if needs_sudo else "docker"
        subprocess.run(f"{docker_cmd} rm -f zenml", shell=True, capture_output=True)
        subprocess.run(
            f"{docker_cmd} run -d -p 8080:8080 --name zenml zenmldocker/zenml-server",
            shell=True, check=True
        )
        time.sleep(15)
        logger.info("✅ ZenML server started with docker command")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start ZenML server: {e}")
        raise RuntimeError("Could not start ZenML server. Please start it manually.")


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    # Priority order: 1) Explicit path, 2) Environment variable, 3) Default
    if config_path is None:
        config_path = os.environ.get('BENCHY_CONFIG', 'config.yaml')
    
    # Check if file exists
    if not os.path.exists(config_path):
        available_configs = []
        if os.path.exists('configs'):
            available_configs = [f"configs/{f}" for f in os.listdir('configs') if f.endswith('.yaml')]
        
        error_msg = f"Configuration file '{config_path}' not found."
        if available_configs:
            error_msg += f"\n\nAvailable configurations:\n" + "\n".join(f"  - {cfg}" for cfg in available_configs[:5])
            if len(available_configs) > 5:
                error_msg += f"\n  ... and {len(available_configs) - 5} more in configs/"
        error_msg += f"\n\nTry: python main.py --config <path-to-config.yaml>"
        
        raise FileNotFoundError(error_msg)
    
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchy - ZenML-powered ML model benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use default config.yaml
  python main.py --config configs/my-model.yaml    # Use specific config
  python main.py -c configs/example-with-limit.yaml # Short form
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: config.yaml or BENCHY_CONFIG env var)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Run the benchmark pipeline with configuration from specified file."""
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting benchy - ML model benchmarking with ZenML")
    
    if args.verbose:
        logger.info(f"Command line arguments: {args}")
        logger.info(f"Python path: {sys.executable}")
        logger.info(f"Working directory: {os.getcwd()}")
    
    # Load environment variables
    if os.path.exists('.env'):
        load_dotenv('.env')
        logger.info("Loaded environment variables from .env")
    
    # Load configuration
    try:
        # Use command line config if provided, otherwise fall back to env var or default
        config_path = args.config or os.environ.get('BENCHY_CONFIG', 'config.yaml')
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Ensure ZenML server is running
    ensure_zenml_server()
    
    # Setup file logging
    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', 'logs')
    
    # Initialize logging system
    log_setup = setup_file_logging(config, log_dir)
    logger.info(f"File logging enabled - log file: {log_setup.get_log_filepath()}")
    
    # Log complete configuration
    log_setup.log_config()
    
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
        # Extract performance configuration
        performance_config = config.get('performance', {})
        
        # Create custom run name based on model
        model_name = config['model']['name']
        limit = eval_config.get('limit')
        custom_run_name = create_run_name(model_name, limit)
        logger.info(f"Running pipeline with custom name: {custom_run_name}")
        if limit is not None:
            logger.info(f"TEST run detected - limit set to {limit} examples per task")
        
        # Run the pipeline with custom run name
        result = benchmark_pipeline.with_options(
            run_name=custom_run_name
        )(
            model_name=model_name,
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
            upload_script_name=upload_config.get('script_name', 'run_pipeline.py'),
            use_accelerate=performance_config.get('use_accelerate', False),
            num_gpus=performance_config.get('num_gpus', 1),
            mixed_precision=performance_config.get('mixed_precision', 'no'),
            cache_requests=eval_config.get('cache_requests', True)
        )
        
        logger.info("Benchmark pipeline completed successfully!")
        logger.info(f"Results: {result}")
        
        # Log summary
        log_setup.log_summary(result)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        
        # Log failure summary - create a simple dict for error cases
        error_result = {
            'model_name': config['model']['name'],
            'return_code': 1,
            'error': str(e)
        }
        log_setup.log_summary(error_result)
        raise


if __name__ == "__main__":
    main()
