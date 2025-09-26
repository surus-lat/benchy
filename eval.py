"""Main entry point for benchy - vLLM-powered ML benchmarking."""

import os
import sys
import argparse
import yaml
import subprocess
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

# Set Prefect API URL BEFORE importing Prefect modules
# This must be done before any Prefect imports
if 'PREFECT_API_URL' not in os.environ:
    os.environ['PREFECT_API_URL'] = 'http://localhost:4200/api'

from src.pipeline import benchmark_pipeline, test_vllm_server
from prefect import serve
from src.logging_utils import setup_file_logging
from src.config_manager import ConfigManager
import logging



logger = logging.getLogger(__name__)

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




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchy - vLLM-powered ML model benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py                                    # Use default config.yaml
  python eval.py --config configs/my-model.yaml    # Use specific config
  python eval.py -c configs/gemma-e4b.yaml         # Short form
  python eval.py --test                             # Test vLLM server only
  python eval.py -t -c configs/test-model.yaml     # Test with specific config
  python eval.py --register                         # Register flows with Prefect server
  python eval.py --prefect-url http://localhost:4200/api  # Use custom Prefect server
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
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run test pipeline (only start and test vLLM server, no evaluation)'
    )
    
    parser.add_argument(
        '--register', '-r',
        action='store_true',
        help='Register flows with Prefect server for dashboard visibility'
    )
    
    parser.add_argument(
        '--prefect-url',
        type=str,
        default='http://localhost:4200/api',
        help='Prefect API URL (default: http://localhost:4200/api)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of examples per task (useful for testing)'
    )
    
    return parser.parse_args()


def main():
    """Run the vLLM benchmark pipeline with configuration from specified file."""
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting benchy - vLLM-powered ML model benchmarking")
    
    if args.verbose:
        logger.info(f"Command line arguments: {args}")
        logger.info(f"Python path: {sys.executable}")
        logger.info(f"Working directory: {os.getcwd()}")
    
    # Load environment variables
    if os.path.exists('.env'):
        load_dotenv('.env')
        logger.info("Loaded environment variables from .env")
    
    # Override Prefect API URL if specified via command line
    if args.prefect_url:
        os.environ['PREFECT_API_URL'] = args.prefect_url
        logger.info(f"Using Prefect API URL from command line: {args.prefect_url}")
    else:
        logger.info(f"Using Prefect API URL: {os.environ.get('PREFECT_API_URL', 'http://localhost:4200/api')}")
    
    # Load configuration
    try:
        # Use command line config if provided, otherwise fall back to env var or default
        config_path = args.config or os.environ.get('BENCHY_CONFIG', 'config.yaml')
        
        # Use ConfigManager for new format, fallback to old load_config for backward compatibility
        config_manager = ConfigManager()
        try:
            config = config_manager.load_model_config(config_path)
            logger.info(f"Loaded configuration from {config_path} using ConfigManager")
        except (FileNotFoundError, KeyError) as e:
            # Fallback to old config loading for backward compatibility
            logger.info(f"Falling back to legacy config loading: {e}")
            config = load_config(config_path)
            logger.info(f"Loaded legacy configuration from {config_path}")
            
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Load centralized config for global settings
    central_config = load_config('configs/config.yaml')
    
    # Setup file logging using centralized config
    log_dir = central_config['logging']['log_dir']
    
    # Initialize logging system
    log_setup = setup_file_logging(config, log_dir)
    logger.info(f"File logging enabled - log file: {log_setup.get_log_filepath()}")
    
    # Log complete configuration
    log_setup.log_config()
    
    # Extract configuration values
    model_name = config['model']['name']
    vllm_config = config['vllm']
    cuda_devices = vllm_config.get('cuda_devices', None)
    
    # Use centralized paths
    output_path = central_config['paths']['benchmark_outputs']
    
    logger.info(f"Model: {model_name}")
    tasks_to_run = config.get('tasks', ['spanish', 'portuguese'])
    logger.info(f"Tasks to run: {tasks_to_run}")
    logger.info(f"vLLM server: {vllm_config['host']}:{vllm_config['port']}")
    
    # Handle flow registration if requested
    if args.register:
        logger.info("Registering flows with Prefect server for dashboard visibility...")
        # Create a timestamped name for the deployment
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        serve(
            benchmark_pipeline.to_deployment(),
            test_vllm_server.to_deployment(),
            limit=1,  # Only run one instance at a time
            print_starting_message=True
        )
        return
    
    try:
        logger.info(f"Running pipeline")
        if args.test:
            # Run test pipeline (only start and test vLLM server)
            logger.info("Running test pipeline (vLLM server test only)")
            result = test_vllm_server(
                model_name=model_name,
                # vLLM server configuration
                host=vllm_config.get('host', '0.0.0.0'),
                port=vllm_config.get('port', 8000),
                tensor_parallel_size=vllm_config.get('tensor_parallel_size', 1),
                max_model_len=vllm_config.get('max_model_len', 8192),
                gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.6),
                enforce_eager=vllm_config.get('enforce_eager', True),
                limit_mm_per_prompt=vllm_config.get('limit_mm_per_prompt', '{"images": 0, "audios": 0}'),
                hf_cache=vllm_config.get('hf_cache', '/home/mauro/.cache/huggingface'),
                hf_token=vllm_config.get('hf_token', ""),
                startup_timeout=vllm_config.get('startup_timeout', 900),
                cuda_devices=cuda_devices,
                kv_cache_memory=vllm_config.get('kv_cache_memory', None)
            )
        else:
            # Run full benchmark pipeline
            result = benchmark_pipeline(
                model_name=model_name,
                tasks=config.get('tasks', ['spanish', 'portuguese']),  # Default to both tasks
                output_path=output_path,  # Use centralized output path
                limit=args.limit,  # Use command line limit parameter
                use_chat_completions=config.get('use_chat_completions', False),  # Default to False
                # vLLM server configuration
                host=vllm_config.get('host', '0.0.0.0'),
                port=vllm_config.get('port', 8000),
                tensor_parallel_size=vllm_config.get('tensor_parallel_size', 1),
                max_model_len=vllm_config.get('max_model_len', 8192),
                gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.6),
                enforce_eager=vllm_config.get('enforce_eager', True),
                limit_mm_per_prompt=vllm_config.get('limit_mm_per_prompt', '{"images": 0, "audios": 0}'),
                hf_cache=vllm_config.get('hf_cache', '/home/mauro/.cache/huggingface'),
                hf_token=vllm_config.get('hf_token', ""),
                startup_timeout=vllm_config.get('startup_timeout', 900),
                cuda_devices=cuda_devices,
                kv_cache_memory=vllm_config.get('kv_cache_memory', None)
            )
        
        logger.info("Benchmark pipeline completed successfully!")
        logger.info(f"Results: {result}")
        
        # Log summary
        log_setup.log_summary(result)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        
        # Log failure summary - create a simple dict for error cases
        error_result = {
            'model_name': model_name,
            'return_code': 1,
            'error': str(e)
        }
        log_setup.log_summary(error_result)
        raise


if __name__ == "__main__":
    main()