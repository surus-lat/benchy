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

# Load environment variables from .env file
load_dotenv()

# Set Prefect API URL BEFORE importing Prefect modules
# This must be done before any Prefect imports
if 'PREFECT_API_URL' not in os.environ:
    os.environ['PREFECT_API_URL'] = 'http://localhost:4200/api'

from src.pipeline import benchmark_pipeline, test_vllm_server
from prefect import serve
from src.logging_utils import setup_file_logging
from src.config_manager import ConfigManager
from src.gpu_config import load_gpu_config
from src.run_id_manager import generate_run_id, get_run_paths, setup_run_directories, get_prefect_flow_name
import logging
import signal
import sys


logger = logging.getLogger(__name__)

# Global state for signal handling
_active_server_info = None

def signal_handler(signum, frame):
    """Handle termination signals by cleaning up processes."""
    logger.info(f"Received signal {signum}. Cleaning up...")
    
    # Import here to avoid circular dependency
    from src.inference.vllm_server import stop_vllm_server, kill_lm_eval_processes
    
    # Kill lm-harness processes first
    try:
        kill_lm_eval_processes()
    except Exception as e:
        logger.warning(f"Error killing lm_eval processes: {e}")
    
    # Stop vLLM server
    if _active_server_info is not None:
        try:
            logger.info("Stopping vLLM server...")
            stop_vllm_server(_active_server_info, {})
        except Exception as e:
            logger.warning(f"Error stopping vLLM server: {e}")
    
    logger.info("Cleanup complete. Exiting.")
    sys.exit(130)  # Standard exit code for SIGINT

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
  python eval.py --log-samples                     # Enable sample logging for all tasks
  python eval.py --no-log-samples                  # Disable sample logging for all tasks
  python eval.py --run-id my_experiment_001         # Use custom run ID for outputs
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
    
    parser.add_argument(
        '--log-samples',
        action='store_true',
        help='Enable sample logging for all tasks (overrides task config defaults)'
    )
    
    parser.add_argument(
        '--no-log-samples',
        action='store_true',
        help='Disable sample logging for all tasks (overrides task config defaults)'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID for organizing outputs (default: auto-generated timestamp)'
    )
    
    return parser.parse_args()


def main():
    """Run the vLLM benchmark pipeline with configuration from specified file."""
    # Parse command line arguments
    args = parse_args()
    
    # Validate mutually exclusive log_samples arguments
    if args.log_samples and args.no_log_samples:
        logger.error("Cannot specify both --log-samples and --no-log-samples")
        return
    
    logger.info("Starting benchy - vLLM-powered ML model benchmarking")
    
    # Register signal handlers for clean termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
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
    
    # Load GPU configuration from central config
    gpu_manager = load_gpu_config(central_config)
    
    # Log GPU configuration
    gpu_summary = gpu_manager.get_config_summary()
    logger.info(f"GPU Configuration: {gpu_summary}")
    
    # Generate run ID using centralized logic
    run_id = generate_run_id(
        custom_run_id=args.run_id,
        is_test=args.test,
        is_limited=args.limit is not None
    )
    logger.info(f"Generated run ID: {run_id}")
    
    # Get standardized paths for this run
    run_paths = get_run_paths(run_id, central_config['paths']['benchmark_outputs'], central_config['logging']['log_dir'])
    setup_run_directories(run_paths)
    
    # Setup file logging using centralized config and run_id
    log_setup = setup_file_logging(config, central_config['logging']['log_dir'], run_id)
    logger.info(f"File logging enabled - log file: {log_setup.get_log_filepath()}")
    
    # Log complete configuration
    log_setup.log_config()
    
    # Extract configuration values
    model_name = config['model']['name']
    provider_type = config.get('provider_type', 'vllm')
    
    # Get provider-specific config based on type
    if provider_type == 'vllm':
        provider_config = config.get('vllm', {})
        vllm_config = provider_config  # Backward compatibility
        # Use GPU configuration from central config, with vLLM config override
        cuda_devices = vllm_config.get('cuda_devices', gpu_manager.get_vllm_cuda_devices())
    elif provider_type == 'openai':
        provider_config = config.get('openai', {})
        vllm_config = None
        cuda_devices = None
        logger.info(f"Using OpenAI cloud provider for model: {model_name}")
    elif provider_type == 'anthropic':
        provider_config = config.get('anthropic', {})
        vllm_config = None
        cuda_devices = None
        logger.info(f"Using Anthropic cloud provider for model: {model_name}")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    # Prepare task defaults overrides
    task_defaults_overrides = {}
    
    # Command line overrides take priority
    if args.log_samples:
        task_defaults_overrides['log_samples'] = True
        logger.info("Command line override: log_samples = True")
    elif args.no_log_samples:
        task_defaults_overrides['log_samples'] = False
        logger.info("Command line override: log_samples = False")
    
    # Merge with config-based task defaults
    config_task_defaults = config.get('task_defaults', {})
    if config_task_defaults:
        logger.info(f"Using task defaults from config: {config_task_defaults}")
        # Command line args override config values
        merged_overrides = {**config_task_defaults, **task_defaults_overrides}
        task_defaults_overrides = merged_overrides
    
    if task_defaults_overrides:
        logger.info(f"Final task defaults overrides: {task_defaults_overrides}")
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Provider type: {provider_type}")
    tasks_to_run = config.get('tasks', ['spanish', 'portuguese'])
    
    # Expand task groups into individual tasks
    expanded_tasks = config_manager.expand_task_groups(tasks_to_run, central_config)
    logger.info(f"Task expansion: {tasks_to_run} -> {expanded_tasks}")
    tasks_to_run = expanded_tasks
    
    logger.info(f"Tasks to run: {tasks_to_run}")
    if provider_type == 'vllm' and vllm_config:
        logger.info(f"vLLM server: {vllm_config['host']}:{vllm_config['port']}")
    elif provider_type in ['openai', 'anthropic']:
        logger.info(f"Cloud provider: {provider_type}")
        logger.info(f"Base URL: {provider_config.get('base_url', 'N/A')}")
    logger.info(f"Output path: {run_paths['output_path']}")
    logger.info(f"Log path: {run_paths['log_path']}")
    
    # Handle flow registration if requested
    if args.register:
        logger.info("Registering flows with Prefect server for dashboard visibility...")
        # Set flow names with run_id prefix
        benchmark_pipeline.name = get_prefect_flow_name("benchmark_pipeline", run_id)
        test_vllm_server.name = get_prefect_flow_name("test_vllm_server", run_id)
        
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
            # Run test pipeline (only for vLLM)
            if provider_type != 'vllm':
                logger.error(f"Test mode is only supported for vLLM provider, not {provider_type}")
                return
            
            logger.info("Running test pipeline (vLLM server test only)")
            result = test_vllm_server(
                model_name=model_name,
                run_id=run_id,
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
                kv_cache_memory=vllm_config.get('kv_cache_memory', None),
                vllm_venv_path=vllm_config.get('vllm_venv_path', '/home/mauro/dev/benchy/.venv'),
                vllm_version=vllm_config.get('vllm_version', None),
                multimodal=vllm_config.get('multimodal', True),
                max_num_seqs=vllm_config.get('max_num_seqs', None),
                max_num_batched_tokens=vllm_config.get('max_num_batched_tokens', None)
            )
        else:
            # Run full benchmark pipeline
            # Prepare vLLM-specific parameters (only used if provider_type == 'vllm')
            vllm_params = {}
            if provider_type == 'vllm' and vllm_config:
                vllm_params = {
                    'host': vllm_config.get('host', '0.0.0.0'),
                    'port': vllm_config.get('port', 8000),
                    'tensor_parallel_size': vllm_config.get('tensor_parallel_size', 1),
                    'max_model_len': vllm_config.get('max_model_len', 8192),
                    'gpu_memory_utilization': vllm_config.get('gpu_memory_utilization', 0.6),
                    'enforce_eager': vllm_config.get('enforce_eager', True),
                    'limit_mm_per_prompt': vllm_config.get('limit_mm_per_prompt', '{"images": 0, "audios": 0}'),
                    'hf_cache': vllm_config.get('hf_cache', '/home/mauro/.cache/huggingface'),
                    'hf_token': vllm_config.get('hf_token', ""),
                    'startup_timeout': vllm_config.get('startup_timeout', 900),
                    'cuda_devices': cuda_devices,
                    'kv_cache_memory': vllm_config.get('kv_cache_memory', None),
                    'vllm_venv_path': vllm_config.get('vllm_venv_path', '/home/mauro/dev/benchy/.venv'),
                    'vllm_version': vllm_config.get('vllm_version', None),
                    'multimodal': vllm_config.get('multimodal', True),
                    'max_num_seqs': vllm_config.get('max_num_seqs', None),
                    'max_num_batched_tokens': vllm_config.get('max_num_batched_tokens', None),
                    'trust_remote_code': vllm_config.get('trust_remote_code', True),
                    'tokenizer_mode': vllm_config.get('tokenizer_mode', None),
                    'config_format': vllm_config.get('config_format', None),
                    'load_format': vllm_config.get('load_format', None),
                    'tool_call_parser': vllm_config.get('tool_call_parser', None),
                    'enable_auto_tool_choice': vllm_config.get('enable_auto_tool_choice', False),
                }
            
            result = benchmark_pipeline(
                model_name=model_name,
                tasks=tasks_to_run,  # Use expanded task list
                output_path=run_paths['output_path'],  # Use run-specific output path
                limit=args.limit,  # Use command line limit parameter
                use_chat_completions=config.get('use_chat_completions', False),  # Default to False
                task_defaults_overrides=task_defaults_overrides or None,  # Pass task overrides
                log_setup=log_setup,  # Pass log setup for task config logging
                run_id=run_id,  # Pass generated run_id for organizing outputs
                provider_type=provider_type,  # Pass provider type
                provider_config=provider_config,  # Pass provider config (for cloud providers)
                **vllm_params  # Unpack vLLM parameters (empty dict for cloud providers)
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