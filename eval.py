"""Main entry point for benchy - vLLM-powered ML benchmarking."""

import os
import sys
import argparse
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

import logging
import signal

from src.config_loader import load_config
from src.config_manager import ConfigManager
from src.gpu_config import load_gpu_config
from src.logging_utils import setup_file_logging
from src.run_id_manager import (
    generate_run_id,
    get_prefect_flow_name,
    get_run_paths,
    setup_run_directories,
)
from src.inference.vllm_config import VLLMServerConfig


logger = logging.getLogger(__name__)

# Global state for signal handling
_active_server_info = None

PROVIDER_SPECS = {
    "vllm": {"config_key": "vllm"},
    "openai": {
        "config_key": "openai",
        "log": "Using OpenAI cloud provider for model: {model_name}",
    },
    "anthropic": {
        "config_key": "anthropic",
        "log": "Using Anthropic cloud provider for model: {model_name}",
    },
    "surus": {
        "config_key": "surus",
        "log": "Using SURUS AI provider for extraction tasks",
    },
    "surus_ocr": {
        "config_key": "surus_ocr",
        "log": "Using SURUS AI OCR provider for image extraction tasks",
    },
    "surus_factura": {
        "config_key": "surus_factura",
        "log": "Using SURUS AI Factura provider for image extraction tasks",
    },
    "together": {
        "config_key": "together",
        "log": "Using Together AI cloud provider for model: {model_name}",
    },
}

def signal_handler(signum, frame):
    """Handle termination signals by cleaning up processes."""
    logger.info(f"Received signal {signum}. Cleaning up...")
    
    # Import here to avoid circular dependency
    from src.inference.vllm_server import stop_vllm_server
    
    # Stop vLLM server
    if _active_server_info is not None:
        try:
            logger.info("Stopping vLLM server...")
            stop_vllm_server(_active_server_info, {})
        except Exception as e:
            logger.warning(f"Error stopping vLLM server: {e}")
    
    logger.info("Cleanup complete. Exiting.")
    sys.exit(130)  # Standard exit code for SIGINT


def resolve_provider_config(
    config: dict,
    provider_type: str,
    model_name: str,
    gpu_manager,
):
    """Resolve provider config and vLLM server config for a provider type."""
    provider_spec = PROVIDER_SPECS.get(provider_type)
    if not provider_spec:
        raise ValueError(f"Unknown provider type: {provider_type}")

    provider_config = config.get(provider_spec["config_key"], {})
    log_message = provider_spec.get("log")
    if log_message:
        logger.info(log_message.format(model_name=model_name))

    vllm_server_config = None
    if provider_type == "vllm":
        cuda_devices = provider_config.get("cuda_devices", gpu_manager.get_vllm_cuda_devices())
        vllm_server_config = VLLMServerConfig.from_config(
            provider_config,
            cuda_devices=cuda_devices,
        )

    return provider_config, vllm_server_config


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

    parser.add_argument(
        '--tasks',
        type=str,
        default=None,
        help='Comma-separated list of tasks or task groups (overrides config tasks)'
    )

    parser.add_argument(
        '--tasks-file',
        type=str,
        default=None,
        help='Path to a task list file (one task per line, overrides config tasks)'
    )

    parser.add_argument(
        '--task-group',
        action='append',
        default=None,
        help='Task group name(s) from configs/config.yaml (can be repeated)'
    )
    
    return parser.parse_args()


def _parse_tasks_arg(value: Optional[str]) -> list:
    if not value:
        return []
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _load_tasks_file(path: str) -> list:
    tasks = []
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tasks.append(line)
    return tasks


def _dedupe_tasks(tasks: list) -> list:
    seen = set()
    ordered = []
    for task_name in tasks:
        if task_name in seen:
            continue
        seen.add(task_name)
        ordered.append(task_name)
    return ordered


def main():
    """Run the vLLM benchmark pipeline with configuration from specified file."""
    # Parse command line arguments
    args = parse_args()
    
    # Validate mutually exclusive log_samples arguments
    if args.log_samples and args.no_log_samples:
        logger.error("Cannot specify both --log-samples and --no-log-samples")
        return

    prefect_enabled = args.register
    if not prefect_enabled:
        prefect_enabled = os.environ.get("BENCHY_ENABLE_PREFECT", "").lower() in (
            "1",
            "true",
            "yes",
        )

    if prefect_enabled:
        os.environ.pop("BENCHY_DISABLE_PREFECT", None)
        if args.prefect_url:
            os.environ["PREFECT_API_URL"] = args.prefect_url
        elif "PREFECT_API_URL" not in os.environ:
            os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    else:
        os.environ.setdefault("BENCHY_DISABLE_PREFECT", "1")

    # Prefect-dependent imports must happen after BENCHY_DISABLE_PREFECT is set.
    from src.prefect_compat import PREFECT_AVAILABLE, serve
    from src.pipeline import benchmark_pipeline, test_vllm_server
    
    logger.info("Starting benchy - vLLM-powered ML model benchmarking")
    
    # Register signal handlers for clean termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.verbose:
        logger.info(f"Command line arguments: {args}")
        logger.info(f"Python path: {sys.executable}")
        logger.info(f"Working directory: {os.getcwd()}")
    
    if prefect_enabled:
        logger.info(
            "Prefect enabled; using API URL: "
            f"{os.environ.get('PREFECT_API_URL', 'http://localhost:4200/api')}"
        )
    else:
        logger.info("Prefect disabled; running pipeline without orchestration")
    
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
    model_config = config.get('model', {})
    model_name = model_config['name']
    organization = model_config.get('organization')
    url = model_config.get('url')
    provider_type = config.get('provider_type', 'vllm')
    
    provider_config, vllm_server_config = resolve_provider_config(
        config=config,
        provider_type=provider_type,
        model_name=model_name,
        gpu_manager=gpu_manager,
    )

    api_endpoint = provider_config.get('api_endpoint', config.get('api_endpoint', "completions"))
    
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

    config_tasks = config.get('tasks', ['spanish', 'portuguese'])
    tasks_override = []
    if args.tasks:
        tasks_override.extend(_parse_tasks_arg(args.tasks))
    if args.tasks_file:
        tasks_override.extend(_load_tasks_file(args.tasks_file))
    if args.task_group:
        for group_entry in args.task_group:
            tasks_override.extend(_parse_tasks_arg(group_entry))

    tasks_override = _dedupe_tasks(tasks_override)

    model_provider_types = {"vllm", "openai", "anthropic", "together"}
    is_system_provider = provider_type not in model_provider_types

    if tasks_override:
        logger.info(f"Using task overrides from CLI: {tasks_override}")
        if is_system_provider and config_tasks:
            # Check if task or its parent group is in config_tasks
            # E.g., 'structured_extraction.email_extract' should be allowed if 'structured_extraction' is in config
            allowed = []
            disallowed = []
            for task in tasks_override:
                # Check if task itself is in config
                if task in config_tasks:
                    allowed.append(task)
                # Check if parent group is in config (e.g., 'structured_extraction' for 'structured_extraction.email_extract')
                elif '.' in task and task.split('.')[0] in config_tasks:
                    allowed.append(task)
                else:
                    disallowed.append(task)
            
            if disallowed:
                logger.warning(
                    f"Ignoring tasks not declared in system config: {disallowed}"
                )
            tasks_to_run = allowed
        else:
            tasks_to_run = tasks_override
    else:
        tasks_to_run = config_tasks

    if not tasks_to_run:
        logger.error("No tasks to run after applying task overrides.")
        return

    # Expand task groups into individual tasks
    expanded_tasks = config_manager.expand_task_groups(tasks_to_run, central_config)
    logger.info(f"Task expansion: {tasks_to_run} -> {expanded_tasks}")
    tasks_to_run = expanded_tasks

    logger.info(f"Tasks to run: {tasks_to_run}")
    if provider_type == 'vllm' and vllm_server_config:
        logger.info(f"vLLM server: {vllm_server_config.host}:{vllm_server_config.port}")
    elif provider_type in ['openai', 'anthropic', 'together']:
        logger.info(f"Cloud provider: {provider_type}")
        logger.info(f"Base URL: {provider_config.get('base_url', 'N/A')}")
    elif provider_type == 'surus':
        logger.info(f"SURUS AI system")
        logger.info(f"Endpoint: {provider_config.get('endpoint', 'N/A')}")
    logger.info(f"Output path: {run_paths['output_path']}")
    logger.info(f"Log path: {run_paths['log_path']}")
    
    # Handle flow registration if requested
    if args.register:
        if not PREFECT_AVAILABLE:
            logger.error(
                "Prefect is disabled or not installed; install with `pip install .[prefect]` "
                "and unset BENCHY_DISABLE_PREFECT to register flows."
            )
            return
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
            if vllm_server_config is None:
                vllm_server_config = VLLMServerConfig()
            result = test_vllm_server(
                model_name=model_name,
                run_id=run_id,
                vllm_config=vllm_server_config,
            )
        else:
            # Run full benchmark pipeline
            result = benchmark_pipeline(
                model_name=model_name,
                tasks=tasks_to_run,  # Use expanded task list
                output_path=run_paths['output_path'],  # Use run-specific output path
                limit=args.limit,  # Use command line limit parameter
                api_endpoint=api_endpoint,
                task_defaults_overrides=task_defaults_overrides or None,  # Pass task overrides
                log_setup=log_setup,  # Pass log setup for task config logging
                run_id=run_id,  # Pass generated run_id for organizing outputs
                provider_type=provider_type,  # Pass provider type
                provider_config=provider_config,  # Pass provider config (for cloud providers)
                organization=organization,  # Pass organization if present in config
                url=url,  # Pass url if present in config
                vllm_config=vllm_server_config,  # vLLM server config (ignored for cloud providers)
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
