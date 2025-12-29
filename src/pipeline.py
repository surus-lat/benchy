"""Main Prefect pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from prefect import flow
from .inference.vllm_server import start_vllm_server, test_vllm_api, stop_vllm_server
from .tasks.portuguese import run_portuguese
from .tasks.spanish import run_spanish
from .tasks.structured import run_structured_extraction
from .tasks.image_extraction import run_image_extraction
from .tasks.translation import run_translation
from .config_manager import ConfigManager
from .generation_config import fetch_generation_config, save_generation_config
from .gpu_config import load_gpu_config
from .task_completion_checker import TaskCompletionChecker
from .run_id_manager import get_prefect_flow_name
import atexit
import os
import sys
import logging
import yaml
import json


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

logger = logging.getLogger(__name__)


SUMMARY_SKIP_KEYS = {
    "total_samples",
    "valid_samples",
    "error_count",
    "throughput",
    "total_duration",
    "samples_with_type_errors",
    "numeric_fields_total",
    "numeric_fields_correct",
    "imperfect_sample_count",
    "match_distribution_counts",
    "subtasks",
}

TASK_REGISTRY = {
    "spanish": {
        "run": run_spanish,
        "config_name": "spanish",
        "display_name": "Spanish language",
        "set_api_endpoint": True,
        "set_generation_config": True,
        "provider_types": ["openai", "anthropic", "surus", "together"],
    },
    "portuguese": {
        "run": run_portuguese,
        "config_name": "portuguese",
        "display_name": "Portuguese language",
        "set_api_endpoint": True,
        "set_generation_config": True,
        "provider_types": ["openai", "anthropic", "surus", "together"],
    },
    "translation": {
        "run": run_translation,
        "config_name": "translation",
        "display_name": "translation",
        "set_api_endpoint": False,
        "set_generation_config": False,
        "provider_types": ["openai", "anthropic", "together"],
    },
    "structured_extraction": {
        "run": run_structured_extraction,
        "config_name": "structured_extraction",
        "display_name": "structured data extraction",
        "set_api_endpoint": False,
        "set_generation_config": False,
        "provider_types": ["openai", "anthropic", "surus", "together"],
    },
    "image_extraction": {
        "run": run_image_extraction,
        "config_name": "image_extraction",
        "display_name": "image extraction",
        "set_api_endpoint": False,
        "set_generation_config": False,
        "provider_types": ["openai", "anthropic", "surus", "surus_ocr", "surus_factura", "together"],
    },
}


def _summarize_task_metrics(
    metrics: Dict[str, Any],
    metrics_manifest: Optional[list] = None,
) -> Dict[str, float]:
    """Filter aggregate metrics to numeric values excluding counts/metadata."""
    summarized: Dict[str, float] = {}
    if metrics_manifest:
        for key in metrics_manifest:
            value = metrics.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                summarized[key] = float(value)
        return summarized

    for key, value in metrics.items():
        if key in SUMMARY_SKIP_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            summarized[key] = float(value)
    return summarized


def _write_run_summary(
    model_output_path: str,
    model_name: str,
    run_id: Optional[str],
    tasks: list,
    task_results: Dict[str, Any],
    task_configs_by_name: Dict[str, Any],
) -> None:
    summary = {
        "model": model_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    for task_name in tasks:
        metrics = task_results.get(task_name, {}).get("metrics", {}) or {}
        task_config = task_configs_by_name.get(task_name, {})
        metrics_manifest = task_config.get("metrics_manifest") or None
        summary["tasks"][task_name] = (
            _summarize_task_metrics(metrics, metrics_manifest) if metrics else {}
        )

    summary_path = os.path.join(model_output_path, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote run summary to {summary_path}")


def write_run_config(
    model_name: str,
    run_id: str,
    output_path: str,
    tasks: list,
    limit: Optional[int],
    api_endpoint: str,
    task_defaults_overrides: Optional[Dict[str, Any]],
    vllm_config: Dict[str, Any],
    cuda_devices: Optional[str] = None,
    organization: Optional[str] = None,
    url: Optional[str] = None
) -> str:
    """
    Write complete run configuration to a YAML file.
    
    Args:
        model_name: The model being evaluated
        run_id: Run ID for this execution
        output_path: Base output path
        tasks: List of tasks to run
        limit: Limit for examples per task
        api_endpoint: API endpoint mode ("completions" or "chat")
        task_defaults_overrides: Task configuration overrides
        vllm_config: vLLM server configuration
        cuda_devices: CUDA devices used
        organization: Optional organization name
        url: Optional URL for the model/organization
        
    Returns:
        Path to the written config file
    """
    # Create the complete configuration dictionary
    model_config = {
        'name': model_name,
        'api_endpoint': api_endpoint
    }
    
    # Add organization and url if provided
    if organization is not None:
        model_config['organization'] = organization
    if url is not None:
        model_config['url'] = url
    
    run_config = {
        'run_metadata': {
            'run_id': run_id,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'output_structure': f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
        },
        'model': model_config,
        'tasks': tasks,
        'evaluation': {
            'limit': limit,
            'task_defaults_overrides': task_defaults_overrides or {}
        },
        'vllm_server': {
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
            'max_num_batched_tokens': vllm_config.get('max_num_batched_tokens', None)
        },
        'environment': {
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'prefect_api_url': os.environ.get('PREFECT_API_URL', 'http://localhost:4200/api')
        }
    }
    
    # Create model output directory
    model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)
    
    # Write config file
    config_file_path = f"{model_output_path}/run_config.yaml"
    with open(config_file_path, 'w') as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    logger.info(f"Run configuration written to: {config_file_path}")
    return config_file_path


@flow()
def benchmark_pipeline(
    model_name: str,
    tasks: list,
    output_path: str,
    limit: Optional[int] = None,
    api_endpoint: str = "completions",
    task_defaults_overrides: Optional[Dict[str, Any]] = None,
    log_setup: Optional[Any] = None,
    run_id: Optional[str] = None,
    provider_type: str = "vllm",
    provider_config: Optional[Dict[str, Any]] = None,
    organization: Optional[str] = None,
    url: Optional[str] = None,
    # vLLM server configuration (only used if provider_type == 'vllm')
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.6,
    enforce_eager: bool = True,
    limit_mm_per_prompt: str = '{"images": 0, "audios": 0}',
    hf_cache: str = "/home/mauro/.cache/huggingface",
    hf_token: str = "",
    startup_timeout: int = 900,
    cuda_devices: Optional[str] = None,
    kv_cache_memory: Optional[int] = None,
    vllm_venv_path: str = "/home/mauro/dev/benchy/.venv",
    vllm_version: Optional[str] = None,
    multimodal: bool = True,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    # New parameters for better model compatibility
    trust_remote_code: bool = True,
    tokenizer_mode: Optional[str] = None,
    config_format: Optional[str] = None,
    load_format: Optional[str] = None,
    tool_call_parser: Optional[str] = None,
    enable_auto_tool_choice: bool = False,
    kv_cache_dtype: Optional[str] = None,
    kv_offloading_size: Optional[int] = None,
    skip_mm_profiling: bool = False
) -> Dict[str, Any]:
    """
    Complete benchmarking pipeline for vLLM and cloud providers.
    
    The pipeline:
    - For vLLM: Starts server, tests API, runs tasks, stops server
    - For cloud providers (OpenAI/Anthropic): Directly runs tasks using API
    
    Args:
        model_name: The model to evaluate
        tasks: List of task names to run (e.g., ["spanish", "portuguese"])
        output_path: Base output path for results
        limit: Limit number of examples per task (useful for testing)
        api_endpoint: API endpoint mode ("completions" or "chat")
        task_defaults_overrides: Optional dict to override task default parameters
        log_setup: Logging setup object
        run_id: Optional run ID for organizing outputs
        provider_type: Provider type ('vllm', 'openai', or 'anthropic')
        provider_config: Provider configuration (for cloud providers)
        # vLLM server configuration (only used if provider_type == 'vllm')
        host: Host to bind vLLM server to
        port: Port for vLLM server
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model length
        gpu_memory_utilization: GPU memory utilization
        enforce_eager: Whether to enforce eager execution
        limit_mm_per_prompt: Multimodal limits as JSON string
        hf_cache: Hugging Face cache directory
        hf_token: Hugging Face token
        startup_timeout: Server startup timeout
        cuda_devices: CUDA devices to use (e.g., "3" or "2,3")
        kv_cache_memory: KV cache memory allocation
    """
    logger.info(f"Starting benchmark pipeline for model: {model_name}")
    logger.info(f"Provider type: {provider_type}")
    logger.info(f"Tasks to run: {tasks}")
    logger.info(f"Using run_id: {run_id}")
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load GPU configuration from central config
    central_config = load_config('configs/config.yaml')
    gpu_manager = load_gpu_config(central_config)
    
    # Log GPU configuration
    gpu_summary = gpu_manager.get_config_summary()
    logger.info(f"GPU Configuration: {gpu_summary}")
    
    # Expand task groups into individual tasks
    expanded_tasks = config_manager.expand_task_groups(tasks, central_config)
    logger.info(f"Task expansion: {tasks} -> {expanded_tasks}")
    
    # Update tasks with expanded list
    tasks = expanded_tasks
    
    # Check for completed tasks to enable resuming failed runs
    completion_checker = TaskCompletionChecker(
        output_path=output_path,
        run_id=run_id,
        model_name=model_name
    )
    
    # Check completion status for all requested tasks
    completion_status = completion_checker.get_completed_tasks(tasks)
    completion_checker.log_completion_summary(completion_status)
    
    # Filter out completed tasks
    pending_tasks = [task for task, completed in completion_status.items() if not completed]
    
    if not pending_tasks:
        logger.info("All tasks are already completed! Nothing to run.")
        # Return early with a success result
        return {
            "model_name": model_name,
            "run_id": run_id,
            "status": "all_tasks_completed",
            "completed_tasks": tasks,
            "message": "All tasks were already completed in previous run"
        }
    
    logger.info(f"Running {len(pending_tasks)} pending tasks: {pending_tasks}")
    
    # Step 0: Fetch generation config from model repository (only for vLLM)
    generation_config = None
    if provider_type == 'vllm':
        generation_config = fetch_generation_config(
            model_name=model_name,
            hf_cache=hf_cache,
            hf_token=hf_token
        )
    
    # Write complete run configuration
    config_file_path = write_run_config(
        model_name=model_name,
        run_id=run_id,
        output_path=output_path,
        tasks=tasks,
        limit=limit,
        api_endpoint=api_endpoint,
        task_defaults_overrides=task_defaults_overrides,
        vllm_config={
            'host': host,
            'port': port,
            'tensor_parallel_size': tensor_parallel_size,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'enforce_eager': enforce_eager,
            'limit_mm_per_prompt': limit_mm_per_prompt,
            'hf_cache': hf_cache,
            'hf_token': hf_token,
            'startup_timeout': startup_timeout,
            'kv_cache_memory': kv_cache_memory,
            'vllm_venv_path': vllm_venv_path,
            'vllm_version': vllm_version,
            'multimodal': multimodal,
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens
        },
        cuda_devices=cuda_devices,
        organization=organization,
        url=url
    )
    
    # Initialize server_info and api_test_result
    server_info = None
    api_test_result = {"status": "skipped"}
    
    # Step 1 & 2: Start and test server (only for vLLM)
    if provider_type == 'vllm':
        # Use GPU configuration from central config, with command line override
        vllm_cuda_devices = cuda_devices if cuda_devices is not None else gpu_manager.get_vllm_cuda_devices()
        logger.info(f"Starting vLLM server with CUDA devices: {vllm_cuda_devices}")
        
        server_info = start_vllm_server(
            model_name=model_name,
            host=host,
            port=port,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            hf_cache=hf_cache,
            hf_token=hf_token,
            vllm_venv_path=vllm_venv_path,
            startup_timeout=startup_timeout,
            cuda_devices=vllm_cuda_devices,
            kv_cache_memory=kv_cache_memory,
            vllm_version=vllm_version,
            multimodal=multimodal,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            trust_remote_code=trust_remote_code,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            load_format=load_format,
            tool_call_parser=tool_call_parser,
            enable_auto_tool_choice=enable_auto_tool_choice,
            kv_cache_dtype=kv_cache_dtype,
            kv_offloading_size=kv_offloading_size,
            skip_mm_profiling=skip_mm_profiling
        )
        
        # Store server info globally for signal handler
        import eval
        eval._active_server_info = server_info
            
        # Step 2: Test vLLM API
        api_test_result = test_vllm_api(
            server_info=server_info,
            model_name=model_name
        )
    else:
        logger.info(f"Using cloud provider {provider_type}, skipping vLLM server startup")
    
    # Create run-specific output directory structure: {output_path}/{run_id}/{model_name}
    model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save generation config to output directory
    if generation_config:
        save_generation_config(generation_config, model_output_path, model_name)
    
    # Step 3: Run evaluation tasks (only pending ones)
    task_results = {}
    task_configs_by_name: Dict[str, Any] = {}
    
    # Load results from already completed tasks
    completed_tasks = [task for task, completed in completion_status.items() if completed]
    for task in completed_tasks:
        logger.info(f"Loading results from previously completed task: {task}")
        # Create a placeholder result for completed tasks
        task_results[task] = {
            "model_name": model_name,
            "task": task,
            "status": "previously_completed",
            "output_path": f"{model_output_path}/{task}",
            "message": f"Task {task} was completed in a previous run"
        }
    
    for task_name in pending_tasks:
        task_spec = TASK_REGISTRY.get(task_name)
        if not task_spec:
            logger.warning(f"Unknown task '{task_name}', skipping")
            continue

        display_name = task_spec.get("display_name", task_name)
        logger.info(f"Running {display_name} evaluation...")

        config_name = task_spec.get("config_name", task_name)
        task_config = config_manager.get_task_config(config_name, task_defaults_overrides)
        task_configs_by_name[task_name] = task_config

        if task_spec.get("set_api_endpoint"):
            task_config["api_endpoint"] = api_endpoint
        if task_spec.get("set_generation_config"):
            task_config["generation_config"] = generation_config

        if log_setup:
            log_setup.log_task_config(task_name, task_config)

        cloud_provider_config = None
        provider_types = task_spec.get("provider_types", [])
        if provider_type in provider_types and provider_config:
            cloud_provider_config = {
                **provider_config,
                "provider_type": provider_type,
            }

        run_fn = task_spec["run"]
        task_results[task_name] = run_fn(
            model_name=model_name,
            output_path=model_output_path,
            server_info=server_info,
            api_test_result=api_test_result,
            task_config=task_config,
            limit=limit,
            cuda_devices=gpu_manager.get_task_cuda_devices() if provider_type == 'vllm' else None,
            provider_config=cloud_provider_config,
        )
            
    # Step 4: Gather results
    gather_result = {
        f"{task_name}_results": result for task_name, result in task_results.items()
    }
    
    # Step 5: Stop vLLM server (cleanup) - only for vLLM
    if provider_type == 'vllm' and server_info:
        cleanup_result = stop_vllm_server(server_info=server_info, upload_result=gather_result)
        
        # Clear global server info
        import eval
        eval._active_server_info = None
    else:
        # For cloud providers, just return the gather result
        cleanup_result = gather_result
    
    # Log final completion summary
    newly_completed = [task for task in pending_tasks if task in task_results]
    total_completed = len(completed_tasks) + len(newly_completed)
    
    logger.info("=" * 60)
    logger.info("BENCHMARK PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tasks requested: {len(tasks)}")
    logger.info(f"Previously completed: {len(completed_tasks)}")
    logger.info(f"Newly completed: {len(newly_completed)}")
    logger.info(f"Total completed: {total_completed}")
    logger.info("=" * 60)

    _write_run_summary(
        model_output_path=model_output_path,
        model_name=model_name,
        run_id=run_id,
        tasks=tasks,
        task_results=task_results,
        task_configs_by_name=task_configs_by_name,
    )
    
    logger.info("Benchmark pipeline completed successfully")
    return cleanup_result

@flow()
def test_vllm_server(
    model_name: str,
    run_id: Optional[str] = None,
    # vLLM server configuration
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.6,
    enforce_eager: bool = True,
    limit_mm_per_prompt: str = '{"images": 0, "audios": 0}',
    hf_cache: str = "/home/mauro/.cache/huggingface",
    hf_token: str = "",
    startup_timeout: int = 900,
    cuda_devices: Optional[str] = None,
    kv_cache_memory: Optional[int] = None,
    vllm_venv_path: str = "/home/mauro/dev/benchy/.venv",
    vllm_version: Optional[str] = None,
    multimodal: bool = True,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    # New parameters for better model compatibility
    trust_remote_code: bool = True,
    tokenizer_mode: Optional[str] = None,
    config_format: Optional[str] = None,
    load_format: Optional[str] = None,
    tool_call_parser: Optional[str] = None,
    enable_auto_tool_choice: bool = False,
    kv_cache_dtype: Optional[str] = None,
    kv_offloading_size: Optional[int] = None,
    skip_mm_profiling: bool = False
) -> Dict[str, Any]:
    """
    Test vLLM server functionality without running full evaluation.
    
    Args:
        model_name: The model to test
        run_id: Optional run ID for organizing outputs. If not provided, auto-generated.
        # vLLM server configuration parameters...
    """
    logger.info(f"Using run_id for test: {run_id}")
    
    # Write test run configuration (simplified version)
    test_config = {
        'run_metadata': {
            'run_id': run_id,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'run_type': 'test'
        },
        'model': {
            'name': model_name
        },
        'vllm_server': {
            'host': host,
            'port': port,
            'tensor_parallel_size': tensor_parallel_size,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'enforce_eager': enforce_eager,
            'limit_mm_per_prompt': limit_mm_per_prompt,
            'hf_cache': hf_cache,
            'hf_token': hf_token,
            'startup_timeout': startup_timeout,
            'cuda_devices': cuda_devices,
            'kv_cache_memory': kv_cache_memory,
            'vllm_venv_path': vllm_venv_path,
            'vllm_version': vllm_version,
            'multimodal': multimodal,
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens
        },
        'environment': {
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'prefect_api_url': os.environ.get('PREFECT_API_URL', 'http://localhost:4200/api')
        }
    }
    
    # Create test output directory and write config
    test_output_path = f"/tmp/benchy_test_outputs/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(test_output_path, exist_ok=True)
    test_config_path = f"{test_output_path}/test_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, sort_keys=False, indent=2)
    logger.info(f"Test configuration written to: {test_config_path}")
    
    # Load GPU configuration from central config
    central_config = load_config('configs/config.yaml')
    gpu_manager = load_gpu_config(central_config)
    
    # Log GPU configuration
    gpu_summary = gpu_manager.get_config_summary()
    logger.info(f"GPU Configuration for test: {gpu_summary}")
        
    # Step 1: Start vLLM server
    # Use GPU configuration from central config, with command line override
    vllm_cuda_devices = cuda_devices if cuda_devices is not None else gpu_manager.get_vllm_cuda_devices()
    logger.info(f"Starting vLLM test server with CUDA devices: {vllm_cuda_devices}")
    
    server_info = start_vllm_server(
        model_name=model_name,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt=limit_mm_per_prompt,
        hf_cache=hf_cache,
        hf_token=hf_token,
        vllm_venv_path=vllm_venv_path,
        startup_timeout=startup_timeout,
        cuda_devices=vllm_cuda_devices,
        kv_cache_memory=kv_cache_memory,
        vllm_version=vllm_version,
        multimodal=multimodal,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=trust_remote_code,
        tokenizer_mode=tokenizer_mode,
        config_format=config_format,
        load_format=load_format,
        tool_call_parser=tool_call_parser,
        enable_auto_tool_choice=enable_auto_tool_choice,
        kv_cache_dtype=kv_cache_dtype,
        kv_offloading_size=kv_offloading_size,
        skip_mm_profiling=skip_mm_profiling
    )
    
    # Store server info globally for signal handler
    import eval
    eval._active_server_info = server_info
        
    # Step 2: Test vLLM API
    api_test_result = test_vllm_api(
        server_info=server_info,
        model_name=model_name
    )
    
    # Step 3: Stop vLLM server (cleanup) - depends on upload completion
    cleanup_result = stop_vllm_server(server_info=server_info, upload_result=api_test_result)
    
    # Clear global server info
    import eval
    eval._active_server_info = None
    
    logger.info("Test model config completed successfully")
    return cleanup_result
