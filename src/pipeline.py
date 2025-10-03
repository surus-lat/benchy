"""Main Prefect pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from prefect import flow
from .inference.vllm_server import start_vllm_server, test_vllm_api, stop_vllm_server
from .tasks.lm_harness import run_spanish_evaluation, run_portuguese_evaluation, gather_results, run_translation_evaluation
from .config_manager import ConfigManager
from .generation_config import fetch_generation_config, save_generation_config
import atexit
import os
import sys
import logging
import yaml
import json

logger = logging.getLogger(__name__)


def write_run_config(
    model_name: str,
    run_id: str,
    output_path: str,
    tasks: list,
    limit: Optional[int],
    use_chat_completions: bool,
    task_defaults_overrides: Optional[Dict[str, Any]],
    vllm_config: Dict[str, Any],
    cuda_devices: Optional[str] = None
) -> str:
    """
    Write complete run configuration to a YAML file.
    
    Args:
        model_name: The model being evaluated
        run_id: Run ID for this execution
        output_path: Base output path
        tasks: List of tasks to run
        limit: Limit for examples per task
        use_chat_completions: Whether to use chat completions API
        task_defaults_overrides: Task configuration overrides
        vllm_config: vLLM server configuration
        cuda_devices: CUDA devices used
        
    Returns:
        Path to the written config file
    """
    # Create the complete configuration dictionary
    run_config = {
        'run_metadata': {
            'run_id': run_id,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'output_structure': f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
        },
        'model': {
            'name': model_name,
            'use_chat_completions': use_chat_completions
        },
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
    use_chat_completions: bool = False,
    task_defaults_overrides: Optional[Dict[str, Any]] = None,
    log_setup: Optional[Any] = None,
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
    max_num_batched_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete vLLM-based benchmarking pipeline.
    
    The pipeline:
    1. Starts a vLLM server with the specified model
    2. Tests the API server to ensure it's working
    3. Runs specified evaluation tasks (spanish, portuguese, etc.)
    4. Gathers results
    5. Stops the vLLM server (guaranteed cleanup)
    
    Args:
        model_name: The model to evaluate
        tasks: List of task names to run (e.g., ["spanish", "portuguese"])
        output_path: Base output path for results
        limit: Limit number of examples per task (useful for testing)
        use_chat_completions: Whether to use chat completions API (/v1/chat/completions) or completions API (/v1/completions)
        task_defaults_overrides: Optional dict to override task default parameters (e.g., log_samples, batch_size)
        log_setup: Logging setup object
        run_id: Optional run ID for organizing outputs. If not provided, auto-generated.
        # vLLM server configuration
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
    logger.info(f"Starting vLLM benchmark pipeline for model: {model_name}")
    logger.info(f"Tasks to run: {tasks}")
    
    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Auto-generated run_id: {run_id}")
    else:
        logger.info(f"Using provided run_id: {run_id}")
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Step 0: Fetch generation config from model repository
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
        use_chat_completions=use_chat_completions,
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
        cuda_devices=cuda_devices
    )
    
    # Step 1: Start vLLM server
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
        cuda_devices=cuda_devices,
        kv_cache_memory=kv_cache_memory,
        vllm_version=vllm_version,
        multimodal=multimodal,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens
    )
        
    # Step 2: Test vLLM API
    api_test_result = test_vllm_api(
        server_info=server_info,
        model_name=model_name
    )
    
    # Create run-specific output directory structure: {output_path}/{run_id}/{model_name}
    model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save generation config to output directory
    if generation_config:
        save_generation_config(generation_config, model_output_path, model_name)
    
    # Step 3: Run evaluation tasks
    task_results = {}
    
    if "spanish" in tasks:
        logger.info("Running Spanish language evaluation...")
        spanish_task_config = config_manager.get_task_config("spanish", task_defaults_overrides)
        # Merge use_chat_completions from model config into task config
        spanish_task_config['use_chat_completions'] = use_chat_completions
        # Add generation config
        spanish_task_config['generation_config'] = generation_config
        
        # Log task configuration
        if log_setup:
            log_setup.log_task_config("spanish", spanish_task_config)
        spanish_results = run_spanish_evaluation(
            model_name=model_name,
            output_path=model_output_path,
            server_info=server_info,
            api_test_result=api_test_result,
            task_config=spanish_task_config,
            limit=limit,
            cuda_devices=cuda_devices
        )
        task_results["spanish"] = spanish_results
    
    if "portuguese" in tasks:
        logger.info("Running Portuguese language evaluation...")
        portuguese_task_config = config_manager.get_task_config("portuguese", task_defaults_overrides)
        # Merge use_chat_completions from model config into task config
        portuguese_task_config['use_chat_completions'] = use_chat_completions
        # Add generation config
        portuguese_task_config['generation_config'] = generation_config
        
        # Log task configuration
        if log_setup:
            log_setup.log_task_config("portuguese", portuguese_task_config)
        portuguese_results = run_portuguese_evaluation(
            model_name=model_name,
            output_path=model_output_path,
            server_info=server_info,
            api_test_result=api_test_result,
            task_config=portuguese_task_config,
            limit=limit,
            cuda_devices=cuda_devices
        )
        task_results["portuguese"] = portuguese_results
    
    if "translation" in tasks:
        logger.info("Running translation language evaluation...")
        translation_task_config = config_manager.get_task_config("translation", task_defaults_overrides)
        # Merge use_chat_completions from model config into task config
        translation_task_config['use_chat_completions'] = use_chat_completions
        # Add generation config
        translation_task_config['generation_config'] = generation_config
        
        # Log task configuration
        if log_setup:
            log_setup.log_task_config("translation", translation_task_config)
        translation_results = run_translation_evaluation(
            model_name=model_name,
            output_path=model_output_path,
            server_info=server_info,
            api_test_result=api_test_result,
            task_config=translation_task_config,
            limit=limit,
            cuda_devices=cuda_devices
        )
        task_results["translation"] = translation_results
            
    # Step 4: Gather results
    gather_result = gather_results(
        spanish_results=task_results.get("spanish", {}),
        portuguese_results=task_results.get("portuguese", {}),
        translation_results=task_results.get("translation", {})
    )
    
    # Step 5: Stop vLLM server (cleanup)
    cleanup_result = stop_vllm_server(server_info=server_info, upload_result=gather_result)
    
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
    max_num_batched_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test vLLM server functionality without running full evaluation.
    
    Args:
        model_name: The model to test
        run_id: Optional run ID for organizing outputs. If not provided, auto-generated.
        # vLLM server configuration parameters...
    """
    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Auto-generated run_id for test: {run_id}")
    else:
        logger.info(f"Using provided run_id for test: {run_id}")
    
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
        
    # Step 1: Start vLLM server
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
        cuda_devices=cuda_devices,
        kv_cache_memory=kv_cache_memory,
        vllm_version=vllm_version,
        multimodal=multimodal,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens
    )
        
    # Step 2: Test vLLM API
    api_test_result = test_vllm_api(
        server_info=server_info,
        model_name=model_name
    )
    
    # Step 3: Stop vLLM server (cleanup) - depends on upload completion
    cleanup_result = stop_vllm_server(server_info=server_info, upload_result=api_test_result)
    
    logger.info("Test model config completed successfully")
    return cleanup_result