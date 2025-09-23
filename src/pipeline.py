"""Main Prefect pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from prefect import flow
from .inference.vllm_server import start_vllm_server, test_vllm_api, stop_vllm_server
from .tasks.lm_harness import run_spanish_evaluation, run_portuguese_evaluation, gather_results
from .config_manager import ConfigManager
import atexit
import os
import logging

logger = logging.getLogger(__name__)


@flow()
def benchmark_pipeline(
    model_name: str,
    tasks: list,
    output_path: str,
    limit: Optional[int] = None,
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
    kv_cache_memory: Optional[int] = None
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
        wandb_args: Weights & Biases arguments
        limit: Limit number of examples per task (useful for testing)
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
    
    # Initialize config manager
    config_manager = ConfigManager()
    
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
        vllm_venv_path="/home/mauro/dev/benchy/.venv",
        startup_timeout=startup_timeout,
        cuda_devices=cuda_devices,
        kv_cache_memory=kv_cache_memory
    )
        
    # Step 2: Test vLLM API
    api_test_result = test_vllm_api(
        server_info=server_info,
        model_name=model_name
    )
    
    # Create model-specific output directory
    model_output_path = f"{output_path}/{model_name.split('/')[-1]}"
    os.makedirs(model_output_path, exist_ok=True)
    
    # Step 3: Run evaluation tasks
    task_results = {}
    
    if "spanish" in tasks:
        logger.info("Running Spanish language evaluation...")
        spanish_task_config = config_manager.get_task_config("spanish")
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
        portuguese_task_config = config_manager.get_task_config("portuguese")
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
    
    # Step 4: Gather results
    gather_result = gather_results(
        spanish_results=task_results.get("spanish", {}),
        portuguese_results=task_results.get("portuguese", {})
    )
    
    # Step 5: Stop vLLM server (cleanup)
    cleanup_result = stop_vllm_server(server_info=server_info, upload_result=gather_result)
    
    logger.info("Benchmark pipeline completed successfully")
    return cleanup_result

@flow()
def test_vllm_server(
    model_name: str,
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
    kv_cache_memory: Optional[int] = None
) -> Dict[str, Any]:
        
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
        vllm_venv_path="/home/mauro/dev/benchy/.venv",
        startup_timeout=startup_timeout,
        cuda_devices=cuda_devices,
        kv_cache_memory=kv_cache_memory
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