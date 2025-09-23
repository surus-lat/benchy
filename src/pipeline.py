"""Main Prefect pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from prefect import flow
from .inference.vllm_server import start_vllm_server, test_vllm_api, stop_vllm_server
from .tasks.lm_harness import run_spanish_evaluation, run_portuguese_evaluation, gather_results
import atexit
import os
import logging

logger = logging.getLogger(__name__)


@flow()
def benchmark_pipeline(
    model_name: str,
    tasks_spanish: str,
    tasks_portuguese: str,
    batch_size: str,
    output_path: str,
    wandb_args: str = "",
    log_samples: bool = True,
    limit: Optional[int] = None,
    lm_eval_spanish_venv: str = "/home/mauro/dev/lm-evaluation-harness",
    lm_eval_portuguese_venv: str = "/home/mauro/dev/portu",
    cache_requests: bool = True,
    trust_remote_code: bool = False,
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
    num_concurrent: int = 8,
    startup_timeout: int = 900,
    cuda_devices: Optional[str] = None,
    kv_cache_memory: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete vLLM-based benchmarking pipeline.
    
    The pipeline:
    1. Starts a vLLM server with the specified model
    2. Tests the API server to ensure it's working
    3. Runs Spanish language evaluation tasks
    4. Runs Portuguese language evaluation tasks  
    5. Uploads results
    6. Stops the vLLM server (guaranteed cleanup)
    
    Args:
        model_name: The model to evaluate
        tasks_spanish: Spanish tasks to run (e.g., "latam_es")
        tasks_portuguese: Portuguese tasks to run (e.g., "latam_pt")
        batch_size: Batch size configuration  
        output_path: Output path for results
        wandb_args: Weights & Biases arguments
        log_samples: Whether to log samples
        limit: Limit number of examples per task (useful for testing)
        lm_eval_spanish_venv: Path to Spanish lm-evaluation-harness installation
        lm_eval_portuguese_venv: Path to Portuguese lm-evaluation-harness installation
        upload_script_path: Path to upload script directory
        upload_script_name: Name of upload script
        cache_requests: Whether to enable request caching
        trust_remote_code: Whether to trust remote code when loading models
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
        num_concurrent: Number of concurrent API requests
        cuda_devices: CUDA devices to use (e.g., "3" or "2,3")
    """
    logger.info(f"Starting vLLM benchmark pipeline for model: {model_name}")
    
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
    
    output_path = f"{output_path}/{model_name.split('/')[-1]}"
    os.makedirs(output_path, exist_ok=True)
    
    # Step 3: Run Spanish evaluation
    logger.info("Running Spanish language evaluation...")
    spanish_results = run_spanish_evaluation(
        model_name=model_name,
        tasks=tasks_spanish,
        batch_size=batch_size,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        max_length=max_model_len,
        wandb_args=wandb_args,
        log_samples=log_samples,
        limit=limit,
        lm_eval_path=lm_eval_spanish_venv,
        cache_requests=cache_requests,
        trust_remote_code=trust_remote_code,
        num_concurrent=num_concurrent,
        cuda_devices=cuda_devices
    )

    
    # Step 4: Run Portuguese evaluation  
    logger.info("Running Portuguese language evaluation...")
    portuguese_results = run_portuguese_evaluation(
        model_name=model_name,
        tasks=tasks_portuguese,
        batch_size=batch_size,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        max_length=max_model_len,
        wandb_args=wandb_args,
        log_samples=log_samples,
        limit=limit,
        lm_eval_path=lm_eval_portuguese_venv,
        cache_requests=cache_requests,
        trust_remote_code=True,  # Always True for Portuguese
        num_concurrent=num_concurrent,
        tokenizer_backend="huggingface",  # Always huggingface for Portuguese
        cuda_devices=cuda_devices
    )
    
    # Step 4: gather results
    gather_result = gather_results(spanish_results=spanish_results, portuguese_results=portuguese_results)
    
    # Step 5: Stop vLLM server (cleanup) - depends on upload completion
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