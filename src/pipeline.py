"""Main ZenML pipeline for vLLM-based ML model benchmarking."""

from datetime import datetime
from typing import Optional, Dict, Any
from zenml import pipeline
from zenml.logger import get_logger
from .steps import start_vllm_server, test_vllm_api, run_lm_evaluation, stop_vllm_server, upload_results

logger = get_logger(__name__)


def create_run_name(model_name: str, limit: Optional[int] = None) -> str:
    """
    Create a custom run name based on model name and timestamp.
    
    Args:
        model_name: Full model name (e.g., "google/gemma-3n-E4B-it")
        limit: Optional limit parameter - if set, adds TEST prefix
        
    Returns:
        Custom run name in format: [TEST_]model_name_YYYY_MM_DD_HH_MM_SS
    """
    # Extract just the model name part (after the slash if present)
    if "/" in model_name:
        clean_model_name = model_name.split("/")[-1]
    else:
        clean_model_name = model_name
    
    # Replace any remaining problematic characters
    clean_model_name = clean_model_name.replace("-", "_").replace(".", "_")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Add TEST prefix if limit is used
    prefix = "TEST_" if limit is not None else ""
    
    return f"{prefix}{clean_model_name}_{timestamp}"


@pipeline
def benchmark_pipeline(
    model_name: str,
    tasks_spanish: str,
    tasks_portuguese: str,
    batch_size: str,
    output_path: str,
    wandb_args: str = "",
    log_samples: bool = True,
    limit: Optional[int] = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    upload_script_path: str = "/home/mauro/dev/leaderboard",
    upload_script_name: str = "run_pipeline.py",
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
    hf_token: str = None,
    num_concurrent: int = 8
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
        lm_eval_path: Path to lm-evaluation-harness installation
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
    """
    logger.info(f"Starting vLLM benchmark pipeline for model: {model_name}")
    
    # Initialize variables for cleanup
    server_pid = None
    server_url = None
    server_port = None
    
    try:
        # Step 1: Start vLLM server
        server_pid, server_url, server_port = start_vllm_server(
            model_name=model_name,
            host=host,
            port=port,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            hf_cache=hf_cache,
            hf_token=hf_token
        )
        
        # Step 2: Test vLLM API
        api_test_result = test_vllm_api(
            server_url=server_url,
            model_name=model_name,
            port=server_port
        )
        
        # Step 3: Run Spanish evaluation
        logger.info("Running Spanish language evaluation...")
        spanish_results = run_lm_evaluation(
            model_name=model_name,
            tasks=tasks_spanish,
            batch_size=batch_size,
            output_path=f"{output_path}/spanish",
            server_url=server_url,
            port=server_port,
            wandb_args=wandb_args,
            log_samples=log_samples,
            limit=limit,
            lm_eval_path=lm_eval_path,
            cache_requests=cache_requests,
            trust_remote_code=trust_remote_code,
            num_concurrent=num_concurrent
        )
        
        # Step 4: Run Portuguese evaluation  
        logger.info("Running Portuguese language evaluation...")
        portuguese_results = run_lm_evaluation(
            model_name=model_name,
            tasks=tasks_portuguese,
            batch_size=batch_size,
            output_path=f"{output_path}/portuguese",
            server_url=server_url,
            port=server_port,
            wandb_args=wandb_args,
            log_samples=log_samples,
            limit=limit,
            lm_eval_path=lm_eval_path,
            cache_requests=cache_requests,
            trust_remote_code=trust_remote_code,
            num_concurrent=num_concurrent
        )
        
        # Combine results for upload
        combined_results = {
            "model_name": model_name,
            "spanish_results": spanish_results,
            "portuguese_results": portuguese_results,
            "api_test": api_test_result,
            "tasks": f"{tasks_spanish},{tasks_portuguese}",
            "output_path": output_path
        }
        
        # Step 5: Upload results
        upload_result = upload_results(
            eval_results=combined_results,
            script_path=upload_script_path,
            script_name=upload_script_name
        )
        
        logger.info("Benchmark pipeline completed successfully")
        return upload_result
        
    finally:
        # Step 6: Always stop vLLM server (guaranteed cleanup)
        if server_pid is not None:
            try:
                cleanup_result = stop_vllm_server(server_pid)
                if cleanup_result["status"] != "success":
                    logger.warning(f"⚠️ vLLM server cleanup had issues: {cleanup_result}")
                else:
                    logger.info("✅ vLLM server cleanup completed successfully")
            except Exception as cleanup_error:
                logger.error(f"❌ Critical: Failed to stop vLLM server (PID: {server_pid}): {cleanup_error}")
                logger.error("⚠️ WARNING: vLLM server may still be running. Please check and stop manually if needed.")
        else:
            logger.warning("⚠️ No vLLM server PID available for cleanup - server may not have started")