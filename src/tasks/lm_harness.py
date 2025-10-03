"""LM evaluation harness functions for ML model benchmarking."""

import os
import subprocess
import logging
from typing import Dict, Any, Optional
from prefect import task
from ..generation_config import format_generation_params_for_lm_eval

logger = logging.getLogger(__name__)


@task
def run_spanish_evaluation(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Spanish task_group_name evaluation using lm-evaluation-harness via API.
    """
    return _run_evaluation(
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=task_config,
        limit=limit,
        task_group_name="Spanish",
        cuda_devices=cuda_devices
    )


@task
def run_portuguese_evaluation(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Portuguese task_group_name evaluation using lm-evaluation-harness via API.
    """
    return _run_evaluation(
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=task_config,
        limit=limit,
        task_group_name="Portuguese",
        cuda_devices=cuda_devices
    )
    
@task
def run_translation_evaluation(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Spanish task_group_name evaluation using lm-evaluation-harness via API.
    """
    return _run_evaluation(
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        task_config=task_config,
        limit=limit,
        task_group_name="Spanish",
        cuda_devices=cuda_devices
    )



def _run_evaluation(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    task_group_name: str = "Unknown",
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness using vLLM API server.
    
    Args:
        model_name: The model to evaluate
        output_path: Base output path for results
        server_info: Dictionary containing server info from start_vllm_server
        api_test_result: API test result (unused but kept for compatibility)
        task_config: Task configuration dictionary
        limit: Limit number of examples per task (useful for testing)
        task_group_name: Language name for logging
        cuda_devices: CUDA devices to use (e.g., "3" or "2,3")
        
    Returns:
        Dictionary with execution results and metadata
    """
    # Extract parameters from task config
    tasks = task_config['task_name']
    lm_eval_path = task_config['lm_eval_path']
    tokenizer_backend = task_config.get('tokenizer_backend', 'huggingface')
    use_chat_completions = task_config.get('use_chat_completions', False)  # Default to False
    generation_config = task_config.get('generation_config', None)
    
    # Get defaults from task config
    defaults = task_config.get('defaults', {})
    batch_size = defaults.get('batch_size', '4')
    log_samples = defaults.get('log_samples', True)
    cache_requests = defaults.get('cache_requests', True)
    trust_remote_code = defaults.get('trust_remote_code', False)
    num_concurrent = defaults.get('num_concurrent', 8)
    max_length = defaults.get('max_length', 2048)
    # bootstrap_iters is controlled via direct modifications to lm-eval defaults (now set to 1000)
    
    # Create task-specific output path
    output_subdir = task_config.get('output', {}).get('subdirectory', task_group_name.lower())
    task_output_path = f"{output_path}/{output_subdir}"
    
    server_url = server_info["url"]
    
    logger.info(f"Starting {task_group_name} evaluation for model: {model_name}, tasks: {tasks}")
    
    file_logger = logging.getLogger('benchy.lm_eval')
    try:
        file_logger.info(f"=== Starting {task_group_name} LM Evaluation ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Tasks: {tasks}")
        file_logger.info(f"Server URL: {server_url}")
        file_logger.info(f"Batch size: {batch_size}")
        file_logger.info(f"Concurrent requests: {num_concurrent}")
        if generation_config:
            file_logger.info(f"Using generation config: {generation_config}")
        if batch_size != "1":
            file_logger.info("Note: vLLM supports batched requests with varying sequence lengths")
        if limit:
            file_logger.info(f"Limit: {limit} (testing mode)")
    except (RuntimeError, OSError):
        pass
    
    # Build model_args string with optimizations for API-only usage
    # Determine the correct endpoint based on use_chat_completions
    endpoint = "/v1/chat/completions" if use_chat_completions else "/v1/completions"
    
    model_args_parts = [
        f"model={model_name}",
        f"max_length={max_length}",
        f"base_url={server_url}{endpoint}",
        f"num_concurrent={num_concurrent}",
        "max_retries=3",
        # Optimize for minimal local compute
        "tokenized_requests=False",  # Send text instead of tokens to API
    ]
    
    # Add trust_remote_code if True (needed for tokenizer)
    if trust_remote_code:
        model_args_parts.append("trust_remote_code=True")
    
    # Add tokenizer_backend if provided, defaulting to lightweight option
    if tokenizer_backend:
        model_args_parts.append(f"tokenizer_backend={tokenizer_backend}")
    else:
        # Use huggingface but with optimizations for CPU-only usage
        model_args_parts.append("tokenizer_backend=huggingface")
    
    # Add generation config parameters if available
    generation_params = format_generation_params_for_lm_eval(generation_config)
    if generation_params:
        model_args_parts.append(generation_params)
        logger.info(f"Added generation config parameters: {generation_params}")
    
    model_args_str = ",".join(model_args_parts)
    
    # Build the lm_eval command for API mode
    # Determine the correct model type based on use_chat_completions
    model_type = "local-chat-completions" if use_chat_completions else "local-completions"
    
    cmd_parts = [
        "lm_eval",
        "--model", model_type,
        "--model_args", model_args_str,
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--output_path", task_output_path
    ]
    
    # Add --apply_chat_template only if using chat completions
    if use_chat_completions:
        cmd_parts.append("--apply_chat_template")
        
    if log_samples:
        cmd_parts.append("--log_samples")
        
    if limit is not None:
        cmd_parts.extend(["--limit", str(limit)])
        
    if cache_requests:
        cmd_parts.extend(["--cache_requests", "true"]) # refresh
        
    if trust_remote_code:
        cmd_parts.append("--trust_remote_code")
    
    # Note: bootstrap_iters is not available as a command-line argument in lm-eval
    # We control this via our direct code modifications to metrics.py instead
    
    cmd = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd}")
    try:
        file_logger.info(f"Full command: {cmd}")
    except (RuntimeError, OSError):
        pass
    
    try:
        # Set environment variables for datasets trust and multiprocessing
        env_vars = "HF_DATASETS_TRUST_REMOTE_CODE=true DISABLE_MULTIPROC=0 OMP_NUM_THREADS=24"
        
        # Activate the lm-eval venv and run command
        venv_cmd = f"source {lm_eval_path}/.venv/bin/activate && {env_vars} {cmd}"
        
        # Set up CPU-only environment for lm-eval client
        env = os.environ.copy()
        # Force PyTorch to use CPU only
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTORCH_CUDA_ALLOC_CONF"] = ""  # Clear any CUDA memory settings
        
        # Force multiprocessing and CPU optimization
        env["DISABLE_MULTIPROC"] = "0"  # Ensure multiprocessing is enabled
        env["OMP_NUM_THREADS"] = "24"   # Use up to 24 threads
        env["MKL_NUM_THREADS"] = "24"   # For Intel MKL
        env["OPENBLAS_NUM_THREADS"] = "24"  # For OpenBLAS
        
        # Stream output in real-time
        process = subprocess.Popen(
            venv_cmd,
            shell=True,
            cwd=lm_eval_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            executable="/bin/bash",
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output while streaming it
        output_lines = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"[lm_eval] {line}")
                    # Safe file logging with error handling
                    try:
                        file_logger.info(f"[lm_eval] {line}")
                    except (RuntimeError, OSError):
                        # Continue if file logging fails
                        pass
                    output_lines.append(line)
        except Exception as stream_error:
            logger.warning(f"Stream reading interrupted: {stream_error}")
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            error_msg = f"lm_eval failed with return code {return_code}"
            logger.error(error_msg)
            try:
                file_logger.error(error_msg)
                file_logger.error("=== LM Evaluation FAILED ===")
            except (RuntimeError, OSError):
                pass
            raise RuntimeError(f"lm_eval execution failed with return code {return_code}")
            
        logger.info("lm_eval completed successfully")
        try:
            file_logger.info("=== LM Evaluation COMPLETED SUCCESSFULLY ===")
            file_logger.info(f"Output saved to: {task_output_path}")
        except (RuntimeError, OSError):
            pass
        
        return {
            "model_name": model_name,
            "tasks": tasks,
            "output_path": task_output_path,
            "stdout": stdout,
            "return_code": return_code,
            "command": cmd
        }
        
    except Exception as e:
        error_msg = f"Error running lm_eval: {str(e)}"
        logger.error(error_msg)
        try:
            file_logger.error(error_msg)
            file_logger.error("=== LM Evaluation FAILED ===")
        except (RuntimeError, OSError):
            pass
        raise


@task
def gather_results(spanish_results: Dict[str, Any], portuguese_results: Dict[str, Any], translation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gather results from Spanish and Portuguese evaluations.
    """
    return {
        "spanish_results": spanish_results,
        "portuguese_results": portuguese_results,
        "translation_results": translation_results
    }
