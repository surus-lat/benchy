"""LM evaluation harness functions for ML model benchmarking."""

import os
import subprocess
import logging
from typing import Dict, Any, Optional
from prefect import task

logger = logging.getLogger(__name__)


@task
def run_spanish_evaluation(
    model_name: str,
    tasks: str,
    batch_size: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    wandb_args: str = "",
    log_samples: bool = True,
    max_length: int = 2048,
    limit: Optional[int] = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    cache_requests: bool = True,
    trust_remote_code: bool = False,
    num_concurrent: int = 8,
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Spanish language evaluation using lm-evaluation-harness via API.
    """
    return _run_evaluation(
        model_name=model_name,
        tasks=tasks,
        batch_size=batch_size,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        max_length=max_length,
        wandb_args=wandb_args,
        log_samples=log_samples,
        limit=limit,
        lm_eval_path=lm_eval_path,
        cache_requests=cache_requests,
        trust_remote_code=trust_remote_code,
        num_concurrent=num_concurrent,
        language="Spanish",
        cuda_devices=cuda_devices
    )


@task
def run_portuguese_evaluation(
    model_name: str,
    tasks: str,
    batch_size: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    max_length: int = 2048,
    wandb_args: str = "",
    log_samples: bool = True,
    limit: Optional[int] = None,
    lm_eval_path: str = "/home/mauro/dev/portu",
    cache_requests: bool = True,
    trust_remote_code: bool = False,
    num_concurrent: int = 8,
    tokenizer_backend: str = "huggingface",
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Portuguese language evaluation using lm-evaluation-harness via API.
    """
    return _run_evaluation(
        model_name=model_name,
        tasks=tasks,
        batch_size=batch_size,
        output_path=output_path,
        server_info=server_info,
        api_test_result=api_test_result,
        wandb_args=wandb_args,
        max_length=max_length,
        log_samples=log_samples,
        limit=limit,
        lm_eval_path=lm_eval_path,
        cache_requests=cache_requests,
        trust_remote_code=trust_remote_code,
        num_concurrent=num_concurrent,
        language="Portuguese",
        tokenizer_backend=tokenizer_backend,
        cuda_devices=cuda_devices
    )


def _run_evaluation(
    model_name: str,
    tasks: str,
    batch_size: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    max_length: int = 2048,
    wandb_args: str = "",
    log_samples: bool = True,
    limit: Optional[int] = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    cache_requests: bool = True,
    trust_remote_code: bool = False,
    num_concurrent: int = 8,
    language: str = "Unknown",
    tokenizer_backend: Optional[str] = None,
    cuda_devices: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness using vLLM API server.
    
    Args:
        model_name: The model to evaluate
        tasks: Tasks to run (e.g., "latam_es" or "latam_pt")
        batch_size: Batch size configuration
        output_path: Output path for results
        server_info: Dictionary containing server info from start_vllm_server
        wandb_args: Weights & Biases arguments
        log_samples: Whether to log samples
        limit: Limit number of examples per task (useful for testing)
        lm_eval_path: Path to lm-evaluation-harness installation
        cache_requests: Whether to enable request caching
        trust_remote_code: Whether to trust remote code when loading models
        num_concurrent: Number of concurrent API requests
        cuda_devices: CUDA devices to use (e.g., "3" or "2,3")
        
    Returns:
        Dictionary with execution results and metadata
    """
    server_url = server_info["url"]
    
    logger.info(f"Starting {language} evaluation for model: {model_name}, tasks: {tasks}")
    
    file_logger = logging.getLogger('benchy.lm_eval')
    try:
        file_logger.info(f"=== Starting {language} LM Evaluation ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Tasks: {tasks}")
        file_logger.info(f"Server URL: {server_url}")
        file_logger.info(f"Batch size: {batch_size}")
        file_logger.info(f"Concurrent requests: {num_concurrent}")
        if batch_size != "1":
            file_logger.info("Note: vLLM supports batched requests with varying sequence lengths")
        if limit:
            file_logger.info(f"Limit: {limit} (testing mode)")
    except (RuntimeError, OSError):
        pass
    
    # Build model_args string with optimizations for API-only usage
    model_args_parts = [
        f"model={model_name}",
        f"max_length={max_length}",
        f"base_url={server_url}/v1/completions",
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
    
    model_args_str = ",".join(model_args_parts)
    
    # Build the lm_eval command for API mode
    cmd_parts = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args_str,
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--output_path", output_path
    ]
    
    # Note: Chat template application is disabled due to compatibility issues
    # with JsonChatStr objects in lm-eval. The model will still work but may
    # not be optimally formatted for instruct models.
    # TODO: Investigate proper chat template handling in lm-eval
    if False and ("instruct" in model_name.lower() or "chat" in model_name.lower()):
        cmd_parts.append("--apply_chat_template")
        logger.info("Detected instruct/chat model, enabling chat template")
        try:
            file_logger.info("Auto-enabled chat template for instruct/chat model")
        except (RuntimeError, OSError):
            pass
    
    if log_samples:
        cmd_parts.append("--log_samples")
        
    if limit is not None:
        cmd_parts.extend(["--limit", str(limit)])
        
    if wandb_args:
        cmd_parts.extend(["--wandb_args", wandb_args])
        
    if cache_requests:
        cmd_parts.extend(["--cache_requests", "true"])
        
    if trust_remote_code:
        cmd_parts.append("--trust_remote_code")
    
    cmd = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd}")
    try:
        file_logger.info(f"Full command: {cmd}")
    except (RuntimeError, OSError):
        pass
    
    try:
        # Set environment variables for datasets trust
        env_vars = "HF_DATASETS_TRUST_REMOTE_CODE=true"
        
        # Activate the lm-eval venv and run command
        venv_cmd = f"source {lm_eval_path}/.venv/bin/activate && {env_vars} {cmd}"
        
        # Set up CPU-only environment for lm-eval client
        env = os.environ.copy()
        # Force PyTorch to use CPU only
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTORCH_CUDA_ALLOC_CONF"] = ""  # Clear any CUDA memory settings
        
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
            file_logger.info(f"Output saved to: {output_path}")
        except (RuntimeError, OSError):
            pass
        
        return {
            "model_name": model_name,
            "tasks": tasks,
            "output_path": output_path,
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
def gather_results(spanish_results: Dict[str, Any], portuguese_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gather results from Spanish and Portuguese evaluations.
    """
    return {
        "spanish_results": spanish_results,
        "portuguese_results": portuguese_results
    }
