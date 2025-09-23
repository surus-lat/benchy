"""Prefect steps for vLLM-based ML model benchmarking."""

import os
import subprocess
import logging
import time
import requests
import atexit
import psutil
from typing import Dict, Any, Optional, Tuple
from prefect import task
from prefect.cache_policies import NO_CACHE
import logging

logger = logging.getLogger(__name__)


def kill_existing_vllm_processes(model_name: str, port: int = 8000) -> None:
    """
    Kill existing vLLM processes running the same model or using the same port.
    
    Args:
        model_name: The model name to match against running processes
        port: The port to check for conflicts
    """
    current_user = os.getenv('USER', '')
    killed_count = 0
    
    logger.info(f"Checking for existing vLLM processes with model: {model_name} or port: {port}")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
        try:
            # Check if it's a Python process owned by current user
            if (proc.info['name'] == 'python' and 
                proc.info['username'] == current_user and 
                proc.info['cmdline']):
                
                cmdline = ' '.join(proc.info['cmdline'])
                
                # Check if it's a vLLM API server process with the same model OR same port
                should_kill = False
                if 'vllm.entrypoints.openai.api_server' in cmdline:
                    if f'--model {model_name}' in cmdline:
                        logger.info(f"Found existing vLLM process (PID: {proc.info['pid']}) with model: {model_name}")
                        should_kill = True
                    elif f'--port {port}' in cmdline:
                        logger.info(f"Found existing vLLM process (PID: {proc.info['pid']}) using port: {port}")
                        should_kill = True
                
                if should_kill:
                    logger.info(f"Command: {cmdline}")
                    
                    # Kill the process
                    proc.terminate()
                    killed_count += 1
                    
                    # Wait a bit for graceful termination
                    try:
                        proc.wait(timeout=5)
                        logger.info(f"Successfully terminated process {proc.info['pid']}")
                    except psutil.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        proc.kill()
                        logger.warning(f"Force killed process {proc.info['pid']}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process might have died or we don't have access
            continue
        except Exception as e:
            logger.warning(f"Error checking process {proc.info.get('pid', 'unknown')}: {e}")
            continue
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} existing vLLM process(es)")
        # Give a moment for cleanup
        time.sleep(3)
    else:
        logger.info("No existing vLLM processes found")


@task(cache_policy=NO_CACHE)
def start_vllm_server(
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.6,
    enforce_eager: bool = True,
    limit_mm_per_prompt: str = '{"images": 0, "audios": 0}',
    hf_cache: str = "/home/mauro/.cache/huggingface",
    hf_token: str = "",
    vllm_venv_path: str = "/home/mauro/dev/benchy/.venv",
    startup_timeout: int = 900,
    cuda_devices: Optional[str] = None,
    kv_cache_memory: Optional[int] = None
) -> Dict[str, Any]:
    """
    Start vLLM server with configurable parameters.
    
    Args:
        model_name: The model to serve
        host: Host to bind to
        port: Port to use
        tensor_parallel_size: Number of GPUs for tensor parallelism (-tp)
        max_model_len: Maximum model length
        gpu_memory_utilization: GPU memory utilization
        enforce_eager: Whether to enforce eager execution
        limit_mm_per_prompt: Multimodal limits as JSON string
        hf_cache: Hugging Face cache directory
        hf_token: Hugging Face token
        vllm_venv_path: Path to vLLM virtual environment
        startup_timeout: Timeout in seconds for server startup (default: 900s = 15min)
        cuda_devices: CUDA devices to use (e.g., "3" or "2,3")
        
    Returns:
        Dictionary with server info: {"pid": int, "url": str, "port": int}
    """
    logger.info(f"Starting vLLM server for model: {model_name}")
    
    # Kill any existing vLLM processes for the same model or port
    kill_existing_vllm_processes(model_name, port)
    
    # Get Python logger for detailed file logging with safe error handling
    file_logger = logging.getLogger('benchy.vllm_server')
    try:
        file_logger.info(f"=== Starting vLLM Server ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Host: {host}, Port: {port}")
        file_logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        file_logger.info(f"Max model length: {max_model_len}")
    except (RuntimeError, OSError):
        # Continue if file logging fails
        pass
    
    # Build command parts
    cuda_visible_devices = cuda_devices if cuda_devices is not None else "2,3"
    cmd_parts = [
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", host,
        "--model", model_name,
        "--port", str(port),
        "-tp", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--limit-mm-per-prompt", f"'{limit_mm_per_prompt}'"  # Quote the JSON string
    ]
    
    if enforce_eager:
        cmd_parts.append("--enforce-eager")
    
    if kv_cache_memory is not None:
        cmd_parts.extend(["--kv-cache-memory", str(kv_cache_memory)])
    
    # Set up environment
    env = os.environ.copy()
    if hf_cache:
        env["HF_CACHE"] = hf_cache
    if hf_token:  # Only set if not empty string
        env["HF_TOKEN"] = hf_token
    
    # Add PyTorch CUDA memory management optimization
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    cmd_str = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd_str}")
    try:
        file_logger.info(f"Command: {cmd_str}")
    except (RuntimeError, OSError):
        pass
    
    # Activate the vLLM venv and start the server
    venv_cmd = f"source {vllm_venv_path}/bin/activate && {cmd_str}"
    
    # Start the server using the vLLM virtual environment
    process = subprocess.Popen(
        venv_cmd,
        shell=True,
        env=env,
        executable="/bin/bash"
    )
    
    # Wait for server to be ready
    server_url = f"http://{host}:{port}"
    start_time = time.time()
    
    logger.info(f"Waiting for vLLM server to start at {server_url}...")
    logger.info(f"Timeout set to {startup_timeout} seconds ({startup_timeout//60} minutes)")
    
    while time.time() - start_time < startup_timeout:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ vLLM server is ready!")
                try:
                    file_logger.info("vLLM server started successfully")
                except (RuntimeError, OSError):
                    pass
                return {"pid": process.pid, "url": server_url, "port": port}
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    # Server failed to start
    process.kill()
    error_msg = f"vLLM server failed to start within {startup_timeout} seconds ({startup_timeout//60} minutes)"
    logger.error(error_msg)
    try:
        file_logger.error(error_msg)
    except (RuntimeError, OSError):
        pass
    raise TimeoutError(error_msg)


@task(cache_policy=NO_CACHE)
def test_vllm_api(server_info: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Test vLLM API server with a simple completion request.
    
    Args:
        server_info: Dictionary containing server info from start_vllm_server
        model_name: Model name to test
        
    Returns:
        Dictionary with test results
    """
    server_url = server_info["url"]
    port = server_info["port"]
    
    logger.info(f"Testing vLLM API at {server_url}")
    
    file_logger = logging.getLogger('benchy.api_test')
    try:
        file_logger.info(f"=== Testing vLLM API ===")
        file_logger.info(f"Server URL: {server_url}")
        file_logger.info(f"Model: {model_name}")
    except (RuntimeError, OSError):
        pass
    
    test_payload = {
        "model": model_name,
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ API test successful!")
            try:
                file_logger.info("API test completed successfully")
                file_logger.info(f"Response: {result}")
            except (RuntimeError, OSError):
                pass
            
            return {
                "status": "success",
                "server_url": server_url,
                "port": port,
                "model_name": model_name,
                "response": result
            }
        else:
            error_msg = f"API test failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            try:
                file_logger.error(error_msg)
            except (RuntimeError, OSError):
                pass
            raise RuntimeError(error_msg)
            
    except Exception as e:
        error_msg = f"API test failed: {str(e)}"
        logger.error(error_msg)
        try:
            file_logger.error(error_msg)
        except (RuntimeError, OSError):
            pass
        raise


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


@task(cache_policy=NO_CACHE)
def stop_vllm_server(server_info: Dict[str, Any], upload_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stop vLLM server process.
    
    Args:
        server_info: Dictionary containing server info from start_vllm_server
        upload_result: Dictionary containing upload result from upload_results
    Returns:
        Dictionary with termination results
    """
    pid = server_info["pid"]
    logger.info(f"Stopping vLLM server (PID: {pid})")
    
    file_logger = logging.getLogger('benchy.vllm_server')
    try:
        file_logger.info(f"=== Stopping vLLM Server (PID: {pid}) ===")
    except (RuntimeError, OSError):
        pass
    
    try:
        import psutil
        process = psutil.Process(pid)
        process.terminate()
        
        # Wait for graceful termination
        try:
            process.wait(timeout=30)
            logger.info("✅ vLLM server terminated gracefully")
            try:
                file_logger.info("vLLM server terminated gracefully")
            except (RuntimeError, OSError):
                pass
            return {"status": "success", "method": "graceful", "pid": pid}
        except psutil.TimeoutExpired:
            # Force kill if graceful termination fails
            process.kill()
            logger.warning("⚠️ vLLM server force-killed after timeout")
            try:
                file_logger.warning("vLLM server force-killed after timeout")
            except (RuntimeError, OSError):
                pass
            return {"status": "success", "method": "force_kill", "pid": pid}
            
    except psutil.NoSuchProcess:
        logger.info("vLLM server process already terminated")
        try:
            file_logger.info("vLLM server process already terminated")
        except (RuntimeError, OSError):
            pass
        return {"status": "already_terminated", "pid": pid}
    except Exception as e:
        error_msg = f"Error stopping vLLM server: {str(e)}"
        logger.error(error_msg)
        try:
            file_logger.error(error_msg)
        except (RuntimeError, OSError):
            pass
        return {"status": "error", "error": str(e), "pid": pid}

