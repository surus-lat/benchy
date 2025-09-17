"""ZenML steps for vLLM-based ML model benchmarking."""

import os
import subprocess
import logging
import time
import requests
import atexit
from typing import Dict, Any, Tuple
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
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
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness"
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
        lm_eval_path: Path to lm-evaluation-harness installation (for venv)
        
    Returns:
        Dictionary with server info: {"pid": int, "url": str, "port": int}
    """
    logger.info(f"Starting vLLM server for model: {model_name}")
    
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
    cmd_parts = [
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
    
    # Set up environment
    env = os.environ.copy()
    if hf_cache:
        env["HF_CACHE"] = hf_cache
    if hf_token:  # Only set if not empty string
        env["HF_TOKEN"] = hf_token
    
    cmd_str = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd_str}")
    try:
        file_logger.info(f"Command: {cmd_str}")
    except (RuntimeError, OSError):
        pass
    
    # Activate the lm-eval venv and start the server
    venv_cmd = f"source {lm_eval_path}/.venv/bin/activate && {cmd_str}"
    
    # Start the server using the lm-eval virtual environment
    process = subprocess.Popen(
        venv_cmd,
        shell=True,
        env=env,
        executable="/bin/bash"
    )
    
    # Wait for server to be ready
    server_url = f"http://{host}:{port}"
    start_time = time.time()
    timeout = 300  # 5 minutes
    
    logger.info(f"Waiting for vLLM server to start at {server_url}...")
    
    while time.time() - start_time < timeout:
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
    error_msg = f"vLLM server failed to start within {timeout} seconds"
    logger.error(error_msg)
    try:
        file_logger.error(error_msg)
    except (RuntimeError, OSError):
        pass
    raise TimeoutError(error_msg)


@step(enable_cache=False)
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


@step
def run_lm_evaluation(
    model_name: str,
    tasks: str,
    batch_size: str,
    output_path: str,
    server_info: Dict[str, Any],
    api_test_result: Dict[str, Any],
    wandb_args: str = "",
    log_samples: bool = True,
    limit: int = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    cache_requests: bool = True,
    trust_remote_code: bool = False,
    num_concurrent: int = 8
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
        
    Returns:
        Dictionary with execution results and metadata
    """
    server_url = server_info["url"]
    
    logger.info(f"Starting evaluation for model: {model_name}, tasks: {tasks}")
    
    file_logger = logging.getLogger('benchy.lm_eval')
    try:
        file_logger.info(f"=== Starting LM Evaluation ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Tasks: {tasks}")
        file_logger.info(f"Server URL: {server_url}")
        file_logger.info(f"Batch size: {batch_size}")
        file_logger.info(f"Concurrent requests: {num_concurrent}")
        if limit:
            file_logger.info(f"Limit: {limit} (testing mode)")
    except (RuntimeError, OSError):
        pass
    
    # Build the lm_eval command for API mode
    cmd_parts = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", f"model={model_name},base_url={server_url}/v1/completions,num_concurrent={num_concurrent},max_retries=3",
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--output_path", output_path
    ]
    
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
        # Activate the lm-eval venv and run command
        venv_cmd = f"source {lm_eval_path}/.venv/bin/activate && {cmd}"
        
        # Stream output in real-time
        process = subprocess.Popen(
            venv_cmd,
            shell=True,
            cwd=lm_eval_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
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


@step(enable_cache=False)
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


@step  
def upload_results(
    spanish_results: Dict[str, Any],
    portuguese_results: Dict[str, Any],
    model_name: str,
    script_path: str = "/home/mauro/dev/leaderboard",
    script_name: str = "run_pipeline.py"
) -> Dict[str, Any]:
    """
    Upload evaluation results using the leaderboard upload script.
    
    Args:
        spanish_results: Results from Spanish evaluation step
        portuguese_results: Results from Portuguese evaluation step
        model_name: Name of the model being evaluated
        script_path: Path to the leaderboard script directory  
        script_name: Name of the upload script
        
    Returns:
        Dictionary with upload results and metadata
    """
    logger.info(f"Starting upload for model: {model_name}")
    
    file_logger = logging.getLogger('benchy.upload')
    try:
        file_logger.info(f"=== Starting Upload ===")
        file_logger.info(f"Model: {model_name}")
        file_logger.info(f"Script path: {script_path}")
        file_logger.info(f"Script name: {script_name}")
    except (RuntimeError, OSError):
        pass
    
    # Build the upload command using uv
    cmd = f"uv run {script_name}"
    logger.info(f"Executing command: {cmd}")
    try:
        file_logger.info(f"Upload command: {cmd}")
    except (RuntimeError, OSError):
        pass
    
    try:
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=script_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output while streaming it
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[upload] {line}")
                try:
                    file_logger.info(f"[upload] {line}")
                except (RuntimeError, OSError):
                    pass
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            error_msg = f"Upload script failed with return code {return_code}"
            logger.error(error_msg)
            try:
                file_logger.error(error_msg)
                file_logger.error("=== Upload FAILED ===")
            except (RuntimeError, OSError):
                pass
            raise RuntimeError(f"Upload script execution failed with return code {return_code}")
            
        logger.info("Upload completed successfully")
        try:
            file_logger.info("=== Upload COMPLETED SUCCESSFULLY ===")
        except (RuntimeError, OSError):
            pass
        
        return {
            "model_name": model_name,
            "upload_stdout": stdout,
            "upload_return_code": return_code,
            "upload_command": cmd,
            "spanish_results": spanish_results,
            "portuguese_results": portuguese_results
        }
        
    except Exception as e:
        error_msg = f"Error running upload script: {str(e)}"
        logger.error(error_msg)
        try:
            file_logger.error(error_msg)
        except (RuntimeError, OSError):
            pass
        raise