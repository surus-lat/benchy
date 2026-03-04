"""vLLM server management functions for ML model benchmarking."""

import os
import subprocess
import logging
import time
import requests
import psutil
import shlex
from typing import Dict, Any, Optional
from ..prefect_compat import task, NO_CACHE

from .venv_manager import VLLMVenvManager
from .vllm_config import VLLMServerConfig

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
    vllm_config: Optional[VLLMServerConfig] = None,
    *,
    model_path: Optional[str] = None,
    served_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Start vLLM server with configurable parameters.
    
    Args:
        model_name: The model name used for client requests and outputs.
        vllm_config: vLLM server configuration
        model_path: Optional local path / HF ID for the model weights to load in vLLM.
        served_model_name: Optional name exposed by the OpenAI API server (defaults to model_name).
        
    Returns:
        Dictionary with server info: {"pid": int, "url": str, "port": int}
    """
    model_to_load = model_path or model_name
    served_name = served_model_name or model_name
    logger.info("Starting vLLM server for model: %s (load=%s)", served_name, model_to_load)
    vllm_config = vllm_config or VLLMServerConfig()

    host = vllm_config.host
    port = vllm_config.port
    tensor_parallel_size = vllm_config.tensor_parallel_size
    max_model_len = vllm_config.max_model_len
    gpu_memory_utilization = vllm_config.gpu_memory_utilization
    enforce_eager = vllm_config.enforce_eager
    limit_mm_per_prompt = vllm_config.limit_mm_per_prompt
    hf_cache = vllm_config.hf_cache
    hf_token = vllm_config.hf_token
    vllm_venv_path = vllm_config.vllm_venv_path
    startup_timeout = vllm_config.startup_timeout
    cuda_devices = vllm_config.cuda_devices
    kv_cache_memory = vllm_config.kv_cache_memory
    vllm_version = vllm_config.vllm_version
    multimodal = vllm_config.multimodal
    max_num_seqs = vllm_config.max_num_seqs
    max_num_batched_tokens = vllm_config.max_num_batched_tokens
    trust_remote_code = vllm_config.trust_remote_code
    tokenizer_mode = vllm_config.tokenizer_mode
    config_format = vllm_config.config_format
    load_format = vllm_config.load_format
    tool_call_parser = vllm_config.tool_call_parser
    enable_auto_tool_choice = vllm_config.enable_auto_tool_choice
    kv_cache_dtype = vllm_config.kv_cache_dtype
    kv_offloading_size = vllm_config.kv_offloading_size
    skip_mm_profiling = vllm_config.skip_mm_profiling

    # Handle vLLM version - use main project environment if None
    if vllm_version is None:
        logger.info("No vLLM version specified, using main project environment")
        print("✅ Using default vLLM version from main project environment")
        actual_venv_path = vllm_venv_path  # Use the provided fallback path
    else:
        logger.info(f"Using vLLM version: {vllm_version}")
        # Manage vLLM virtual environment
        print(f"🔍 Configuring vLLM {vllm_version} environment...")
        venv_manager = VLLMVenvManager()
        actual_venv_path = venv_manager.ensure_venv_exists(vllm_version)
        
        logger.info(f"Using virtual environment: {actual_venv_path}")
        print(f"✅ Using vLLM {vllm_version} from: {actual_venv_path}")
    
    # Kill any existing vLLM processes for the same model or port
    kill_existing_vllm_processes(model_to_load, port)
    
    # Get Python logger for detailed file logging with safe error handling
    file_logger = logging.getLogger('benchy.vllm_server')
    try:
        file_logger.info("=== Starting vLLM Server ===")
        file_logger.info(f"Model: {served_name}")
        file_logger.info(f"Model load ref: {model_to_load}")
        file_logger.info(f"Host: {host}, Port: {port}")
        file_logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        file_logger.info(f"Max model length: {max_model_len}")
    except (RuntimeError, OSError):
        # Continue if file logging fails
        pass
    
    venv_python = os.path.join(actual_venv_path, "bin", "python")
    if not os.path.exists(venv_python):
        raise FileNotFoundError(f"vLLM Python executable not found: {venv_python}")
    cuda_visible_devices = cuda_devices if cuda_devices is not None else "2,3"
    cmd_parts = [
        venv_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--model",
        model_to_load,
        "--port",
        str(port),
        "-tp",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--uvicorn-log-level",
        "warning",
    ]

    if served_name != model_to_load:
        cmd_parts.extend(["--served-model-name", served_name])
    
    # Handle version-specific multimodal arguments (only when explicitly configured).
    # If limit_mm_per_prompt is unset, do not pass --limit-mm-per-prompt at all.
    if multimodal and limit_mm_per_prompt:
        if vllm_version and vllm_version.startswith(("0.6", "0.7")):
            # Older versions use repeated key=value format.
            # Accept either a JSON dict string or a simple "images=N,audios=M" style.
            try:
                import json as _json

                parsed = _json.loads(limit_mm_per_prompt) if isinstance(limit_mm_per_prompt, str) else {}
                images = parsed.get("images")
                audios = parsed.get("audios")
                if images is not None:
                    cmd_parts.extend(["--limit-mm-per-prompt", f"images={images}"])
                if audios is not None:
                    cmd_parts.extend(["--limit-mm-per-prompt", f"audios={audios}"])
                if images is None and audios is None:
                    # Fall back to raw string if we couldn't parse anything useful.
                    cmd_parts.extend(["--limit-mm-per-prompt", str(limit_mm_per_prompt)])
            except Exception:
                cmd_parts.extend(["--limit-mm-per-prompt", str(limit_mm_per_prompt)])
        else:
            # vLLM recent versions accept JSON form.
            cmd_parts.extend(["--limit-mm-per-prompt", str(limit_mm_per_prompt)])
    
    if enforce_eager:
        cmd_parts.append("--enforce-eager")
    
    # Add trust_remote_code parameter
    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")
    
    # Add Mistral-specific parameters
    if tokenizer_mode:
        cmd_parts.extend(["--tokenizer-mode", tokenizer_mode])
    
    if config_format:
        cmd_parts.extend(["--config-format", config_format])
    
    if load_format:
        cmd_parts.extend(["--load-format", load_format])
    
    if tool_call_parser:
        cmd_parts.extend(["--tool-call-parser", tool_call_parser])
    
    if enable_auto_tool_choice:
        cmd_parts.append("--enable-auto-tool-choice")
    
    # Add KV cache optimization parameters
    if kv_cache_dtype:
        cmd_parts.extend(["--kv-cache-dtype", kv_cache_dtype])
    
    if kv_offloading_size is not None:
        cmd_parts.extend(["--kv-offloading-size", str(kv_offloading_size)])
    
    if skip_mm_profiling:
        cmd_parts.append("--skip-mm-profiling")
    
    # Add performance optimization parameters
    if max_num_seqs is not None:
        cmd_parts.extend(["--max-num-seqs", str(max_num_seqs)])
    
    if max_num_batched_tokens is not None:
        cmd_parts.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    
    # Handle version-specific arguments
    if kv_cache_memory is not None:
        # --kv-cache-memory was removed in newer vLLM versions (0.8.0+)
        # For older versions, include it; for newer versions, skip it
        if vllm_version and vllm_version.startswith(("0.6", "0.7")):
            cmd_parts.extend(["--kv-cache-memory", str(kv_cache_memory)])
        # For vLLM 0.8.0+ or None (latest), this argument doesn't exist, so we skip it
    
    # Set up environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if hf_cache:
        env["HF_CACHE"] = hf_cache
    if hf_token:  # Only set if not empty string
        env["HF_TOKEN"] = hf_token
    
    # Add PyTorch CUDA memory management optimization
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Reduce vLLM server logging verbosity
    env["VLLM_LOGGING_LEVEL"] = "WARNING"  # Only show warnings and errors
    
    # Disable nanobind leak checking to prevent shutdown warnings
    env["NANOBIND_LEAK_CHECK"] = "0"
    
    # Additional cleanup environment variables
    env["PYTHONUNBUFFERED"] = "1"  # Ensure clean output
    env["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable CUDA blocking for cleaner shutdown
    
    cmd_str = shlex.join(cmd_parts)
    logger.info("Executing command: %s", cmd_str)
    try:
        file_logger.info("Command: %s", cmd_str)
    except (RuntimeError, OSError):
        pass
    
    # Start the server using the vLLM virtual environment
    process = subprocess.Popen(
        cmd_parts,
        env=env,
    )
    
    # Wait for server to be ready
    server_url = f"http://{host}:{port}"
    start_time = time.time()
    
    logger.info(f"Waiting for vLLM server to start at {server_url}...")
    logger.info(f"Timeout set to {startup_timeout} seconds ({startup_timeout//60} minutes)")
    
    while time.time() - start_time < startup_timeout:
        # Check if process crashed early (early failure detection)
        if process.poll() is not None:
            # Process exited - this is likely an error since vLLM should keep running
            error_msg = f"vLLM server process exited early with return code {process.returncode}"
            logger.error(error_msg)
            try:
                file_logger.error(error_msg)
            except (RuntimeError, OSError):
                pass
            raise RuntimeError(error_msg)
        
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
        file_logger.info("=== Testing vLLM API ===")
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
        import signal
        process = psutil.Process(pid)
        
        # Try graceful shutdown first
        try:
            # Send SIGTERM for graceful shutdown
            process.terminate()
            
            # Wait for graceful termination
            process.wait(timeout=15)
            logger.info("✅ vLLM server terminated gracefully")
            try:
                file_logger.info("vLLM server terminated gracefully")
            except (RuntimeError, OSError):
                pass
            return {"status": "success", "method": "graceful", "pid": pid}
        except psutil.TimeoutExpired:
            # If graceful termination fails, try SIGINT (Ctrl+C equivalent)
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=10)
                logger.info("✅ vLLM server terminated with SIGINT")
                try:
                    file_logger.info("vLLM server terminated with SIGINT")
                except (RuntimeError, OSError):
                    pass
                return {"status": "success", "method": "sigint", "pid": pid}
            except psutil.TimeoutExpired:
                # Force kill as last resort
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
