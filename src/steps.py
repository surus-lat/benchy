"""ZenML steps for ML model benchmarking."""

import os
import subprocess
import logging
from typing import Dict, Any, Optional
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def test_model_download(
    model_name: str,
    model_args: str,
    use_accelerate: bool = False,
    num_gpus: int = 1,
    mixed_precision: str = "no",
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    use_vllm: bool = False,
    gpus_per_model: int = 1,
    model_replicas: int = 2,
    use_local_api: bool = False,
    local_api_base_url: str = "http://localhost:8000/v1/completions",
    num_concurrent: int = 8
) -> Dict[str, Any]:
    """
    Test model download and basic functionality before running full evaluation.
    
    This step downloads the model and runs a simple test to ensure it works correctly.
    It helps separate model download time from evaluation time and catches errors early.
    
    For API mode, this step skips the actual model testing and just validates the configuration.
    
    Args:
        model_name: The model to test
        model_args: Model arguments string
        use_accelerate: Whether accelerate will be used for evaluation
        num_gpus: Number of GPUs that will be used
        mixed_precision: Mixed precision mode
        lm_eval_path: Path to lm-evaluation-harness installation
        use_vllm: Whether VLLM will be used for evaluation
        gpus_per_model: Number of GPUs per model instance for VLLM
        model_replicas: Number of model replicas for VLLM
        use_local_api: Whether to use local API mode instead of direct model loading
        local_api_base_url: Base URL for the local API server
        
    Returns:
        Dictionary with test results and model metadata
    """
    logger.info(f"Testing model download and functionality: {model_name}")
    
    # Get Python logger for detailed file logging
    file_logger = logging.getLogger('benchy.model_test')
    file_logger.info(f"=== Starting Model Test ===")
    file_logger.info(f"Model: {model_name}")
    file_logger.info(f"Model args: {model_args}")
    
    # For local API mode, skip the actual model testing since the API server handles the model
    if use_vllm and use_local_api:
        logger.info("Local API mode detected - skipping direct model testing")
        file_logger.info("Local API mode: model will be served via API, skipping direct testing")
        file_logger.info(f"API URL: {local_api_base_url}")
        return {
            "model_name": model_name,
            "test_status": "skipped_api_mode",
            "message": "Model testing skipped - using local API mode",
            "api_url": local_api_base_url,
            "model_size_gb": "unknown_api_mode",
            "peak_memory_gb": "unknown_api_mode"
        }
    
    if use_accelerate:
        file_logger.info(f"Multi-GPU: {num_gpus} GPUs with accelerate")
        file_logger.info(f"Mixed precision: {mixed_precision}")
    elif use_vllm:
        file_logger.info(f"VLLM mode: {gpus_per_model} GPUs per model, {model_replicas} replicas")
    file_logger.info(f"LM eval path: {lm_eval_path}")
    
    # Build the test command
    cmd_parts = [
        "python", "scripts/test_model.py",
        "--model_name", model_name,
        "--model_args", model_args
    ]
    
    if use_accelerate:
        cmd_parts.extend(["--use_accelerate"])
        cmd_parts.extend(["--num_gpus", str(num_gpus)])
        cmd_parts.extend(["--mixed_precision", mixed_precision])
    elif use_vllm:
        cmd_parts.extend(["--use_vllm"])
        cmd_parts.extend(["--gpus_per_model", str(gpus_per_model)])
        cmd_parts.extend(["--model_replicas", str(model_replicas)])
    
    # Properly quote the model_args if it contains JSON
    for i, part in enumerate(cmd_parts):
        if i > 0 and cmd_parts[i-1] == "--model_args" and "{" in part:
            cmd_parts[i] = f"'{part}'"  # Single quote to prevent shell parsing
    
    cmd = " ".join(cmd_parts)
    logger.info(f"Executing test command: {cmd}")
    file_logger.info(f"Test command: {cmd}")
    
    try:
        # Explicitly activate the lm-eval venv and run test
        venv_cmd = f"source {lm_eval_path}/.venv/bin/activate && {cmd}"
        
        # Stream output in real-time
        process = subprocess.Popen(
            venv_cmd,
            shell=True,
            cwd=lm_eval_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=os.environ.copy(),
            executable="/bin/bash",
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output while streaming it
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:  # Only log non-empty lines
                logger.info(f"[model_test] {line}")
                # Add error handling for file logging to prevent lock issues
                try:
                    file_logger.info(f"[model_test] {line}")
                except (RuntimeError, OSError) as log_err:
                    # If file logging fails due to lock conflict, continue with console only
                    pass
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            logger.error(f"Model test failed with return code {return_code}")
            try:
                file_logger.error(f"Model test failed with return code {return_code}")
                file_logger.error("=== Model Test FAILED ===")
            except (RuntimeError, OSError):
                pass
            raise RuntimeError(f"Model test execution failed with return code {return_code}")
            
        logger.info("Model test completed successfully")
        try:
            file_logger.info("=== Model Test COMPLETED SUCCESSFULLY ===")
        except (RuntimeError, OSError):
            pass
        
        # Extract key information from output for metadata
        test_result = {
            "model_name": model_name,
            "model_args": model_args,
            "test_stdout": stdout,
            "test_return_code": return_code,
            "test_command": cmd,
            "use_accelerate": use_accelerate,
            "num_gpus": num_gpus,
            "mixed_precision": mixed_precision,
            "use_vllm": use_vllm,
            "gpus_per_model": gpus_per_model,
            "model_replicas": model_replicas
        }
        
        # Try to extract model size info from output
        for line in output_lines:
            if "Parameters:" in line:
                try:
                    params_str = line.split("Parameters:")[-1].strip().replace(",", "")
                    test_result["num_parameters"] = int(params_str)
                except (ValueError, IndexError):
                    pass
            elif "Size:" in line and "GB" in line:
                try:
                    size_str = line.split("Size:")[-1].split("GB")[0].strip()
                    test_result["model_size_gb"] = float(size_str)
                except (ValueError, IndexError):
                    pass
        
        return test_result
        
    except Exception as e:
        logger.error(f"Error running model test: {str(e)}")
        try:
            file_logger.error(f"Error running model test: {str(e)}")
            file_logger.error("=== Model Test FAILED ===")
        except (RuntimeError, OSError):
            pass
        raise


@step
def run_lm_evaluation(
    model_name: str,
    model_args: str,
    tasks: str,
    device: str,
    batch_size: str,
    output_path: str,
    wandb_args: str = "",
    log_samples: bool = True,
    limit: Optional[int] = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    use_accelerate: bool = False,
    num_gpus: int = 1,
    mixed_precision: str = "no",
    cache_requests: bool = True,
    trust_remote_code: bool = False,
    model_test_results: Dict[str, Any] = None,
    use_vllm: bool = False,
    gpus_per_model: int = 1,
    model_replicas: int = 2,
    use_local_api: bool = False,
    local_api_base_url: str = "http://localhost:8000/v1/completions",
    num_concurrent: int = 8
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness in its own virtual environment.
    
    Args:
        model_name: The model to evaluate
        model_args: Model arguments string
        tasks: Tasks to run
        device: Device to use (ignored if use_accelerate=True or use_vllm=True)
        batch_size: Batch size configuration
        output_path: Output path for results
        wandb_args: Weights & Biases arguments
        log_samples: Whether to log samples
        limit: Limit number of examples per task (useful for testing)
        lm_eval_path: Path to lm-evaluation-harness installation
        use_accelerate: Whether to use accelerate for multi-GPU (mutually exclusive with use_vllm)
        num_gpus: Number of GPUs to use (when use_accelerate=True)
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        cache_requests: Whether to enable request caching
        trust_remote_code: Whether to trust remote code when loading models
        use_vllm: Whether to use VLLM backend (mutually exclusive with use_accelerate)
        gpus_per_model: Number of GPUs per model instance for VLLM
        model_replicas: Number of model replicas for VLLM
        
    Returns:
        Dictionary with execution results and metadata
    """
    logger.info(f"Starting evaluation for model: {model_name}")
    
    # Get Python logger for detailed file logging
    file_logger = logging.getLogger('benchy.lm_eval')
    file_logger.info(f"=== Starting LM Evaluation ===")
    file_logger.info(f"Model: {model_name}")
    file_logger.info(f"Tasks: {tasks}")
    if use_accelerate:
        file_logger.info(f"Multi-GPU: {num_gpus} GPUs with accelerate")
        file_logger.info(f"Mixed precision: {mixed_precision}")
    elif use_vllm:
        file_logger.info(f"VLLM mode: {gpus_per_model} GPUs per model, {model_replicas} replicas")
    else:
        file_logger.info(f"Device: {device}")
    file_logger.info(f"Batch size: {batch_size}")
    file_logger.info(f"Cache requests: {cache_requests}")
    if limit:
        file_logger.info(f"Limit: {limit} (testing mode)")
    
    # Build the lm_eval command
    if use_accelerate:
        # Use accelerate launch for multi-GPU (with debugging info)
        cmd_parts = [
            "accelerate", "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            f"--mixed_precision={mixed_precision}",
            "--dynamo_backend=no",
            "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", tasks,
            "--batch_size", batch_size,
            "--output_path", output_path
        ]
    elif use_vllm:
        if use_local_api:
            # Local API mode - connects to running vLLM server
            # API models only support simple batch sizes, not "auto:N" format
            api_batch_size = "auto" if batch_size.startswith("auto") else batch_size
            cmd_parts = [
                "lm_eval",
                "--model", "local-completions",
                "--model_args", f"model={model_name},base_url={local_api_base_url},num_concurrent={num_concurrent},max_retries=3",
                "--tasks", tasks,
                "--batch_size", api_batch_size,
                "--output_path", output_path
            ]
        else:
            # Direct VLLM mode
            cmd_parts = [
                "lm_eval",
                "--model", "vllm",
                "--model_args", model_args,
                "--tasks", tasks,
                "--batch_size", batch_size,
                "--output_path", output_path
            ]
    else:
        # Single GPU mode with HF
        cmd_parts = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", tasks,
            "--device", device,
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
    
    # Properly quote the model_args if it contains JSON
    for i, part in enumerate(cmd_parts):
        if i > 0 and cmd_parts[i-1] == "--model_args" and "{" in part:
            cmd_parts[i] = f"'{part}'"  # Single quote to prevent shell parsing
    
    cmd = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd}")
    file_logger.info(f"Full command: {cmd}")
    
    try:
        # Setup environment for accelerate/multiprocessing safety
        env = os.environ.copy()
        
        # Disable problematic logging in subprocess to avoid lock conflicts
        # This prevents the subprocess from inheriting our complex logging setup
        if use_accelerate:
            env["PYTHONPATH"] = f"{lm_eval_path}:{env.get('PYTHONPATH', '')}"
            # Disable file logging in subprocess to prevent lock conflicts
            env["DISABLE_FILE_LOGGING"] = "1"
        
        # Explicitly activate the lm-eval venv and run command
        # This ensures we use the correct Python environment
        venv_cmd = f"source {lm_eval_path}/.venv/bin/activate && {cmd}"
        
        # Stream output in real-time
        process = subprocess.Popen(
            venv_cmd,
            shell=True,
            cwd=lm_eval_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=env,
            executable="/bin/bash",
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output while streaming it
        output_lines = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:  # Only log non-empty lines
                    # Use thread-safe logging approach
                    logger.info(f"[lm_eval] {line}")
                    # Only log to file if we're not in accelerate mode or if it's safe
                    try:
                        file_logger.info(f"[lm_eval] {line}")
                    except (RuntimeError, OSError) as log_err:
                        # If file logging fails due to lock conflict, continue with console only
                        pass
                    output_lines.append(line)
        except Exception as stream_error:
            logger.warning(f"Stream reading interrupted: {stream_error}")
            # Continue to wait for process completion
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            logger.error(f"lm_eval failed with return code {return_code}")
            try:
                file_logger.error(f"lm_eval failed with return code {return_code}")
                file_logger.error("=== LM Evaluation FAILED ===")
            except (RuntimeError, OSError):
                # File logging failed, but continue with error reporting
                pass
            raise RuntimeError(f"lm_eval execution failed with return code {return_code}")
            
        logger.info("lm_eval completed successfully")
        try:
            file_logger.info("=== LM Evaluation COMPLETED SUCCESSFULLY ===")
            file_logger.info(f"Output saved to: {output_path}")
        except (RuntimeError, OSError):
            # File logging failed, but the evaluation succeeded
            pass
        
        return {
            "model_name": model_name,
            "tasks": tasks,
            "output_path": output_path,
            "stdout": stdout,
            "stderr": "",  # Merged into stdout
            "return_code": return_code,
            "command": cmd
        }
        
    except Exception as e:
        logger.error(f"Error running lm_eval: {str(e)}")
        try:
            file_logger.error(f"Error running lm_eval: {str(e)}")
            file_logger.error("=== LM Evaluation FAILED ===")
        except (RuntimeError, OSError):
            # File logging failed, but continue with error reporting
            pass
        raise


@step
def upload_results(
    eval_results: Dict[str, Any],
    script_path: str = "/home/mauro/dev/leaderboard",
    script_name: str = "run_pipeline.py"
) -> Dict[str, Any]:
    """
    Upload evaluation results using the leaderboard upload script.
    
    Args:
        eval_results: Results from the evaluation step
        script_path: Path to the leaderboard script directory  
        script_name: Name of the upload script
        
    Returns:
        Dictionary with upload results and metadata
    """
    logger.info(f"Starting upload for model: {eval_results['model_name']}")
    
    # Get Python logger for detailed file logging
    file_logger = logging.getLogger('benchy.upload')
    file_logger.info(f"=== Starting Upload ===")
    file_logger.info(f"Model: {eval_results['model_name']}")
    file_logger.info(f"Script path: {script_path}")
    file_logger.info(f"Script name: {script_name}")
    
    # Build the upload command using uv
    cmd = f"uv run {script_name}"
    logger.info(f"Executing command: {cmd}")
    file_logger.info(f"Upload command: {cmd}")
    
    try:
        # Run upload script using uv (which manages its own venv)
        # uv automatically uses the project's virtual environment
        
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=script_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=os.environ.copy(),
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Collect output while streaming it
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:  # Only log non-empty lines
                logger.info(f"[upload] {line}")
                file_logger.info(f"[upload] {line}")
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            logger.error(f"Upload script failed with return code {return_code}")
            file_logger.error(f"Upload script failed with return code {return_code}")
            file_logger.error("=== Upload FAILED ===")
            raise RuntimeError(f"Upload script execution failed with return code {return_code}")
            
        logger.info("Upload completed successfully")
        file_logger.info("=== Upload COMPLETED SUCCESSFULLY ===")
        
        return {
            "model_name": eval_results["model_name"],
            "upload_stdout": stdout,
            "upload_stderr": "",  # Merged into stdout
            "upload_return_code": return_code,
            "upload_command": cmd,
            "eval_results": eval_results
        }
        
    except Exception as e:
        logger.error(f"Error running upload script: {str(e)}")
        raise
