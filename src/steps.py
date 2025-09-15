"""ZenML steps for ML model benchmarking."""

import os
import subprocess
import logging
from typing import Dict, Any
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


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
    limit: int = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    use_accelerate: bool = False,
    num_gpus: int = 1,
    mixed_precision: str = "no",
    cache_requests: bool = True
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness in its own virtual environment.
    
    Args:
        model_name: The model to evaluate
        model_args: Model arguments string
        tasks: Tasks to run
        device: Device to use (ignored if use_accelerate=True)
        batch_size: Batch size configuration
        output_path: Output path for results
        wandb_args: Weights & Biases arguments
        log_samples: Whether to log samples
        limit: Limit number of examples per task (useful for testing)
        lm_eval_path: Path to lm-evaluation-harness installation
        use_accelerate: Whether to use accelerate for multi-GPU
        num_gpus: Number of GPUs to use (when use_accelerate=True)
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        cache_requests: Whether to enable request caching
        
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
    else:
        # Single GPU mode
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
    
    cmd = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd}")
    file_logger.info(f"Full command: {cmd}")
    
    try:
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
                logger.info(f"[lm_eval] {line}")
                file_logger.info(f"[lm_eval] {line}")
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            logger.error(f"lm_eval failed with return code {return_code}")
            file_logger.error(f"lm_eval failed with return code {return_code}")
            file_logger.error("=== LM Evaluation FAILED ===")
            raise RuntimeError(f"lm_eval execution failed with return code {return_code}")
            
        logger.info("lm_eval completed successfully")
        file_logger.info("=== LM Evaluation COMPLETED SUCCESSFULLY ===")
        file_logger.info(f"Output saved to: {output_path}")
        
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
