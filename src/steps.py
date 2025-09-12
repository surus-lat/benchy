"""ZenML steps for ML model benchmarking."""

import os
import subprocess
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
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness"
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness in its own virtual environment.
    
    Args:
        model_name: The model to evaluate
        model_args: Model arguments string
        tasks: Tasks to run
        device: Device to use
        batch_size: Batch size configuration
        output_path: Output path for results
        wandb_args: Weights & Biases arguments
        log_samples: Whether to log samples
        limit: Limit number of examples per task (useful for testing)
        lm_eval_path: Path to lm-evaluation-harness installation
        
    Returns:
        Dictionary with execution results and metadata
    """
    logger.info(f"Starting evaluation for model: {model_name}")
    
    # Build the lm_eval command
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
    
    cmd = " ".join(cmd_parts)
    logger.info(f"Executing command: {cmd}")
    
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
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            logger.error(f"lm_eval failed with return code {return_code}")
            raise RuntimeError(f"lm_eval execution failed with return code {return_code}")
            
        logger.info("lm_eval completed successfully")
        
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
    
    # Build the upload command using uv
    cmd = f"uv run {script_name}"
    logger.info(f"Executing command: {cmd}")
    
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
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout = "\n".join(output_lines)
        
        if return_code != 0:
            logger.error(f"Upload script failed with return code {return_code}")
            raise RuntimeError(f"Upload script execution failed with return code {return_code}")
            
        logger.info("Upload completed successfully")
        
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
