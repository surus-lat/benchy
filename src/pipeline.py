"""Main ZenML pipeline for ML model benchmarking."""

from datetime import datetime
from zenml import pipeline
from zenml.logger import get_logger
from .steps import test_model_download, run_lm_evaluation, upload_results

logger = get_logger(__name__)


def create_run_name(model_name: str, limit: int = None) -> str:
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
    model_args: str, 
    tasks: str,
    device: str,
    batch_size: str,
    output_path: str,
    wandb_args: str = "",
    log_samples: bool = True,
    limit: int = None,
    lm_eval_path: str = "/home/mauro/dev/lm-evaluation-harness",
    upload_script_path: str = "/home/mauro/dev/leaderboard",
    upload_script_name: str = "run_pipeline.py",
    use_accelerate: bool = False,
    num_gpus: int = 1,
    mixed_precision: str = "no",
    cache_requests: bool = True
):
    """
    Complete benchmarking pipeline that runs evaluation and uploads results.
    
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
        upload_script_path: Path to upload script directory
        upload_script_name: Name of upload script
        use_accelerate: Whether to use accelerate for multi-GPU
        num_gpus: Number of GPUs to use (when use_accelerate=True)
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        cache_requests: Whether to enable request caching
    """
    logger.info(f"Starting benchmark pipeline for model: {model_name}")
    
    # Step 1: Test model download and functionality
    test_results = test_model_download(
        model_name=model_name,
        model_args=model_args,
        use_accelerate=use_accelerate,
        num_gpus=num_gpus,
        mixed_precision=mixed_precision,
        lm_eval_path=lm_eval_path
    )
    
    # Step 2: Run evaluation
    eval_results = run_lm_evaluation(
        model_name=model_name,
        model_args=model_args,
        tasks=tasks,
        device=device,
        batch_size=batch_size,
        output_path=output_path,
        wandb_args=wandb_args,
        log_samples=log_samples,
        limit=limit,
        lm_eval_path=lm_eval_path,
        use_accelerate=use_accelerate,
        num_gpus=num_gpus,
        mixed_precision=mixed_precision,
        cache_requests=cache_requests,
        model_test_results=test_results
    )
    
    # Step 3: Upload results
    upload_result = upload_results(
        eval_results=eval_results,
        script_path=upload_script_path,
        script_name=upload_script_name
    )
    
    logger.info("Benchmark pipeline completed successfully")
    return upload_result
