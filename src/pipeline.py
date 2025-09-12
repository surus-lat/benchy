"""Main ZenML pipeline for ML model benchmarking."""

from zenml import pipeline
from zenml.logger import get_logger
from .steps import run_lm_evaluation, upload_results

logger = get_logger(__name__)


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
    upload_script_name: str = "run_pipeline.py"
):
    """
    Complete benchmarking pipeline that runs evaluation and uploads results.
    
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
        upload_script_path: Path to upload script directory
        upload_script_name: Name of upload script
    """
    logger.info(f"Starting benchmark pipeline for model: {model_name}")
    
    # Step 1: Run evaluation
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
        lm_eval_path=lm_eval_path
    )
    
    # Step 2: Upload results
    upload_result = upload_results(
        eval_results=eval_results,
        script_path=upload_script_path,
        script_name=upload_script_name
    )
    
    logger.info("Benchmark pipeline completed successfully")
    return upload_result
