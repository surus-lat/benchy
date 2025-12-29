"""Generic benchmark runner for any task + interface combination.

This module provides the core benchmark execution loop that works with
any task implementing BaseTask and any interface implementing BaseInterface.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .checkpoint import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint,
)
from .protocols import BaseTask, BaseInterface, check_compatibility

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Generic benchmark runner that works with any task + interface.
    
    This runner handles:
    - Batch processing with configurable batch size
    - Checkpointing for resumable runs
    - Progress logging
    - Result aggregation
    
    Tasks provide data and metrics, interfaces handle communication.
    The runner orchestrates the evaluation loop.
    """
    
    def __init__(
        self,
        task: BaseTask,
        interface: BaseInterface,
        config: Dict[str, Any],
    ):
        """Initialize the benchmark runner.
        
        Args:
            task: Task instance providing samples and metrics
            interface: Interface instance for AI system communication
            config: Configuration dictionary with:
                - model_name: Name of model being evaluated
                - batch_size: Number of concurrent requests (default: 20)
                - output_dir: Directory for results and checkpoints
                - log_samples: Whether to save sample details (default: False)
        """
        self.task = task
        self.interface = interface
        self.config = config
        
        self.model_name = config.get("model_name", "unknown")
        self.batch_size = config.get("batch_size", 20)
        self.output_dir = Path(config.get("output_dir", "./results"))
        self.log_samples = config.get("log_samples", False)
        
        # Check compatibility
        report = check_compatibility(task, interface)
        if not report.compatible:
            raise ValueError(f"Task and interface incompatible: {', '.join(report.errors)}")
        for warning in report.warnings:
            logger.warning(warning)
    
    async def run(
        self,
        limit: Optional[int] = None,
        no_resume: bool = False,
    ) -> Dict[str, Any]:
        """Execute the benchmark.
        
        Args:
            limit: Maximum samples to evaluate (None for all)
            no_resume: If True, ignore existing checkpoints
            
        Returns:
            Results dictionary with:
            - per_sample_metrics: List of per-sample metric dicts
            - aggregate_metrics: Aggregated summary metrics
            - samples: Sample details (if log_samples=True)
        """
        logger.info(f"Starting benchmark: {self.task.get_task_name()}")
        logger.info(f"Model: {self.model_name}")
        
        try:
            # Test connection
            if not await self.interface.test_connection(max_retries=3, timeout=30):
                raise ConnectionError("Cannot establish connection to AI system")
            
            # Load dataset
            self.task.load()
            all_samples = list(self.task.get_samples(limit=limit))
            
            if not all_samples:
                logger.warning("No samples to process")
                return {"per_sample_metrics": [], "aggregate_metrics": {}, "samples": []}
            
            # Handle checkpointing
            task_name = self.task.get_task_name()
            checkpoint_path = get_checkpoint_path(str(self.output_dir), self.model_name, task_name)
            
            config_for_hash = {
                "model": self.model_name,
                "task": task_name,
                "batch_size": self.batch_size,
            }
            config_hash = get_config_hash(config_for_hash)
            
            completed_ids = set()
            metrics_by_id = {}
            if not no_resume:
                completed_ids, metrics_by_id = load_checkpoint(checkpoint_path, config_hash)
            
            # Filter out completed samples
            samples = [s for s in all_samples if s["id"] not in completed_ids]
            
            if not samples:
                logger.info("All samples already completed!")
                if metrics_by_id:
                    prior_metrics = list(metrics_by_id.values())
                    aggregate = self.task.aggregate_metrics(prior_metrics)
                    aggregate["total_samples"] = len(prior_metrics)
                    aggregate["total_duration"] = 0.0
                    aggregate["throughput"] = 0.0
                    self._log_summary(aggregate)
                    return {
                        "per_sample_metrics": prior_metrics,
                        "aggregate_metrics": aggregate,
                        "samples": [],
                    }
                return {"per_sample_metrics": [], "aggregate_metrics": {}, "samples": []}
            
            logger.info(f"Processing {len(samples)} samples in batches of {self.batch_size}")
            if completed_ids:
                logger.info(f"Resuming from checkpoint: {len(completed_ids)} already done")
            
            # Run evaluation
            results = await self._evaluate_samples(
                samples=samples,
                checkpoint_path=checkpoint_path,
                config_hash=config_hash,
                no_resume=no_resume,
                completed_ids=list(completed_ids),
                metrics_by_id=metrics_by_id,
            )
            
            # Cleanup checkpoint on successful completion
            if not no_resume and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Benchmark completed - checkpoint removed")
            
            return results
        finally:
            close_fn = getattr(self.interface, "close", None)
            if close_fn:
                try:
                    result = close_fn()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.debug(f"Failed to close interface: {exc}")
    
    async def _evaluate_samples(
        self,
        samples: List[Dict],
        checkpoint_path: Path,
        config_hash: str,
        no_resume: bool,
        completed_ids: List[str],
        metrics_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate samples in batches."""
        results = {
            "samples": [],
            "per_sample_metrics": list(metrics_by_id.values()),
        }
        start_time = time.time()
        
        # Log example for first sample
        if samples:
            self._log_example(samples[0])
        
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(samples), self.batch_size):
            batch = samples[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")
            
            # Prepare requests using interface
            requests = [
                self.interface.prepare_request(sample, self.task)
                for sample in batch
            ]
            
            # Generate outputs
            batch_start = time.time()
            outputs = await self.interface.generate_batch(requests)
            batch_duration = time.time() - batch_start
            
            # Count errors
            errors = sum(1 for r in outputs if r.get("error"))
            if errors:
                logger.warning(f"  Batch had {errors}/{len(batch)} errors")
            
            throughput = len(batch) / batch_duration if batch_duration > 0 else 0
            logger.info(f"  Batch completed in {batch_duration:.2f}s ({throughput:.2f} samples/s)")
            
            # Calculate metrics for each sample
            for sample, output in zip(batch, outputs):
                try:
                    metrics = self.task.calculate_metrics(
                        prediction=output.get("output"),
                        expected=sample.get("expected"),
                        sample=sample,
                        error=output.get("error"),
                        error_type=output.get("error_type"),
                    )
                except Exception as e:
                    logger.error(f"Error calculating metrics for {sample['id']}: {e}")
                    # Use task's error metrics structure for fallback
                    # This ensures task-specific metric structures are preserved
                    error_msg = str(e)
                    error_type = output.get("error_type", "connectivity_error")
                    metrics = self.task.get_error_metrics(error_msg, error_type)
                
                results["per_sample_metrics"].append(metrics)
                metrics_by_id[sample["id"]] = metrics
                
                # Log sample details if requested
                if self.log_samples:
                    sample_data = {
                        "id": sample["id"],
                        "prediction": output.get("output"),
                        "raw_prediction": output.get("raw"),
                        "expected": sample.get("expected"),
                        "metrics": metrics,
                        "error": output.get("error"),
                    }
                    # Add task-specific fields
                    for key in ["title", "topic", "text"]:
                        if key in sample:
                            sample_data[key] = sample[key][:500] if isinstance(sample.get(key), str) else sample[key]
                    results["samples"].append(sample_data)
                
                completed_ids.append(sample["id"])
            
            # Periodic checkpoint
            if not no_resume and len(completed_ids) > 0 and len(completed_ids) % 50 == 0:
                save_checkpoint(
                    checkpoint_path,
                    completed_ids,
                    config_hash,
                    metrics_by_id=metrics_by_id,
                )
        
        # Aggregate metrics
        logger.info("Calculating aggregate metrics...")
        aggregate = self.task.aggregate_metrics(results["per_sample_metrics"])
        total_samples = len(results["per_sample_metrics"])
        aggregate["total_samples"] = total_samples
        aggregate["total_duration"] = time.time() - start_time
        run_samples = len(samples)
        aggregate["throughput"] = run_samples / aggregate["total_duration"] if aggregate["total_duration"] > 0 else 0
        aggregate["run_samples"] = run_samples
        results["aggregate_metrics"] = aggregate
        
        self._log_summary(aggregate)
        return results
    
    def _log_example(self, sample: Dict) -> None:
        """Log example prompt for the first sample."""
        logger.info("=" * 60)
        logger.info("EXAMPLE (First Sample)")
        logger.info("=" * 60)
        logger.info(f"Sample ID: {sample['id']}")
        logger.info(f"Task: {self.task.get_task_name()}")
        
        try:
            system_prompt, user_prompt = self.task.get_prompt(sample)
            logger.info("System Prompt:")
            for line in system_prompt.split('\n')[:5]:
                logger.info(f"  {line}")
            logger.info("User Prompt (first 500 chars):")
            logger.info(f"  {user_prompt[:500]}...")
        except Exception as e:
            logger.debug(f"Could not get prompt: {e}")
        
        logger.info("=" * 60)
        
        # Save to file
        example_file = self.output_dir / f"example_{self.task.get_task_name()}.txt"
        example_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            system_prompt, user_prompt = self.task.get_prompt(sample)
            with open(example_file, "w", encoding="utf-8") as f:
                f.write(f"Sample ID: {sample['id']}\n")
                f.write(f"Task: {self.task.get_task_name()}\n")
                f.write(f"Model: {self.model_name}\n\n")
                f.write("System Prompt:\n")
                f.write("-" * 40 + "\n")
                f.write(system_prompt + "\n\n")
                f.write("User Prompt:\n")
                f.write("-" * 40 + "\n")
                f.write(user_prompt + "\n")
        except Exception:
            pass
    
    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log benchmark summary."""
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Task: {self.task.get_task_name()}")
        logger.info(f"Samples: {metrics.get('total_samples', 'N/A')}")
        logger.info(f"Duration: {metrics.get('total_duration', 0):.2f}s")
        logger.info(f"Throughput: {metrics.get('throughput', 0):.2f} samples/s")
        
        # Log task-specific metrics
        for key, value in metrics.items():
            if key not in ['total_samples', 'total_duration', 'throughput']:
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                elif isinstance(value, (int, str)):
                    logger.info(f"{key}: {value}")
        
        logger.info("=" * 60)


async def run_benchmark(
    task: BaseTask,
    interface: BaseInterface,
    config: Dict[str, Any],
    limit: Optional[int] = None,
    no_resume: bool = False,
) -> Dict[str, Any]:
    """Convenience function to run a benchmark.
    
    Args:
        task: Task instance
        interface: Interface instance
        config: Runner configuration
        limit: Sample limit
        no_resume: Skip checkpointing
        
    Returns:
        Benchmark results
    """
    runner = BenchmarkRunner(task, interface, config)
    return await runner.run(limit=limit, no_resume=no_resume)


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    task_name: str,
    log_samples: bool = False,
    mark_complete: bool = True,
) -> None:
    """Save benchmark results to files.
    
    Args:
        results: Results from BenchmarkRunner.run()
        output_dir: Output directory
        model_name: Model name for filename
        task_name: Task name for filename
        log_samples: Whether to save sample details
        mark_complete: Whether to write .done file (default: True)
    """
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_")
    
    # Save metrics JSON
    metrics_file = output_dir / f"{safe_name}_{timestamp}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "model": model_name,
            "task": task_name,
            "timestamp": timestamp,
            "metrics": results["aggregate_metrics"],
        }, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Save samples if requested
    if log_samples and results.get("samples"):
        samples_file = output_dir / f"{safe_name}_{timestamp}_samples.json"
        with open(samples_file, "w") as f:
            json.dump({
                "model": model_name,
                "task": task_name,
                "timestamp": timestamp,
                "total_samples": len(results["samples"]),
                "samples": results["samples"],
            }, f, indent=2)
        logger.info(f"Saved {len(results['samples'])} samples to {samples_file}")
    
    # Save text report
    report_file = output_dir / f"{safe_name}_{timestamp}_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Metrics:\n")
        f.write("-" * 40 + "\n")
        for key, value in results["aggregate_metrics"].items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("=" * 60 + "\n")
    logger.info(f"Saved report to {report_file}")
    
    # Mark task as complete (for pipeline resumption)
    if mark_complete:
        mark_task_complete(output_dir)


def mark_task_complete(output_dir: Path) -> None:
    """Write a .done file to mark task completion.
    
    This is used by the pipeline's TaskCompletionChecker to skip
    already-completed tasks when resuming a failed run.
    
    Args:
        output_dir: Task output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done_file = output_dir / ".done"
    done_file.touch()
    logger.info(f"Task marked complete: {done_file}")
