"""Generic benchmark runner for any task + interface combination.

This module provides the core benchmark execution loop that works with
any task implementing BaseTask and any interface implementing BaseInterface.
"""

import asyncio
import base64
import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List

from .checkpoint import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint,
)
from .protocols import BaseTask, BaseInterface, check_compatibility
from .output_diagnostics import analyze_output, aggregate_diagnostics

logger = logging.getLogger(__name__)

PERFORMANCE_METRIC_CANDIDATES = (
    "document_extraction_score",
    "overall_extraction_quality_score",
    "extraction_quality_score",
    "field_f1_partial",
    "accuracy",
    "exact_match",
    "bleu",
    "chrf",
    "comet",
    "score",
)
PERFORMANCE_METRIC_EXCLUDE_KEYS = {
    "sample_id",
    "valid",
    "error",
    "error_type",
    "schema_fingerprint",
}


def _coerce_numeric_metric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    return None


def _select_primary_performance_metric(per_sample_metrics: List[Dict[str, Any]]) -> Optional[str]:
    available_counts: Dict[str, int] = {}

    for metric in per_sample_metrics:
        if not isinstance(metric, dict):
            continue
        for key, value in metric.items():
            if key in PERFORMANCE_METRIC_EXCLUDE_KEYS:
                continue
            if _coerce_numeric_metric(value) is None:
                continue
            available_counts[key] = available_counts.get(key, 0) + 1

    if not available_counts:
        return None

    for key in PERFORMANCE_METRIC_CANDIDATES:
        if key in available_counts:
            return key

    return sorted(
        available_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]


def _compute_quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * q
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    if lower_idx == upper_idx:
        return float(lower)
    weight = position - lower_idx
    return float(lower + (upper - lower) * weight)


def _performance_bucket(value: float, *, q1: float, median: float, q3: float) -> str:
    if value <= q1:
        return "q1_bottom"
    if value <= median:
        return "q2_lower_mid"
    if value <= q3:
        return "q3_upper_mid"
    return "q4_top"


def build_performance_summary(
    per_sample_metrics: List[Dict[str, Any]],
    *,
    top_k: int = 10,
) -> Dict[str, Any]:
    total_samples = len(per_sample_metrics)
    primary_metric = _select_primary_performance_metric(per_sample_metrics)
    if not primary_metric:
        return {
            "status": "unavailable",
            "reason": "no_numeric_metric",
            "total_samples": total_samples,
        }

    metric_rows: List[Dict[str, Any]] = []
    for metric in per_sample_metrics:
        if not isinstance(metric, dict):
            continue
        value = _coerce_numeric_metric(metric.get(primary_metric))
        if value is None:
            continue
        metric_rows.append(
            {
                "sample_id": metric.get("sample_id"),
                "value": value,
                "valid": bool(metric.get("valid", True)),
            }
        )

    if not metric_rows:
        return {
            "status": "unavailable",
            "reason": "primary_metric_not_populated",
            "primary_metric": primary_metric,
            "total_samples": total_samples,
        }

    ordered_values = sorted(row["value"] for row in metric_rows)
    q1 = _compute_quantile(ordered_values, 0.25)
    median = _compute_quantile(ordered_values, 0.50)
    q3 = _compute_quantile(ordered_values, 0.75)

    ascending_rows = sorted(
        metric_rows,
        key=lambda row: (row["value"], str(row.get("sample_id") or "")),
    )
    descending_rows = sorted(
        metric_rows,
        key=lambda row: (-row["value"], str(row.get("sample_id") or "")),
    )

    return {
        "status": "ok",
        "primary_metric": primary_metric,
        "direction": "higher_is_better",
        "total_samples": total_samples,
        "samples_with_metric": len(metric_rows),
        "samples_missing_metric": max(0, total_samples - len(metric_rows)),
        "quartiles": {
            "min": float(ordered_values[0]),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(ordered_values[-1]),
        },
        "bottom_quartile_threshold": q1,
        "top_quartile_threshold": q3,
        "lowest_samples": ascending_rows[:top_k],
        "highest_samples": descending_rows[:top_k],
    }


def build_per_sample_metrics_artifact(
    *,
    model_name: str,
    task_name: str,
    timestamp: str,
    per_sample_metrics: List[Dict[str, Any]],
    performance_summary: Dict[str, Any],
) -> Dict[str, Any]:
    primary_metric = performance_summary.get("primary_metric")
    quartiles = performance_summary.get("quartiles") or {}
    q1 = _coerce_numeric_metric(quartiles.get("q1"))
    median = _coerce_numeric_metric(quartiles.get("median"))
    q3 = _coerce_numeric_metric(quartiles.get("q3"))

    ranking_lookup: Dict[str, Dict[str, Any]] = {}
    if performance_summary.get("status") == "ok" and primary_metric:
        all_ranked_rows = sorted(
            [
                {
                    "sample_id": metric.get("sample_id"),
                    "value": float(metric.get(primary_metric)),
                }
                for metric in per_sample_metrics
                if isinstance(metric, dict) and _coerce_numeric_metric(metric.get(primary_metric)) is not None
            ],
            key=lambda row: (row["value"], str(row.get("sample_id") or "")),
        )
        metric_count = len(all_ranked_rows)
        for index, row in enumerate(all_ranked_rows, start=1):
            ranking_lookup[str(row.get("sample_id"))] = {
                "rank_ascending": index,
                "rank_descending": metric_count - index + 1,
                "percentile": (index - 1) / (metric_count - 1) if metric_count > 1 else 1.0,
                "value": row["value"],
            }

    entries: List[Dict[str, Any]] = []
    for metric in per_sample_metrics:
        if not isinstance(metric, dict):
            continue
        entry = deepcopy(metric)
        sample_id = entry.get("sample_id")
        if primary_metric and sample_id is not None and str(sample_id) in ranking_lookup:
            ranking = ranking_lookup[str(sample_id)]
            entry["performance_metric"] = primary_metric
            entry["performance_value"] = ranking["value"]
            entry["performance_rank_ascending"] = ranking["rank_ascending"]
            entry["performance_rank_descending"] = ranking["rank_descending"]
            entry["performance_percentile"] = ranking["percentile"]
            if q1 is not None and median is not None and q3 is not None:
                entry["performance_bucket"] = _performance_bucket(
                    ranking["value"],
                    q1=q1,
                    median=median,
                    q3=q3,
                )
        entries.append(entry)

    return {
        "model": model_name,
        "task": task_name,
        "timestamp": timestamp,
        "total_samples": len(entries),
        "performance_summary": performance_summary,
        "entries": entries,
    }


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
                    aggregate["performance_summary"] = build_performance_summary(prior_metrics)
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
        diagnostics_entries: List[Dict[str, Any]] = []
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

                if isinstance(metrics, dict):
                    metrics.setdefault("sample_id", sample.get("id"))

                results["per_sample_metrics"].append(metrics)
                metrics_by_id[sample["id"]] = metrics

                diagnostics = analyze_output(sample=sample, output=output, task=self.task)
                diagnostics_entries.append(diagnostics)

                if output.get("error") or metrics.get("error_type") == "invalid_response":
                    logger.warning(
                        "Sample %s invalid output: error=%s error_type=%s finish_reason=%s completion_tokens=%s prompt_tokens=%s diagnostic_class=%s whitespace_run=%s repetition=%s raw_len=%s",
                        sample["id"],
                        output.get("error"),
                        output.get("error_type"),
                        diagnostics.get("finish_reason"),
                        diagnostics.get("completion_tokens"),
                        diagnostics.get("prompt_tokens"),
                        diagnostics.get("diagnostic_class"),
                        diagnostics.get("max_whitespace_run"),
                        diagnostics.get("repetition_detected"),
                        diagnostics.get("raw_length"),
                    )
                
                # Log sample details if requested
                if self.log_samples:
                    sample_data = {
                        "id": sample["id"],
                        "prediction": output.get("output"),
                        "raw_prediction": output.get("raw"),
                        "expected": sample.get("expected"),
                        "metrics": metrics,
                        "error": output.get("error"),
                        "finish_reason": diagnostics.get("finish_reason"),
                        "completion_tokens": diagnostics.get("completion_tokens"),
                        "prompt_tokens": diagnostics.get("prompt_tokens"),
                    }
                    sample_data["diagnostics"] = diagnostics
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
        counts, rates = aggregate_diagnostics(diagnostics_entries, run_samples=run_samples)
        aggregate["diagnostic_counts"] = counts
        aggregate["diagnostic_rates"] = rates
        aggregate["performance_summary"] = build_performance_summary(results["per_sample_metrics"])
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

        # Print compact diagnostic summary (dict metrics are skipped above).
        diagnostic_counts = metrics.get("diagnostic_counts")
        diagnostic_rates = metrics.get("diagnostic_rates")
        if isinstance(diagnostic_counts, dict) and diagnostic_counts:
            parts = []
            for cls, count in sorted(diagnostic_counts.items(), key=lambda kv: kv[1], reverse=True):
                rate = None
                if isinstance(diagnostic_rates, dict):
                    rate = diagnostic_rates.get(cls)
                if isinstance(rate, (float, int)):
                    parts.append(f"{cls}={count} ({float(rate):.3f})")
                else:
                    parts.append(f"{cls}={count}")
            logger.info("diagnostic_summary: %s", ", ".join(parts))

        performance_summary = metrics.get("performance_summary")
        if isinstance(performance_summary, dict) and performance_summary.get("status") == "ok":
            metric_name = performance_summary.get("primary_metric")
            quartiles = performance_summary.get("quartiles") or {}
            logger.info(
                "performance_summary: metric=%s min=%.4f q1=%.4f median=%.4f q3=%.4f max=%.4f samples=%s/%s",
                metric_name,
                float(quartiles.get("min", 0.0)),
                float(quartiles.get("q1", 0.0)),
                float(quartiles.get("median", 0.0)),
                float(quartiles.get("q3", 0.0)),
                float(quartiles.get("max", 0.0)),
                performance_summary.get("samples_with_metric", 0),
                performance_summary.get("total_samples", 0),
            )
            lowest_samples = performance_summary.get("lowest_samples") or []
            if lowest_samples:
                lowest_summary = ", ".join(
                    f"{sample.get('sample_id')}={float(sample.get('value', 0.0)):.4f}"
                    for sample in lowest_samples[:5]
                )
                logger.info("lowest_samples: %s", lowest_summary)

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


def _save_image_artifacts(samples: List[Dict], output_dir: Path, timestamp: str) -> List[Dict]:
    """Extract and save image artifacts from samples, replacing base64 with file references.
    
    Also generates comparison visualizations for mask-based tasks if ground truth is available.
    
    Args:
        samples: List of sample dictionaries
        output_dir: Output directory for images
        timestamp: Timestamp for file naming
        
    Returns:
        Modified samples with image file references instead of base64
    """
    from ..tasks.common.visualization import save_mask_comparison_for_sample
    
    images_dir = output_dir / f"images_{timestamp}"
    comparisons_dir = output_dir / f"comparisons_{timestamp}"
    images_dir.mkdir(exist_ok=True)
    
    modified_samples = []
    comparison_count = 0
    
    for sample in samples:
        modified_sample = sample.copy()
        
        # Check if prediction contains base64 image data
        prediction = sample.get("prediction")
        if prediction and isinstance(prediction, str):
            # Try to decode as base64
            try:
                # Strip data URI prefix if present
                b64_data = prediction
                if b64_data.startswith("data:"):
                    b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
                
                # Decode and save
                image_data = base64.b64decode(b64_data)
                sample_id = sample.get("id", "unknown")
                image_filename = f"{sample_id}.png"
                image_path = images_dir / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                # Replace prediction with file reference
                modified_sample["prediction"] = str(image_path.relative_to(output_dir))
                modified_sample["prediction_saved_as"] = image_filename
                
                logger.debug(f"Saved image artifact for {sample_id} to {image_filename}")
                
                # Generate comparison visualization if ground truth available
                ground_truth = sample.get("mask_path") or sample.get("expected")
                if ground_truth and Path(str(ground_truth)).exists():
                    # Check if this is a valid sample (not an error)
                    metrics = sample.get("metrics", {})
                    if metrics.get("valid", True):
                        source_image = sample.get("image_path")
                        comparison_path = comparisons_dir / f"{sample_id}_comparison.png"
                        
                        if save_mask_comparison_for_sample(
                            prediction_bytes=image_data,
                            ground_truth_path=str(ground_truth),
                            source_image_path=source_image,
                            output_path=comparison_path,
                            metrics=metrics,
                        ):
                            modified_sample["comparison"] = str(comparison_path.relative_to(output_dir))
                            comparison_count += 1
                
            except Exception as e:
                # If decoding fails, keep original (might not be an image)
                logger.debug(f"Could not save image for {sample.get('id')}: {e}")
        
        modified_samples.append(modified_sample)
    
    if comparison_count > 0:
        logger.info(f"Generated {comparison_count} comparison visualizations in {comparisons_dir.name}")
    
    return modified_samples


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    task_name: str,
    log_samples: bool = False,
    task_answer_type: Optional[str] = None,
    task_instance: Optional[Any] = None,
) -> None:
    """Save benchmark results to files.
    
    Args:
        results: Results from BenchmarkRunner.run()
        output_dir: Output directory
        model_name: Model name for filename
        task_name: Task name for filename
        log_samples: Whether to save sample details
        task_answer_type: Task answer type (e.g., "image_artifact") for special handling
        task_instance: Optional task/handler instance for extra artifact hooks
    """
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_")
    performance_summary = results.get("aggregate_metrics", {}).get("performance_summary")
    if not isinstance(performance_summary, dict):
        performance_summary = build_performance_summary(results.get("per_sample_metrics", []))
        results.setdefault("aggregate_metrics", {})["performance_summary"] = performance_summary
    
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

    per_sample_metrics = results.get("per_sample_metrics", [])
    if per_sample_metrics:
        per_sample_metrics_file = output_dir / f"{safe_name}_{timestamp}_per_sample_metrics.json"
        per_sample_payload = build_per_sample_metrics_artifact(
            model_name=model_name,
            task_name=task_name,
            timestamp=timestamp,
            per_sample_metrics=per_sample_metrics,
            performance_summary=performance_summary,
        )
        with open(per_sample_metrics_file, "w", encoding="utf-8") as f:
            json.dump(per_sample_payload, f, indent=2, default=str)
        logger.info(f"Saved per-sample metrics to {per_sample_metrics_file}")

        performance_summary_file = output_dir / f"{safe_name}_{timestamp}_performance_summary.json"
        with open(performance_summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model_name,
                    "task": task_name,
                    "timestamp": timestamp,
                    "performance_summary": performance_summary,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Saved performance summary to {performance_summary_file}")
    
    # Save samples if requested
    if log_samples and results.get("samples"):
        samples = results["samples"]
        
        # For image artifact tasks, save actual images and replace base64 with file references
        if task_answer_type == "image_artifact":
            logger.info(f"Processing image artifacts for {len(samples)} samples")
            samples = _save_image_artifacts(samples, output_dir, timestamp)
        
        samples_file = output_dir / f"{safe_name}_{timestamp}_samples.json"
        with open(samples_file, "w") as f:
            json.dump({
                "model": model_name,
                "task": task_name,
                "timestamp": timestamp,
                "total_samples": len(samples),
                "samples": samples,
            }, f, indent=2)
        logger.info(f"Saved {len(samples)} samples to {samples_file}")
    
    # Save text report
    report_file = output_dir / f"{safe_name}_{timestamp}_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Metrics:\n")
        f.write("-" * 40 + "\n")
        for key, value in results["aggregate_metrics"].items():
            if isinstance(value, dict):
                continue
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        if performance_summary.get("status") == "ok":
            quartiles = performance_summary.get("quartiles") or {}
            f.write("\nPerformance Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  primary_metric: {performance_summary.get('primary_metric')}\n")
            f.write(f"  samples_with_metric: {performance_summary.get('samples_with_metric', 0)}\n")
            f.write(f"  min: {float(quartiles.get('min', 0.0)):.4f}\n")
            f.write(f"  q1: {float(quartiles.get('q1', 0.0)):.4f}\n")
            f.write(f"  median: {float(quartiles.get('median', 0.0)):.4f}\n")
            f.write(f"  q3: {float(quartiles.get('q3', 0.0)):.4f}\n")
            f.write(f"  max: {float(quartiles.get('max', 0.0)):.4f}\n")
            lowest_samples = performance_summary.get("lowest_samples") or []
            if lowest_samples:
                f.write("  lowest_samples:\n")
                for sample in lowest_samples[:5]:
                    f.write(
                        f"    - {sample.get('sample_id')}: {float(sample.get('value', 0.0)):.4f}\n"
                    )
        f.write("=" * 60 + "\n")
    logger.info(f"Saved report to {report_file}")

    if task_instance is not None and hasattr(task_instance, "build_additional_artifacts"):
        try:
            artifact_paths = task_instance.build_additional_artifacts(
                results=results,
                output_dir=output_dir,
                safe_model_name=safe_name,
                timestamp=timestamp,
                task_name=task_name,
            )
            if artifact_paths:
                for artifact_path in artifact_paths:
                    logger.info(f"Saved additional artifact: {artifact_path}")
        except Exception as exc:
            logger.warning(f"Failed to write additional artifacts for {task_name}: {exc}")
    
