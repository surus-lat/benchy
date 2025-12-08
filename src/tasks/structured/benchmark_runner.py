"""Core benchmark runner logic."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from scipy import stats

from ...interfaces.llm_interface import LLMInterface
from ...common.checkpoint_utils import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint,
)
from .metrics import MetricsCalculator, ReportGenerator, classify_complexity

logger = logging.getLogger(__name__)


def _get_interface(config: Dict[str, Any], model_name: str, provider_type: str):
    """Factory function for interface selection based on provider type.
    
    Args:
        config: Configuration dictionary
        model_name: Model/system identifier
        provider_type: Provider type ('vllm', 'openai', 'anthropic', 'surus', etc.)
        
    Returns:
        Appropriate interface instance
    """
    if provider_type == "surus":
        from ...interfaces.surus_interface import SurusInterface
        return SurusInterface(config, model_name, provider_type)
    else:
        # Default to LLM interface for vllm, openai, anthropic
        return LLMInterface(config, model_name, provider_type)


class BenchmarkRunner:
    """Manages benchmark execution with checkpointing and complexity analysis."""
    
    def __init__(self, model_name: str, config: Dict[str, Any], task=None, provider_type: str = "vllm"):
        """Initialize benchmark runner.
        
        Args:
            model_name: Name of the model being evaluated
            config: Configuration dictionary
            task: Task instance (if None, will try to create from config)
            provider_type: Type of provider ('vllm', 'openai', 'anthropic', 'surus', etc.)
        """
        self.model_name = model_name
        self.config = config
        self.provider_type = provider_type
        self.llm = _get_interface(config, model_name, provider_type)
        
        # Initialize task - can be passed in or created from config
        if task is not None:
            self.task = task
        else:
            # Default to ParaloqTask for backward compatibility
            from .tasks import ParaloqTask
            self.task = ParaloqTask(config)
        
        self.metrics_calc = MetricsCalculator(config)
        self.batch_size = config.get("performance", {}).get("batch_size", 20)
    
    async def run(
        self,
        limit: Optional[int] = None,
        log_samples: bool = False,
        no_resume: bool = False,
    ) -> Dict[str, Any]:
        """Execute benchmark with optional checkpointing."""
        logger.info(f"Starting benchmark for model: {self.model_name}")
        
        # Test connection
        if not await self.llm.test_connection(max_retries=3, timeout=30):
            logger.error("Cannot establish connection to LLM provider")
            raise ConnectionError("LLM provider unavailable")
        
        # Load dataset
        self.task.load()
        all_samples = list(self.task.get_samples(limit=limit))
        
        # Handle checkpointing
        results_dir = self.config.get("output", {}).get("results_dir", "./results")
        task_name = self.task.get_task_name()
        checkpoint_path = get_checkpoint_path(results_dir, self.model_name, task_name)
        
        config_for_hash = {
            "model": self.model_name,
            "base_url": self.config.get("model", {}).get("base_url"),
            "temperature": self.config.get("model", {}).get("temperature"),
            "max_tokens": self.config.get("model", {}).get("max_tokens"),
            "batch_size": self.batch_size,
        }
        config_hash = get_config_hash(config_for_hash)
        
        completed_ids = set()
        if not no_resume:
            completed_ids = load_checkpoint(checkpoint_path, config_hash)
        
        # Filter completed samples
        samples = [s for s in all_samples if s["id"] not in completed_ids]
        if not samples:
            logger.info("All samples already completed!")
            return {"per_sample_metrics": [], "aggregate_metrics": {}, "samples": []}
        
        logger.info(f"Processing {len(samples)} samples in batches of {self.batch_size}")
        
        # Run evaluation
        results = await self._evaluate_samples(
            samples, checkpoint_path, config_hash, no_resume, log_samples, list(completed_ids)
        )
        
        # Compute complexity analysis
        complexity = self._compute_complexity(samples, results["per_sample_metrics"])
        if complexity:
            results["aggregate_metrics"]["complexity_analysis"] = complexity
        
        # Cleanup checkpoint
        if not no_resume and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("✓ Benchmark completed - checkpoint removed")
        
        return results
    
    async def _evaluate_samples(
        self,
        samples: List[Dict],
        checkpoint_path: Path,
        config_hash: str,
        no_resume: bool,
        log_samples: bool,
        completed_ids: List[str]
    ) -> Dict[str, Any]:
        """Evaluate samples in batches."""
        results = {"samples": [], "per_sample_metrics": []}
        start_time = time.time()
        
        # Log example message for first sample (before processing starts)
        if samples:
            first_sample = samples[0]
            system_prompt, user_prompt = self.task.get_prompt(first_sample)
            self._log_example_message(first_sample["id"], system_prompt, user_prompt, first_sample.get("schema"))
        
        for batch_idx in range(0, len(samples), self.batch_size):
            batch = samples[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx//self.batch_size + 1
            logger.info(f"Processing batch {batch_num}/"
                       f"{(len(samples) + self.batch_size - 1)//self.batch_size} "
                       f"({len(batch)} samples)...")
            
            # Prepare requests using interface's format
            # Interface handles whether it needs prompts (LLM) or raw data (HTTP)
            requests = [
                self.llm.prepare_request(s, self.task)
                for s in batch
            ]
            
            # Only log samples from first batch
            should_log_batch_samples = log_samples and batch_num == 1
            
            # Generate
            batch_start = time.time()
            outputs = await self.llm.generate_batch(requests)
            
            # Count errors
            errors = sum(1 for r in outputs if r["error"])
            if errors:
                logger.warning(f"  Batch had {errors}/{len(batch)} errors")
            
            logger.info(f"  Batch completed in {time.time()-batch_start:.2f}s "
                       f"({len(batch)/(time.time()-batch_start):.2f} samples/s)")
            
            # Calculate metrics
            for sample, output in zip(batch, outputs):
                try:
                    metrics = self.metrics_calc.calculate_all(
                        prediction=output["output"],
                        expected=sample["expected"],
                        schema=sample["schema"],
                        error=output["error"],
                    )
                except Exception as e:
                    logger.error(f"Error calculating metrics for sample {sample['id']}: {e}")
                    logger.error(f"  Output type: {type(output['output']).__name__}")
                    logger.error(f"  Expected type: {type(sample['expected']).__name__}")
                    # Create a failed metrics entry
                    metrics = {
                        "valid": False,
                        "schema_validity": 0.0,
                        "exact_match": False,
                        "field_f1_partial": 0.0,
                        "field_f1_strict": 0.0,
                        "field_precision_partial": 0.0,
                        "field_recall_partial": 0.0,
                        "field_precision_strict": 0.0,
                        "field_recall_strict": 0.0,
                        "type_accuracy": 0.0,
                        "hallucination_rate": 0.0,
                        "extraction_quality_score": 0.0,
                        "match_distribution": {},
                        "composite_scores": [],
                        "schema_complexity": {},
                        "error": f"Metrics calculation failed: {str(e)}"
                    }
                results["per_sample_metrics"].append(metrics)
                
                if should_log_batch_samples:
                    # Store both parsed output and raw text for debugging
                    # Only log first batch to reduce file size
                    sample_data = {
                        "id": sample["id"],
                        "title": sample["title"],
                        "topic": sample["topic"],
                        "prediction": output["output"],
                        "raw_prediction": output["raw"],  # Always store the raw text
                        "expected": sample["expected"],
                        "metrics": metrics,
                        "error": output["error"],
                    }
                    results["samples"].append(sample_data)
                
                completed_ids.append(sample["id"])
            
            # Checkpoint
            if not no_resume and len(completed_ids) > 0 and len(completed_ids) % 50 == 0:
                save_checkpoint(checkpoint_path, completed_ids, config_hash)
        
        # Aggregate
        logger.info("Calculating aggregate metrics...")
        aggregate = self.metrics_calc.aggregate_metrics(results["per_sample_metrics"])
        aggregate["total_duration"] = time.time() - start_time
        aggregate["throughput"] = len(samples) / aggregate["total_duration"]
        results["aggregate_metrics"] = aggregate
        
        self._log_summary(aggregate)
        return results
    
    def _compute_complexity(
        self,
        samples: List[Dict],
        metrics_list: List[Dict]
    ) -> Dict[str, Any]:
        """Compute schema complexity analysis."""
        valid_pairs = [
            (s, m) for s, m in zip(samples, metrics_list)
            if m.get("valid") and m.get("error") is None
        ]
        
        if not valid_pairs:
            return {}
        
        # Bin by complexity
        bins = {"simple": [], "medium": [], "complex": []}
        for sample, metrics in valid_pairs:
            bin_name = classify_complexity(sample.get("complexity_score", 0.0))
            bins[bin_name].append({"metrics": metrics, "sample": sample})
        
        # Compute bin stats
        bin_stats = {}
        for name, items in bins.items():
            if not items:
                continue
            ms = [i["metrics"] for i in items]
            bin_stats[name] = {
                "count": len(items),
                "eqs": np.mean([m["extraction_quality_score"] for m in ms]),
                "f1_partial": np.mean([m["field_f1_partial"] for m in ms]),
                "f1_strict": np.mean([m["field_f1_strict"] for m in ms]),
                "hallucination_rate": np.mean([m["hallucination_rate"] for m in ms]),
            }
        
        # Correlations
        correlations = {}
        if len(valid_pairs) >= 10:
            cs = [s.get("complexity_score", 0.0) for s, _ in valid_pairs]
            fs = [s.get("complexity", {}).get("total_fields", 0) for s, _ in valid_pairs]
            ds = [s.get("complexity", {}).get("max_nesting_depth", 0) for s, _ in valid_pairs]
            eqs = [m["extraction_quality_score"] for _, m in valid_pairs]
            f1s = [m["field_f1_partial"] for _, m in valid_pairs]
            
            correlations["complexity_score_vs_eqs"] = stats.pearsonr(cs, eqs)[0]
            correlations["total_fields_vs_f1"] = stats.pearsonr(fs, f1s)[0]
            correlations["max_depth_vs_f1"] = stats.pearsonr(ds, f1s)[0]
        
        return {"bins": bin_stats, "correlations": correlations}
    
    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log benchmark summary."""
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Samples: {metrics['total_samples']}")
        logger.info(f"Duration: {metrics['total_duration']:.2f}s ({metrics['throughput']:.2f} samples/s)")
        logger.info(f"EQS: {metrics['extraction_quality_score']:.3f}")
        logger.info(f"Schema Validity: {metrics['schema_validity_rate']:.2%}")
        logger.info(f"Exact Match: {metrics['exact_match_rate']:.2%}")
        logger.info(f"F1 (Partial): {metrics['field_f1_partial']:.3f}")
        logger.info(f"Hallucination Rate: {metrics['hallucination_rate']:.2%}")
        logger.info(f"Error Rate: {metrics['error_rate']:.2%}")
        logger.info("=" * 60)
    
    def _log_example_message(
        self,
        sample_id: str,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict] = None
    ) -> None:
        """Log example message for the first sample."""
        
        logger.info("=" * 80)
        logger.info("EXAMPLE MESSAGE (First Sample)")
        logger.info("=" * 80)
        logger.info(f"Sample ID: {sample_id}")
        logger.info(f"Task: {self.task.get_task_name()}")
        logger.info("")
        logger.info("System Prompt:")
        logger.info("-" * 80)
        for line in system_prompt.split('\n'):
            logger.info(f"  {line}")
        logger.info("")
        logger.info("User Prompt:")
        logger.info("-" * 80)
        for line in user_prompt.split('\n'):
            logger.info(f"  {line}")
        logger.info("=" * 80)
        
        # Also save to file for traceability
        output_dir = Path(self.config.get("output", {}).get("results_dir", "./results"))
        task_name = self.task.get_task_name()
        example_file = output_dir / f"example_message_{task_name}.txt"
        example_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(example_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("EXAMPLE MESSAGE (First Sample)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Sample ID: {sample_id}\n")
            f.write(f"Task: {task_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("\n")
            f.write("System Prompt:\n")
            f.write("-" * 80 + "\n")
            f.write(system_prompt + "\n")
            f.write("\n")
            f.write("User Prompt:\n")
            f.write("-" * 80 + "\n")
            f.write(user_prompt + "\n")
            if schema:
                f.write("\n")
                f.write("Schema (JSON):\n")
                f.write("-" * 80 + "\n")
                f.write(json.dumps(schema, indent=2) + "\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Example message saved to: {example_file}")


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    log_samples: bool,
    config: Dict[str, Any]
) -> None:
    """Save benchmark results to files."""
    from datetime import datetime
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_")
    
    # Save JSON metrics
    metrics_file = output_dir / f"{safe_name}_{timestamp}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({"model": model_name, "timestamp": timestamp, "metrics": results["aggregate_metrics"]}, f, indent=2)
    logging.info(f"Saved aggregate metrics to {metrics_file}")
    
    # Generate text report
    report_file = output_dir / f"{safe_name}_{timestamp}_report.txt"
    ReportGenerator(config).generate_text_report(model_name, results["aggregate_metrics"], report_file)
    
    # Save samples if requested (only first batch is logged)
    if log_samples:
        samples_file = output_dir / f"{safe_name}_{timestamp}_samples.json"
        with open(samples_file, "w") as f:
            json.dump({
                "model": model_name,
                "timestamp": timestamp,
                "note": "Only first batch of samples logged",
                "samples": results["samples"]
            }, f, indent=2)
        logging.info(f"Saved sample results (first batch only) to {samples_file}")

