#!/usr/bin/env python3
"""CLI interface for LLM structured data extraction benchmark."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from .benchmark_runner import BenchmarkRunner, save_results


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Suppress noisy HTTP logs unless DEBUG
    if log_level != "DEBUG":
        for logger_name in ["httpx", "openai", "httpcore"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


async def main_async(args):
    """Async main function."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        config = load_config(args.config)
        # Ensure output section exists
        config.setdefault("output", {})
        if args.log_samples:
            config["output"]["log_samples"] = True
        
        # Run benchmark
        runner = BenchmarkRunner(args.model, config)
        results = await runner.run(
            limit=args.limit,
            log_samples=args.log_samples,
            no_resume=args.no_resume,
        )
        
        # Save results
        output_dir = Path(config["output"].get("results_dir", "./results"))
        save_results(results, output_dir, args.model, args.log_samples, config)
        
        logger.info("Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Structured Data Extraction Benchmark"
    )
    parser.add_argument("model", help="Model name to evaluate")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--log-samples", action="store_true", help="Save per-sample results")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
