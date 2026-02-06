"""Implementation of `benchy probe` command."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .config_loader import load_config
from .logging_utils import setup_file_logging
from .run_id_manager import generate_run_id

logger = logging.getLogger(__name__)

# Import provider defaults from eval CLI
from .benchy_cli_eval import CLI_PROVIDER_DEFAULTS, MODEL_PROVIDER_TYPES


def add_probe_arguments(parser: argparse.ArgumentParser) -> None:
    """Add probe command arguments."""
    # Required
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to probe"
    )
    
    # Provider resolution (same as eval)
    parser.add_argument(
        "--base-url",
        help="API base URL (e.g., http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--provider",
        choices=["vllm", "openai", "anthropic", "together"],
        help="Provider type"
    )
    parser.add_argument(
        "--api-key",
        help="API key (or use --api-key-env)"
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment variable containing API key"
    )
    
    # Probe options
    parser.add_argument(
        "--profile",
        choices=["quick"],
        default="quick",
        help="Probe profile (only 'quick' in Phase 1)"
    )
    parser.add_argument(
        "--run-id",
        help="Run ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--output-path",
        help="Output directory (default: outputs/probe_outputs)"
    )
    parser.add_argument(
        "--global-timeout",
        type=int,
        default=180,
        help="Global timeout for all checks in seconds (default: 180)"
    )
    parser.add_argument(
        "--image-max-edge",
        type=int,
        help="Max image edge for multimodal test (optional)"
    )


def run_probe(args: argparse.Namespace) -> int:
    """Execute probe command.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load environment
    load_dotenv()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate run ID if not provided
    run_id = args.run_id or generate_run_id(prefix="probe")
    
    # Determine output path
    if args.output_path:
        output_base = Path(args.output_path)
    else:
        output_base = Path("outputs/probe_outputs")
    
    # Create model-specific output directory
    model_name_segment = args.model_name.split("/")[-1]
    output_path = output_base / run_id / model_name_segment
    
    logger.info(f"Benchy Probe")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {output_path}")
    
    # Build connection_info
    try:
        connection_info = _build_connection_info(args)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Run probe
    try:
        from .probe import run_probe as run_probe_impl
        import asyncio
        
        report = asyncio.run(run_probe_impl(
            connection_info=connection_info,
            model_name=args.model_name,
            run_id=run_id,
            output_path=output_path,
            profile=args.profile,
            global_timeout=args.global_timeout,
        ))
        
        # Print full summary
        summary_path = output_path / "probe_summary.txt"
        if summary_path.exists():
            print("\n" + "="*60)
            with open(summary_path, "r") as f:
                print(f.read())
            print("="*60)
        
        print(f"\nFull report: {output_path / 'probe_report.json'}")
        print(f"Summary: {output_path / 'probe_summary.txt'}")
        
        # Return appropriate exit code
        if report['status'] == "failed":
            return 1
        elif report['status'] == "degraded":
            return 0  # Degraded is still success (warnings only)
        else:
            return 0
    
    except Exception as e:
        logger.error(f"Probe failed: {e}", exc_info=True)
        return 1


def _build_connection_info(args: argparse.Namespace) -> Dict[str, Any]:
    """Build connection_info dict from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        connection_info dict
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Determine provider type
    provider_type = args.provider
    
    # If no provider specified, try to infer from base_url or model_name
    if not provider_type:
        if args.base_url:
            if "localhost" in args.base_url or "127.0.0.1" in args.base_url:
                provider_type = "vllm"
            elif "openai.com" in args.base_url:
                provider_type = "openai"
            elif "anthropic.com" in args.base_url:
                provider_type = "anthropic"
            elif "together.xyz" in args.base_url:
                provider_type = "together"
        
        if not provider_type:
            # Try to infer from model name
            model_lower = args.model_name.lower()
            if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
                provider_type = "openai"
            elif "claude" in model_lower:
                provider_type = "anthropic"
            else:
                # Default to vllm for local models
                provider_type = "vllm"
    
    logger.info(f"Using provider: {provider_type}")
    
    # Get provider defaults
    provider_defaults = CLI_PROVIDER_DEFAULTS.get(provider_type, {})
    
    # Build connection_info
    connection_info = {
        "provider_type": provider_type,
        "base_url": args.base_url or provider_defaults.get("base_url"),
        "timeout": provider_defaults.get("timeout", 120),
        "max_retries": provider_defaults.get("max_retries", 3),
        "temperature": provider_defaults.get("temperature", 0.0),
        "max_tokens": provider_defaults.get("max_tokens", 2048),
        "max_tokens_param_name": provider_defaults.get("max_tokens_param_name", "max_tokens"),
        "api_endpoint": "auto",
    }
    
    # Add API key if provided
    if args.api_key:
        connection_info["api_key"] = args.api_key
    elif args.api_key_env:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {args.api_key_env} not set")
        connection_info["api_key"] = api_key
    elif provider_defaults.get("api_key_env"):
        # Try default env var for provider
        api_key = os.getenv(provider_defaults["api_key_env"])
        if api_key:
            connection_info["api_key"] = api_key
    
    # Add image_max_edge if provided
    if args.image_max_edge:
        connection_info["image_max_edge"] = args.image_max_edge
    
    # Set capabilities for OpenAI-compatible providers
    if provider_type in {"vllm", "openai", "anthropic", "together"}:
        connection_info["capabilities"] = {
            "request_modes": ["chat", "completions"],
            "supports_schema": True,
            "supports_logprobs": True,
        }
    
    # Validate required fields
    if not connection_info.get("base_url"):
        raise ValueError("--base-url is required (or use --provider with known defaults)")
    
    return connection_info
