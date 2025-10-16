"""Centralized run ID management for Benchy."""

import os
from datetime import datetime
from typing import Optional


def generate_run_id(
    custom_run_id: Optional[str] = None,
    is_test: bool = False,
    is_limited: bool = False,
    prefix: str = ""
) -> str:
    """
    Generate a unified run ID for Benchy runs.
    
    Args:
        custom_run_id: Custom run ID provided by user
        is_test: Whether this is a test run (adds _TEST suffix)
        is_limited: Whether this is a limited run (adds _LIMITED suffix)
        prefix: Optional prefix for the run ID
        
    Returns:
        Generated run ID string
    """
    if custom_run_id:
        base_run_id = custom_run_id
    else:
        # Generate timestamp-based run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_id = f"{prefix}{timestamp}" if prefix else timestamp
    
    # Add flags as suffixes
    flags = []
    if is_test:
        flags.append("TEST")
    if is_limited:
        flags.append("LIMITED")
    
    if flags:
        run_id = f"{base_run_id}_{'_'.join(flags)}"
    else:
        run_id = base_run_id
    
    return run_id


def get_run_paths(run_id: str, base_output_path: str = "outputs/benchmark_outputs", base_log_path: str = "logs") -> dict:
    """
    Get standardized paths for a run ID.
    
    Args:
        run_id: The run ID
        base_output_path: Base path for benchmark outputs
        base_log_path: Base path for logs
        
    Returns:
        Dictionary with standardized paths
    """
    return {
        "run_id": run_id,
        "output_path": f"{base_output_path}/{run_id}",
        "log_path": f"{base_log_path}/{run_id}",
        "model_output_path": lambda model_name: f"{base_output_path}/{run_id}/{model_name.split('/')[-1]}",
        "model_log_path": lambda model_name: f"{base_log_path}/{run_id}/{model_name.split('/')[-1].replace('/', '_')}"
    }


def setup_run_directories(run_paths: dict) -> None:
    """
    Create necessary directories for a run.
    
    Args:
        run_paths: Dictionary from get_run_paths()
    """
    os.makedirs(run_paths["output_path"], exist_ok=True)
    os.makedirs(run_paths["log_path"], exist_ok=True)


def get_prefect_flow_name(base_name: str, run_id: str) -> str:
    """
    Generate Prefect flow name with run ID prefix.
    
    Args:
        base_name: Base flow name (e.g., "benchmark_pipeline")
        run_id: Run ID to use as prefix
        
    Returns:
        Prefect flow name with run ID prefix
    """
    return f"{run_id}_{base_name}"
