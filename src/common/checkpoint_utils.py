"""Checkpointing utilities for resumable benchmarks."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Set, List

logger = logging.getLogger(__name__)


def get_checkpoint_path(output_dir: str, model_name: str, task_name: str) -> Path:
    """Get checkpoint file path for a benchmark run.
    
    Args:
        output_dir: Base output directory for results
        model_name: Name of the model being evaluated
        task_name: Name of the task being run
        
    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = Path(output_dir) / ".checkpoints"
    safe_model_name = model_name.replace('/', '_')
    return checkpoint_dir / f"{safe_model_name}_{task_name}_checkpoint.json"


def get_config_hash(config_dict: Dict[str, Any]) -> str:
    """Generate hash of configuration for checkpoint validation.
    
    Args:
        config_dict: Configuration dictionary with relevant parameters
        
    Returns:
        MD5 hash of the configuration
    """
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def save_checkpoint(
    path: Path,
    completed_ids: List[str],
    config_hash: str
) -> None:
    """Save checkpoint to disk.
    
    Args:
        path: Path to checkpoint file
        completed_ids: List of completed sample IDs
        config_hash: Hash of the configuration for validation
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "completed_sample_ids": completed_ids,
        "config_hash": config_hash,
        "timestamp": datetime.now().isoformat(),
        "count": len(completed_ids),
    }
    with open(path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(path: Path, expected_config_hash: str) -> Set[str]:
    """Load and validate checkpoint from disk.
    
    Args:
        path: Path to checkpoint file
        expected_config_hash: Expected configuration hash for validation
        
    Returns:
        Set of completed sample IDs, or empty set if invalid/missing
    """
    if not path.exists():
        return set()
    
    with open(path) as f:
        checkpoint = json.load(f)
    
    if checkpoint.get("config_hash") != expected_config_hash:
        logger.warning("Checkpoint found but config changed - ignoring")
        return set()
    
    completed_ids = set(checkpoint.get("completed_sample_ids", []))
    logger.info(f"✓ Loaded checkpoint: {len(completed_ids)} samples completed")
    return completed_ids

