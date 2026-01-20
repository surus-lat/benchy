"""Utility for fetching and managing generation_config.json from HuggingFace models."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def fetch_generation_config(
    model_name: str,
    hf_cache: str = None,
    hf_token: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Fetch generation_config.json from a HuggingFace model repository.
    
    Args:
        model_name: The HuggingFace model name (e.g., "tencent/Hunyuan-MT-7B")
        hf_cache: HuggingFace cache directory
        hf_token: HuggingFace token for private models
        
    Returns:
        Dictionary containing generation config, or None if not found
    """
    try:
        model_path = Path(model_name)
        if model_path.exists():
            local_config = model_path / "generation_config.json"
            if not local_config.exists():
                logger.info("No generation_config.json found at %s", local_config)
                return None
            logger.info("Loading generation_config.json from local model path: %s", model_path)
            with open(local_config, "r") as f:
                generation_config = json.load(f)
            return generation_config

        logger.info(f"Fetching generation_config.json for {model_name}")

        # Download generation_config.json from HF Hub
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="generation_config.json",
            cache_dir=hf_cache,
            token=hf_token if hf_token else None
        )

        # Load and return the config
        with open(config_path, 'r') as f:
            generation_config = json.load(f)

        logger.info(f"Successfully loaded generation_config.json for {model_name}")
        logger.debug(f"Generation config: {generation_config}")

        return generation_config

    except Exception as e:
        logger.info(f"No generation_config.json found for {model_name}: {e}")
        return None


def save_generation_config(
    generation_config: Dict[str, Any],
    output_path: str,
    model_name: str
) -> None:
    """
    Save generation config to output directory for logging.
    
    Args:
        generation_config: The generation config dictionary
        output_path: Output directory path
        model_name: Model name for logging
    """
    if not generation_config:
        return
        
    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = output_dir / "generation_config.json"
        with open(config_file, 'w') as f:
            json.dump(generation_config, f, indent=2)
        
        logger.info(f"Saved generation_config.json to {config_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save generation_config.json: {e}")

