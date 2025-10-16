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


def format_generation_params_for_lm_eval(
    generation_config: Optional[Dict[str, Any]]
) -> str:
    """
    Format generation config parameters for lm-evaluation-harness model_args.
    
    Args:
        generation_config: The generation config dictionary
        
    Returns:
        Comma-separated string of generation parameters suitable for model_args
    """
    if not generation_config:
        return ""
    
    # Parameters that lm-eval supports via the API
    # Based on OpenAI API spec and lm-eval local-completions implementation
    supported_params = {
        'temperature': 'temperature',
        'top_p': 'top_p',
        'top_k': 'top_k',
        'repetition_penalty': 'repetition_penalty',
        'max_tokens': 'max_new_tokens',  # Map to max_new_tokens
        'max_new_tokens': 'max_new_tokens',
        'do_sample': 'do_sample',
        'seed': 'seed'
    }
    
    params = []
    for config_key, eval_key in supported_params.items():
        if config_key in generation_config:
            value = generation_config[config_key]
            # Convert booleans to Python-style strings for command line
            if isinstance(value, bool):
                value_str = str(value)
            else:
                value_str = str(value)
            params.append(f"{eval_key}={value_str}")
    
    param_string = ",".join(params)
    
    if param_string:
        logger.info(f"Formatted generation parameters for lm_eval: {param_string}")
    
    return param_string


