"""Configuration file generator from CLI arguments.

This module provides functionality to generate reusable YAML configuration
files from CLI parameters, enabling users to save their ad-hoc task setups
for future use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def generate_config_from_cli(
    args,  # argparse.Namespace
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate YAML config file from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        output_path: Path where to save the generated config
        config: Optional existing config to extend (for ad-hoc tasks)
    """
    # Start with existing config or create new one
    if config:
        generated_config = dict(config)
    else:
        generated_config = {}
    
    # Model information
    if not generated_config.get("model"):
        generated_config["model"] = {}
    
    if args.model_name:
        generated_config["model"]["name"] = args.model_name
    if args.model_path:
        generated_config["model"]["path"] = args.model_path
    
    # Provider type
    if args.provider:
        generated_config["provider_type"] = args.provider
    elif not generated_config.get("provider_type"):
        generated_config["provider_type"] = "vllm"
    
    # Provider configuration
    provider_type = generated_config.get("provider_type", "vllm")
    if args.base_url and provider_type in ["openai", "together", "anthropic", "alibaba", "google"]:
        if provider_type not in generated_config:
            generated_config[provider_type] = {}
        generated_config[provider_type]["base_url"] = args.base_url
    
    # Task configuration
    if args.task_type:
        # Ad-hoc task configuration
        task_name = f"_adhoc_{args.task_type}"
        
        if "task_configs" not in generated_config:
            generated_config["task_configs"] = {}
        
        # Build dataset config
        dataset_config = _build_dataset_config_from_args(args)
        
        task_config = {
            "dataset": dataset_config,
        }
        
        # Add prompts if specified
        if args.system_prompt:
            task_config["system_prompt"] = args.system_prompt
        if args.user_prompt_template:
            task_config["user_prompt_template"] = args.user_prompt_template
        
        generated_config["task_configs"][task_name] = task_config
        
        # Set tasks list
        generated_config["tasks"] = [task_name]
    
    elif args.dataset_name or args.dataset:
        # Dataset override for existing tasks
        dataset_config = _build_dataset_config_from_args(args)
        
        if "task_defaults" not in generated_config:
            generated_config["task_defaults"] = {}
        
        generated_config["task_defaults"]["dataset"] = dataset_config
        
        # Include tasks if specified
        if args.tasks:
            generated_config["tasks"] = args.tasks
    
    # Task defaults
    if args.batch_size:
        if "task_defaults" not in generated_config:
            generated_config["task_defaults"] = {}
        generated_config["task_defaults"]["batch_size"] = args.batch_size
    
    if args.log_samples:
        if "task_defaults" not in generated_config:
            generated_config["task_defaults"] = {}
        generated_config["task_defaults"]["log_samples"] = True
    elif args.no_log_samples:
        if "task_defaults" not in generated_config:
            generated_config["task_defaults"] = {}
        generated_config["task_defaults"]["log_samples"] = False
    
    # Clean up empty sections
    generated_config = _cleanup_empty_sections(generated_config)
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(
            generated_config,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        )
    
    logger.info(f"Generated config saved to: {output_file}")
    print(f"✓ Configuration saved to: {output_file}")
    print(f"  You can now run: benchy eval --config {output_file}")


def _build_dataset_config_from_args(args) -> Dict[str, Any]:
    """Build dataset configuration from CLI arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Dataset configuration dict
    """
    dataset_config = {}
    
    # Basic dataset info
    if hasattr(args, 'dataset_name') and args.dataset_name:
        dataset_config["name"] = args.dataset_name
    elif hasattr(args, 'dataset') and args.dataset:
        dataset_config["name"] = args.dataset
    
    if hasattr(args, 'dataset_source') and args.dataset_source and args.dataset_source != "auto":
        dataset_config["source"] = args.dataset_source
    if hasattr(args, 'dataset_split') and args.dataset_split and args.dataset_split != "test":
        dataset_config["split"] = args.dataset_split
    
    # Field mappings (only include if non-default)
    if hasattr(args, 'dataset_input_field') and args.dataset_input_field:
        dataset_config["input_field"] = args.dataset_input_field
    if hasattr(args, 'dataset_output_field') and args.dataset_output_field:
        dataset_config["output_field"] = args.dataset_output_field
    if hasattr(args, 'dataset_id_field') and args.dataset_id_field:
        dataset_config["id_field"] = args.dataset_id_field
    
    # Classification-specific
    if hasattr(args, 'dataset_label_field') and args.dataset_label_field:
        dataset_config["label_field"] = args.dataset_label_field
    if hasattr(args, 'dataset_labels') and args.dataset_labels:
        dataset_config["labels"] = args.dataset_labels
    if hasattr(args, 'dataset_choices_field') and args.dataset_choices_field:
        dataset_config["choices_field"] = args.dataset_choices_field
    
    # Structured extraction specific
    if hasattr(args, 'dataset_schema_field') and args.dataset_schema_field:
        dataset_config["schema_field"] = args.dataset_schema_field
    if hasattr(args, 'dataset_schema_path') and args.dataset_schema_path:
        dataset_config["schema_path"] = args.dataset_schema_path
    if hasattr(args, 'dataset_schema_json') and args.dataset_schema_json:
        dataset_config["schema_json"] = args.dataset_schema_json
    
    # Multimodal support
    if hasattr(args, 'multimodal_input') and args.multimodal_input:
        dataset_config["multimodal_input"] = True
    if hasattr(args, 'multimodal_image_field') and args.multimodal_image_field and args.multimodal_image_field != "image_path":
        dataset_config["multimodal_image_field"] = args.multimodal_image_field
    
    return dataset_config


def _cleanup_empty_sections(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty sections from config.
    
    Args:
        config: Configuration dict
        
    Returns:
        Cleaned config dict
    """
    cleaned = {}
    for key, value in config.items():
        if isinstance(value, dict):
            cleaned_value = _cleanup_empty_sections(value)
            if cleaned_value:  # Only include non-empty dicts
                cleaned[key] = cleaned_value
        elif value is not None and value != {}:
            cleaned[key] = value
    return cleaned


def validate_generated_config(config: Dict[str, Any]) -> list[str]:
    """Validate generated configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    if "model" not in config or not config["model"].get("name"):
        errors.append("Missing required field: model.name")
    
    if "tasks" not in config or not config["tasks"]:
        errors.append("Missing required field: tasks (list of tasks to run)")
    
    return errors
