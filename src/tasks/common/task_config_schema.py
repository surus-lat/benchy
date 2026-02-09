"""Task configuration schemas and validation.

This module defines the schemas for different task types and provides
validation and default application functions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Task type schemas defining requirements and defaults
TASK_TYPE_SCHEMAS = {
    "classification": {
        "handler_module": "src.tasks.common.multiple_choice",
        "handler_class": "MultipleChoiceHandler",
        "required_fields": ["input_field", "output_field"],
        "optional_fields": [
            "label_field",
            "labels",
            "choices_field",
            "multimodal_input",
            "multimodal_image_field",
        ],
        "requires_one_of": [
            ["labels", "choices_field"]
        ],  # Must have labels OR per-sample choices
        "default_field_mappings": {
            "input_field": "text",
            "output_field": "label",
            "label_field": "label",
        },
        "default_metrics": ["MultipleChoiceAccuracy"],
        "description": "Binary or multi-class classification tasks",
        "answer_type": "multiple_choice",
    },
    "structured": {
        "handler_module": "src.tasks.common.structured",
        "handler_class": "StructuredHandler",
        "required_fields": ["input_field", "output_field"],
        "requires_one_of": [
            ["schema_field", "schema_path", "schema_json"]
        ],  # Must have schema from somewhere
        "optional_fields": [
            "schema_field",
            "schema_path",
            "schema_json",
            "multimodal_input",
            "multimodal_image_field",
        ],
        "default_field_mappings": {
            "input_field": "text",
            "output_field": "expected",
            "schema_field": "schema",
        },
        "default_metrics": ["MetricsCalculator"],
        "description": "Structured data extraction with JSON schema",
        "answer_type": "structured",
    },
    "freeform": {
        "handler_module": "src.tasks.common.freeform",
        "handler_class": "FreeformHandler",
        "required_fields": ["input_field", "output_field"],
        "optional_fields": ["multimodal_input", "multimodal_image_field"],
        "default_field_mappings": {
            "input_field": "text",
            "output_field": "expected",
        },
        "default_metrics": ["ExactMatch", "F1Score"],
        "description": "Open-ended text generation tasks",
        "answer_type": "freeform",
    },
}


def validate_task_config(
    task_type: str, config: Dict[str, Any]
) -> List[str]:
    """Validate task configuration against schema.
    
    Args:
        task_type: Type of task (classification, structured, freeform)
        config: Task configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check if task type is valid
    if task_type not in TASK_TYPE_SCHEMAS:
        valid_types = ", ".join(TASK_TYPE_SCHEMAS.keys())
        errors.append(
            f"Invalid task type '{task_type}'. Must be one of: {valid_types}"
        )
        return errors
    
    schema = TASK_TYPE_SCHEMAS[task_type]
    dataset_config = config.get("dataset", {})
    
    # Check required fields
    for field in schema.get("required_fields", []):
        if field not in dataset_config:
            errors.append(
                f"Task type '{task_type}' requires field '{field}' in dataset config"
            )
    
    # Check "requires_one_of" constraints
    for field_group in schema.get("requires_one_of", []):
        has_any = any(field in dataset_config or field in config for field in field_group)
        if not has_any:
            field_list = "', '".join(field_group)
            errors.append(
                f"Task type '{task_type}' requires at least one of: '{field_list}'"
            )
    
    # Validate dataset config if present
    if "dataset" in config:
        from .dataset_adapters import validate_dataset_config
        dataset_errors = validate_dataset_config(dataset_config)
        errors.extend(dataset_errors)
    
    # Task-specific validation
    if task_type == "classification":
        # Check that labels is valid if provided
        if "labels" in dataset_config:
            labels = dataset_config["labels"]
            if isinstance(labels, str):
                try:
                    import json
                    labels = json.loads(labels)
                except json.JSONDecodeError:
                    errors.append("Invalid labels JSON string")
            if not isinstance(labels, dict):
                errors.append("Labels must be a dict mapping label values to text")
    
    elif task_type == "structured":
        # Check that schema is provided in some form
        has_schema = any(
            key in dataset_config or key in config
            for key in ["schema_field", "schema_path", "schema_json"]
        )
        if not has_schema:
            errors.append(
                "Structured tasks require schema via schema_field, schema_path, or schema_json"
            )
    
    return errors


def apply_defaults(task_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default field mappings and settings for task type.
    
    Args:
        task_type: Type of task (classification, structured, freeform)
        config: Task configuration (will be modified in place)
        
    Returns:
        Updated config with defaults applied
    """
    if task_type not in TASK_TYPE_SCHEMAS:
        logger.warning(f"Unknown task type '{task_type}', not applying defaults")
        return config
    
    schema = TASK_TYPE_SCHEMAS[task_type]
    
    # Ensure dataset config exists
    if "dataset" not in config:
        config["dataset"] = {}
    
    dataset_config = config["dataset"]
    
    # Apply default field mappings
    for field, default_value in schema.get("default_field_mappings", {}).items():
        if field not in dataset_config:
            dataset_config[field] = default_value
    
    # Set answer type if not present
    if "answer_type" not in config:
        config["answer_type"] = schema.get("answer_type", "freeform")
    
    return config


def get_handler_class(task_type: str):
    """Get the handler class for a task type.
    
    Args:
        task_type: Type of task (classification, structured, freeform)
        
    Returns:
        Handler class
        
    Raises:
        ValueError: If task type is invalid
    """
    if task_type not in TASK_TYPE_SCHEMAS:
        valid_types = ", ".join(TASK_TYPE_SCHEMAS.keys())
        raise ValueError(
            f"Invalid task type '{task_type}'. Must be one of: {valid_types}"
        )
    
    schema = TASK_TYPE_SCHEMAS[task_type]
    module_path = schema["handler_module"]
    class_name = schema["handler_class"]
    
    # Import the module and get the class
    import importlib
    module = importlib.import_module(module_path)
    handler_class = getattr(module, class_name)
    
    return handler_class


def build_task_metadata(task_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build task metadata for registry.
    
    Args:
        task_type: Type of task (classification, structured, freeform)
        config: Task configuration
        
    Returns:
        Task metadata dict suitable for registry
    """
    schema = TASK_TYPE_SCHEMAS.get(task_type, {})
    
    # Extract dataset name for display
    dataset_name = config.get("dataset", {}).get("name", "custom-dataset")
    
    metadata = {
        "name": f"_adhoc_{task_type}",
        "display_name": f"Ad-hoc {task_type.title()} Task",
        "description": f"{schema.get('description', task_type)} - {dataset_name}",
        "capability_requirements": {
            "requires_logprobs": "optional",
            "requires_multimodal": "optional" if config.get("dataset", {}).get("multimodal_input") else "no",
            "requires_schema": "preferred" if task_type == "structured" else "optional",
            "requires_files": "optional",
            "requires_streaming": "optional",
        },
        "defaults": config.get("defaults", {}),
        "prompts": {},
        "task_configs": {},
        "output": {"subdirectory": f"_adhoc_{task_type}"},
        "metrics_manifest": schema.get("default_metrics", []),
    }
    
    # Add prompts if specified
    if "system_prompt" in config:
        metadata["prompts"]["system_prompt"] = config["system_prompt"]
    if "user_prompt_template" in config:
        metadata["prompts"]["user_prompt_template"] = config["user_prompt_template"]
    
    return metadata


def get_task_type_description(task_type: str) -> str:
    """Get human-readable description of task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Description string
    """
    schema = TASK_TYPE_SCHEMAS.get(task_type)
    if not schema:
        return f"Unknown task type: {task_type}"
    
    return schema.get("description", task_type)


def list_task_types() -> List[str]:
    """Get list of available task types.
    
    Returns:
        List of task type names
    """
    return list(TASK_TYPE_SCHEMAS.keys())


def get_required_fields(task_type: str) -> List[str]:
    """Get required fields for a task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        List of required field names
    """
    schema = TASK_TYPE_SCHEMAS.get(task_type, {})
    return schema.get("required_fields", [])


def get_optional_fields(task_type: str) -> List[str]:
    """Get optional fields for a task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        List of optional field names
    """
    schema = TASK_TYPE_SCHEMAS.get(task_type, {})
    return schema.get("optional_fields", [])
