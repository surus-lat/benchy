"""Utilities for sanitizing JSON schemas for API compatibility."""

import copy
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def sanitize_schema_for_openai_strict(schema: Dict) -> Dict:
    """
    Produce a JSON Schema compatible with OpenAI strict mode.
    
    Removes unsupported top-level keys and combiners, enforces object-level additionalProperties=false, and promotes object properties to required so the resulting schema meets OpenAI strict mode constraints.
    
    Parameters:
        schema (Dict): The input JSON Schema as a mapping.
    
    Returns:
        Dict: A sanitized JSON Schema compatible with OpenAI strict mode.
    """
    schema = copy.deepcopy(schema)
    
    # Remove unsupported top-level keys
    unsupported_keys = ["$schema", "$id", "description", "title"]
    for key in unsupported_keys:
        schema.pop(key, None)
    
    # Remove combiners at top level
    for combiner in ["allOf", "oneOf", "anyOf"]:
        if combiner in schema:
            logger.debug(f"Removing top-level {combiner} for OpenAI strict mode compatibility")
            schema.pop(combiner, None)
    
    # Recursively sanitize the schema
    _sanitize_recursive_strict(schema)
    
    return schema


def _sanitize_recursive_strict(obj: Any) -> None:
    """
    Sanitize a JSON Schema object in-place to conform to OpenAI strict mode.
    
    Removes constraints and keywords unsupported by OpenAI strict mode, strips schema combiners (anyOf, oneOf, allOf), ensures object schemas set `additionalProperties` to False, and promotes all declared properties to required at the object level. Operates recursively on nested schemas and mutates `obj` in place.
    
    Parameters:
        obj (Any): Schema object or sub-object to sanitize (modified in place).
    """
    if not isinstance(obj, dict):
        return
    
    # Remove unsupported constraints
    unsupported_constraints = [
        "uniqueItems",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "pattern",
        "format",  # date, email, uuid, phone, etc.
        "contentMediaType",
        "contentEncoding",
        "examples",
        "default",
        "const",
        "title",
        "description",
    ]
    
    for key in unsupported_constraints:
        obj.pop(key, None)
    
    # For objects, ensure additionalProperties is false
    if obj.get("type") == "object":
        if "additionalProperties" not in obj:
            obj["additionalProperties"] = False
        elif obj["additionalProperties"] is True:
            obj["additionalProperties"] = False
        
        # Make all properties required for strict mode
        if "properties" in obj and isinstance(obj["properties"], dict):
            if "required" not in obj:
                obj["required"] = list(obj["properties"].keys())
    
    # Fix invalid "required" inside properties
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_name, prop_value in obj["properties"].items():
            if isinstance(prop_value, dict):
                # Remove invalid "required": true/false from property
                prop_value.pop("required", None)
                # Recursively sanitize nested properties
                _sanitize_recursive_strict(prop_value)
    
    # Recursively sanitize $defs
    if "$defs" in obj and isinstance(obj["$defs"], dict):
        for def_name, def_value in obj["$defs"].items():
            _sanitize_recursive_strict(def_value)
    
    # Recursively sanitize definitions (alternative to $defs)
    if "definitions" in obj and isinstance(obj["definitions"], dict):
        for def_name, def_value in obj["definitions"].items():
            _sanitize_recursive_strict(def_value)
    
    # Recursively sanitize items (for arrays)
    if "items" in obj:
        _sanitize_recursive_strict(obj["items"])
    
    # Recursively sanitize additionalProperties if it's a schema
    if "additionalProperties" in obj and isinstance(obj["additionalProperties"], dict):
        _sanitize_recursive_strict(obj["additionalProperties"])
    
    # Remove anyOf, oneOf, allOf combiners
    for combiner in ["anyOf", "oneOf", "allOf"]:
        if combiner in obj:
            logger.debug(f"Removing {combiner} for OpenAI strict mode compatibility")
            obj.pop(combiner, None)


def sanitize_schema_for_vllm(schema: Dict) -> Dict:
    """
    Sanitize a JSON Schema for compatibility with vLLM's structured_outputs.
    
    Removes top-level metadata keys and combiners (allOf, oneOf, anyOf), and recursively strips constraints and annotations that vLLM's grammar-based structured generation does not support so the returned schema can be used with vLLM structured outputs.
    
    Parameters:
        schema (Dict): The original JSON Schema to sanitize.
    
    Returns:
        Dict: A sanitized copy of the schema compatible with vLLM structured_outputs.
    """
    schema = copy.deepcopy(schema)
    
    # Remove unsupported top-level keys
    unsupported_keys = ["$schema", "$id", "description", "title"]
    for key in unsupported_keys:
        schema.pop(key, None)
    
    # Remove combiners at top level
    for combiner in ["allOf", "oneOf", "anyOf"]:
        if combiner in schema:
            logger.debug(f"Removing top-level {combiner} for vLLM compatibility")
            schema.pop(combiner, None)
    
    # Recursively sanitize the schema
    _sanitize_recursive_vllm(schema)
    
    return schema


def _sanitize_recursive_vllm(obj: Any) -> None:
    """
    Sanitize a JSON Schema subtree in-place for vLLM structured output compatibility.
    
    Removes constraints unsupported by vLLM, strips any per-property "required" entries, recursively processes nested schema containers ($defs, definitions, items, additionalProperties), and removes combiners (`anyOf`, `oneOf`, `allOf`) while logging their removal. If `obj` is not a dict, the function is a no-op.
    
    Parameters:
        obj (Any): Schema object or sub-object to sanitize in-place; expected to be a dict for any changes to occur.
    """
    if not isinstance(obj, dict):
        return
    
    # Remove unsupported constraints (less restrictive than OpenAI strict mode)
    unsupported_constraints = [
        "uniqueItems",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "pattern",
        "format",
        "contentMediaType",
        "contentEncoding",
        "examples",
        "default",
        "const",
    ]
    
    for key in unsupported_constraints:
        obj.pop(key, None)
    
    # Fix invalid "required" inside properties
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_name, prop_value in obj["properties"].items():
            if isinstance(prop_value, dict):
                prop_value.pop("required", None)
                _sanitize_recursive_vllm(prop_value)
    
    # Recursively sanitize $defs
    if "$defs" in obj and isinstance(obj["$defs"], dict):
        for def_name, def_value in obj["$defs"].items():
            _sanitize_recursive_vllm(def_value)
    
    # Recursively sanitize definitions
    if "definitions" in obj and isinstance(obj["definitions"], dict):
        for def_name, def_value in obj["definitions"].items():
            _sanitize_recursive_vllm(def_value)
    
    # Recursively sanitize items
    if "items" in obj:
        _sanitize_recursive_vllm(obj["items"])
    
    # Recursively sanitize additionalProperties
    if "additionalProperties" in obj and isinstance(obj["additionalProperties"], dict):
        _sanitize_recursive_vllm(obj["additionalProperties"])
    
    # Remove combiners
    for combiner in ["anyOf", "oneOf", "allOf"]:
        if combiner in obj:
            logger.debug(f"Removing {combiner} for vLLM compatibility")
            obj.pop(combiner, None)
