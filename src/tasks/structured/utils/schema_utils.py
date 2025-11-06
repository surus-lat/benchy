"""Utilities for sanitizing JSON schemas for vLLM compatibility."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def sanitize_schema_for_vllm(schema: Dict) -> Dict:
    """Sanitize a JSON schema to be compatible with vLLM's guided_json.
    
    vLLM uses grammar-based structured generation which doesn't support all
    JSON Schema features. This function removes or fixes unsupported features.
    
    Args:
        schema: Original JSON schema
        
    Returns:
        Sanitized schema compatible with vLLM
    """
    import copy
    schema = copy.deepcopy(schema)
    
    # Remove unsupported top-level keys
    unsupported_keys = ["$schema", "$id", "description", "title"]
    for key in unsupported_keys:
        schema.pop(key, None)
    
    # Remove allOf, oneOf, anyOf at top level - vLLM has limited support for these
    # We validate with the full schema before sanitization, so removing these is safe
    # for generation purposes
    for combiner in ["allOf", "oneOf", "anyOf"]:
        if combiner in schema:
            logger.debug(f"Removing top-level {combiner} for vLLM compatibility")
            schema.pop(combiner, None)
    
    # Recursively sanitize the schema
    _sanitize_recursive(schema)
    
    return schema


def _sanitize_recursive(obj: Any) -> None:
    """Recursively sanitize a schema object in-place.
    
    Args:
        obj: Schema object or sub-object to sanitize
    """
    if not isinstance(obj, dict):
        return
    
    # Remove unsupported constraints
    unsupported_constraints = [
        "uniqueItems",  # Not supported by vLLM grammar
        "minItems",     # Often causes issues
        "maxItems",     # Often causes issues
        "minLength",    # Can cause issues
        "maxLength",    # Can cause issues
        "minimum",      # Can cause issues
        "maximum",      # Can cause issues
        "pattern",      # Not supported
        "format",       # Not supported (date, email, uuid, phone, etc.)
        "contentMediaType",  # Not supported
        "contentEncoding",   # Not supported
        "examples",     # Not needed for generation
        "default",      # Can cause issues
        "const",        # Can cause issues
    ]
    
    for key in unsupported_constraints:
        obj.pop(key, None)
    
    # Fix invalid "required" inside properties
    # JSON Schema requires "required" to be an array at the object level,
    # not a boolean inside property definitions
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_name, prop_value in obj["properties"].items():
            if isinstance(prop_value, dict):
                # Remove invalid "required": true/false from property
                prop_value.pop("required", None)
                # Recursively sanitize nested properties
                _sanitize_recursive(prop_value)
    
    # Recursively sanitize $defs
    if "$defs" in obj and isinstance(obj["$defs"], dict):
        for def_name, def_value in obj["$defs"].items():
            _sanitize_recursive(def_value)
    
    # Recursively sanitize definitions (alternative to $defs)
    if "definitions" in obj and isinstance(obj["definitions"], dict):
        for def_name, def_value in obj["definitions"].items():
            _sanitize_recursive(def_value)
    
    # Recursively sanitize items (for arrays)
    if "items" in obj:
        _sanitize_recursive(obj["items"])
    
    # Recursively sanitize additionalProperties
    if "additionalProperties" in obj and isinstance(obj["additionalProperties"], dict):
        _sanitize_recursive(obj["additionalProperties"])
    
    # Remove anyOf, oneOf, allOf - vLLM has limited support for these combiners
    # We validate with the full schema before sanitization, so removing these is safe
    # for generation purposes
    for combiner in ["anyOf", "oneOf", "allOf"]:
        if combiner in obj:
            logger.debug(f"Removing {combiner} for vLLM compatibility")
            obj.pop(combiner, None)

