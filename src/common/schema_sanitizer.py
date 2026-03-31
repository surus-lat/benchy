"""Utilities for sanitizing JSON schemas for API compatibility."""

import copy
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _is_valid_required_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _resolve_ref(root: Dict, ref: str) -> Any:
    """Resolve a JSON pointer ``$ref`` against *root*.

    Only handles internal references of the form ``#/a/b/c``.
    Returns ``None`` if the path cannot be resolved.
    """
    if not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    node: Any = root
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _resolve_inline_refs(obj: Any, root: Dict) -> None:
    """Recursively replace ``$ref`` pointers with the referenced sub-schema.

    Only resolves refs that point *inside* the schema itself (``#/...``).
    Top-level ``$defs``/``definitions`` refs are left alone as OpenAI
    supports those.
    """
    if not isinstance(obj, dict):
        return

    props = obj.get("properties")
    if isinstance(props, dict):
        for key in list(props.keys()):
            val = props[key]
            if isinstance(val, dict) and "$ref" in val:
                ref = val["$ref"]
                # Only resolve non-$defs refs
                if ref.startswith("#/") and not (
                    ref.startswith("#/$defs/") or ref.startswith("#/definitions/")
                ):
                    resolved = _resolve_ref(root, ref)
                    if resolved is not None:
                        props[key] = copy.deepcopy(resolved)
            # Recurse
            _resolve_inline_refs(props[key], root)

    # Also recurse into items, $defs, definitions, etc.
    for sub_key in ("items", "additionalProperties"):
        sub = obj.get(sub_key)
        if isinstance(sub, dict):
            _resolve_inline_refs(sub, root)
    for defs_key in ("$defs", "definitions"):
        defs = obj.get(defs_key)
        if isinstance(defs, dict):
            for d in defs.values():
                _resolve_inline_refs(d, root)


def unwrap_openai_response_format_schema(schema: Dict) -> Dict:
    """Unwrap an OpenAI-style response_format json_schema wrapper into a plain JSON Schema.

    Some datasets store schemas in the OpenAI `response_format` shape:
      {"type":"json_schema","json_schema":{"name":...,"strict":...,"schema":{...}}}

    The benchmark tasks/interfaces expect a plain JSON Schema (the inner `schema`).
    """
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") != "json_schema":
        return schema
    inner = schema.get("json_schema")
    if not isinstance(inner, dict):
        return schema
    inner_schema = inner.get("schema")
    return inner_schema if isinstance(inner_schema, dict) else schema


def sanitize_schema_for_openai_strict(schema: Dict) -> Dict:
    """Sanitize a JSON schema to be compatible with OpenAI's strict mode.
    
    OpenAI's strict mode has specific requirements:
    - No $schema, $id, description, title at top level
    - No format constraints (date, email, uuid, etc.)
    - No pattern, minLength, maxLength, minimum, maximum
    - No allOf, oneOf, anyOf combiners
    - Must have additionalProperties: false for objects
    - All properties at each level must be required
    
    Args:
        schema: Original JSON schema
        
    Returns:
        Sanitized schema compatible with OpenAI strict mode
    """
    schema = unwrap_openai_response_format_schema(copy.deepcopy(schema))
    
    # Remove unsupported top-level keys
    unsupported_keys = ["$schema", "$id", "description", "title"]
    for key in unsupported_keys:
        schema.pop(key, None)
    
    # Remove combiners at top level
    for combiner in ["allOf", "oneOf", "anyOf"]:
        if combiner in schema:
            logger.debug(f"Removing top-level {combiner} for OpenAI strict mode compatibility")
            schema.pop(combiner, None)
    
    # Resolve inline $ref pointers that aren't top-level $defs
    # (OpenAI strict mode only allows $ref to top-level definitions)
    _resolve_inline_refs(schema, schema)

    # Recursively sanitize the schema
    _sanitize_recursive_strict(schema)

    return schema


def _sanitize_recursive_strict(obj: Any) -> None:
    """Recursively sanitize a schema object in-place for OpenAI strict mode.
    
    Args:
        obj: Schema object or sub-object to sanitize
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
            obj["required"] = list(obj["properties"].keys())
    
    # Fix invalid "required" inside properties
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_name, prop_value in obj["properties"].items():
            if isinstance(prop_value, dict):
                # Some schemas incorrectly place `required: true/false` inside a property schema.
                # Keep legitimate object-level `required: [..]` lists for nested objects.
                if "required" in prop_value and not _is_valid_required_list(prop_value["required"]):
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
    """Sanitize a JSON schema to be compatible with vLLM's structured_outputs.
    
    vLLM uses grammar-based structured generation which doesn't support all
    JSON Schema features. This is more permissive than OpenAI strict mode.
    
    Args:
        schema: Original JSON schema
        
    Returns:
        Sanitized schema compatible with vLLM
    """
    schema = unwrap_openai_response_format_schema(copy.deepcopy(schema))
    
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
    """Recursively sanitize a schema object in-place for vLLM.
    
    Args:
        obj: Schema object or sub-object to sanitize
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
                if "required" in prop_value and not _is_valid_required_list(prop_value["required"]):
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
