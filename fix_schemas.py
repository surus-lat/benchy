#!/usr/bin/env python3
"""Fix JSON schemas in paraloq_data.jsonl to have root type/properties.

Rewrites schemas that only have $defs to include explicit root structure,
inferred from the expected output. Validates all new schemas against expected.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy

try:
    from jsonschema import Draft202012Validator, validate, ValidationError
    from jsonschema.exceptions import SchemaError
except ImportError:
    print("ERROR: jsonschema library not installed. Install with: pip install jsonschema")
    sys.exit(1)


def infer_root_structure(expected: Any, defs: Dict[str, Dict]) -> Optional[Dict]:
    """Infer root schema structure from expected output and definitions.
    
    Args:
        expected: Expected output value
        defs: Dictionary of $defs definitions
        
    Returns:
        Root schema structure or None if cannot infer
    """
    if expected is None:
        return None
    
    # Case 1: Expected is a dict with keys matching definition names
    if isinstance(expected, dict):
        # Check if keys match definition names
        matching_defs = [key for key in expected.keys() if key in defs]
        
        if matching_defs:
            # Create root object with properties for each matching definition
            properties = {}
            required = []
            
            for key in expected.keys():
                if key in defs:
                    # Value matches a definition - check if it's an array or direct
                    value = expected[key]
                    def_schema = defs[key]
                    
                    if isinstance(value, list):
                        # Array of definition items
                        if value and isinstance(value[0], dict):
                            # Check if array items match the definition structure
                            if _matches_definition_structure(value[0], def_schema, defs):
                                properties[key] = {
                                    "type": "array",
                                    "items": {"$ref": f"#/$defs/{key}"}
                                }
                            else:
                                # Array items don't match - infer from first item
                                properties[key] = {
                                    "type": "array",
                                    "items": _infer_type_schema(value[0])
                                }
                        else:
                            # Empty array or non-dict items
                            properties[key] = {
                                "type": "array",
                                "items": _infer_type_schema(value[0]) if value else {}
                            }
                    elif isinstance(value, dict):
                        # Direct object - check if it matches the definition
                        if _matches_definition_structure(value, def_schema, defs):
                            # Matches definition - use reference
                            properties[key] = {"$ref": f"#/$defs/{key}"}
                        else:
                            # Doesn't match - infer from value
                            properties[key] = _infer_type_schema(value)
                    else:
                        # Primitive value - infer type
                        properties[key] = _infer_type_schema(value)
                    
                    # Add to required if definition has required fields
                    if def_schema.get("required"):
                        required.append(key)
                else:
                    # Key not in definitions - infer type from value
                    properties[key] = _infer_type_schema(expected[key])
            
            result = {
                "type": "object",
                "properties": properties
            }
            if required:
                result["required"] = required
            return result
        
        # Case 2: Expected dict structure matches a definition directly
        # Check if expected matches any definition's structure (by content, not name)
        best_match = None
        best_match_score = 0
        
        for def_name, def_schema in defs.items():
            if _matches_definition_structure(expected, def_schema, defs):
                # Calculate match score (how many properties overlap)
                def_props = set(def_schema.get("properties", {}).keys())
                exp_keys = set(expected.keys())
                overlap = len(def_props & exp_keys)
                score = overlap / max(len(def_props), 1)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match = (def_name, def_schema)
        
        if best_match and best_match_score > 0.7:
            def_name, def_schema = best_match
            # Expected matches this definition - use definition as root
            # But ensure it has type/properties explicitly
            root_schema = deepcopy(def_schema)
            # Ensure type is set
            if "type" not in root_schema:
                root_schema["type"] = "object"
            return root_schema
        
        # Case 3: Expected is a plain object - infer from structure
        # This is a fallback - create properties from expected keys
        properties = {}
        for key, value in expected.items():
            properties[key] = _infer_type_schema(value)
        
        return {
            "type": "object",
            "properties": properties
        }
    
    # Case 4: Expected is an array
    if isinstance(expected, list):
        if not expected:
            # Empty array - can't infer item type
            return {
                "type": "array",
                "items": {}
            }
        
        # Check if array items match a definition
        first_item = expected[0]
        if isinstance(first_item, dict):
            for def_name, def_schema in defs.items():
                if _matches_definition_structure(first_item, def_schema, defs):
                    return {
                        "type": "array",
                        "items": {"$ref": f"#/$defs/{def_name}"}
                    }
        
        # Infer item type from first element
        return {
            "type": "array",
            "items": _infer_type_schema(first_item)
        }
    
    # Case 5: Expected is a primitive (string, number, etc.)
    return {
        "type": _get_json_type(expected)
    }


def _matches_definition_structure(value: Any, def_schema: Dict, all_defs: Dict) -> bool:
    """Check if a value matches a definition's structure.
    
    This is a heuristic check - not full validation.
    """
    if not isinstance(value, dict):
        return False
    
    # If definition is not an object type, can't match
    def_type = def_schema.get("type")
    if def_type != "object":
        return False
    
    def_props = def_schema.get("properties", {})
    def_required = def_schema.get("required", [])
    
    # Check if value has all required fields
    for req_field in def_required:
        if req_field not in value:
            return False
    
    # Check if value has mostly the same keys as definition
    value_keys = set(value.keys())
    def_keys = set(def_props.keys())
    
    # If value has significantly more keys than definition, probably not a match
    # But allow some extra keys (up to 50% more)
    if len(value_keys - def_keys) > max(len(def_keys) * 0.5, 3):
        return False
    
    # Check if at least 50% of definition keys are present in value
    if def_keys:
        overlap = len(value_keys & def_keys)
        if overlap < len(def_keys) * 0.5:
            return False
    
    return True


def _infer_type_schema(value: Any) -> Dict:
    """Infer JSON schema for a value."""
    if value is None:
        return {"type": "null"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, list):
        if value:
            return {
                "type": "array",
                "items": _infer_type_schema(value[0])
            }
        else:
            return {"type": "array", "items": {}}
    elif isinstance(value, dict):
        properties = {}
        for k, v in value.items():
            properties[k] = _infer_type_schema(v)
        return {
            "type": "object",
            "properties": properties
        }
    else:
        return {"type": "string"}  # Fallback


def _get_json_type(value: Any) -> str:
    """Get JSON type name for a value."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    else:
        return "string"


def fix_ref_paths(schema: Dict) -> None:
    """Fix invalid $ref paths in schema (recursively).
    
    Changes:
    - #/definitions/ -> #/$defs/
    - #/defs/ -> #/$defs/ (missing $)
    - Invalid paths like #/$defs/Offer/id -> #/$defs/Offer
    """
    if not isinstance(schema, dict):
        return
    
    # Fix $ref in this level
    if "$ref" in schema:
        ref = schema["$ref"]
        # Fix /definitions/ to /$defs/
        if "/definitions/" in ref:
            schema["$ref"] = ref.replace("/definitions/", "/$defs/")
        # Fix #/defs/ to #/$defs/ (missing $)
        elif ref.startswith("#/defs/"):
            schema["$ref"] = ref.replace("#/defs/", "#/$defs/", 1)
        # Fix invalid paths like #/$defs/Offer/id -> #/$defs/Offer
        elif "/$defs/" in ref and ref.count("/") > 2:
            # Extract just the definition name
            parts = ref.split("/$defs/")
            if len(parts) == 2:
                def_name = parts[1].split("/")[0]
                schema["$ref"] = f"#/$defs/{def_name}"
    
    # Recursively fix in nested structures
    for key, value in schema.items():
        if isinstance(value, dict):
            fix_ref_paths(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    fix_ref_paths(item)


def fix_schema(schema: Dict, expected: Any, sample_id: str) -> Tuple[Dict, List[str]]:
    """Fix a schema to have root type/properties.
    
    Args:
        schema: Original schema
        expected: Expected output value
        sample_id: Sample identifier for error messages
        
    Returns:
        Tuple of (fixed_schema, list_of_warnings)
    """
    warnings = []
    
    # First, fix any invalid $ref paths
    schema = deepcopy(schema)
    fix_ref_paths(schema)
    
    # Check if schema already has root structure
    has_type = "type" in schema
    has_properties = "properties" in schema
    
    if has_type or has_properties:
        # Already has root structure - but may still need $ref fixes
        return schema, warnings
    
    # Schema needs fixing - infer from expected
    defs = schema.get("$defs", {})
    if not defs:
        warnings.append(f"No $defs found - cannot infer root structure")
        # Try to infer from expected directly
        root_schema = infer_root_structure(expected, {})
        if root_schema:
            new_schema = {**root_schema, **schema}
            return new_schema, warnings
        else:
            warnings.append(f"Could not infer root structure from expected")
            return schema, warnings
    
    # Infer root structure from expected and definitions
    root_schema = infer_root_structure(expected, defs)
    
    if not root_schema:
        warnings.append(f"Could not infer root structure")
        return schema, warnings
    
    # Create new schema with root structure
    new_schema = deepcopy(schema)
    new_schema.update(root_schema)
    
    # Remove None from required if present
    if "required" in new_schema and new_schema["required"] is None:
        new_schema.pop("required")
    
    # Fix $ref paths in the new schema
    fix_ref_paths(new_schema)
    
    return new_schema, warnings


def validate_schema_against_expected(schema: Dict, expected: Any, sample_id: str) -> Tuple[bool, List[str]]:
    """Validate that schema matches expected output.
    
    Args:
        schema: Schema to validate
        expected: Expected output
        sample_id: Sample identifier
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # First, validate schema itself
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        errors.append(f"Invalid schema: {e.message}")
        return False, errors
    
    # Then validate expected against schema
    try:
        validate(instance=expected, schema=schema)
        return True, errors
    except ValidationError as e:
        errors.append(f"Expected output doesn't match schema: {e.message}")
        if e.path:
            errors.append(f"  Path: {list(e.path)}")
        return False, errors
    except Exception as e:
        errors.append(f"Validation error: {type(e).__name__}: {e}")
        return False, errors


def main():
    """Main function to fix schemas in paraloq_data.jsonl."""
    data_file = Path("src/tasks/structured/.data/paraloq_data.jsonl")
    backup_file = data_file.with_suffix(".jsonl.backup")
    
    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        sys.exit(1)
    
    print(f"Fixing schemas in: {data_file}")
    print("=" * 80)
    
    # Read all samples
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON on line {line_num}: {e}")
                sys.exit(1)
    
    print(f"Loaded {len(samples)} samples")
    
    # Process each sample
    fixed_count = 0
    already_correct = 0
    failed_count = 0
    validation_errors = []
    valid_samples = []  # Samples that pass validation
    
    for sample in samples:
        sample_id = sample.get("id", "unknown")
        schema = sample.get("schema", {})
        expected = sample.get("expected")
        
        if not schema:
            print(f"⚠️  Sample {sample_id}: Dropping - no schema")
            continue
        
        # Check if needs fixing
        has_type = "type" in schema
        has_properties = "properties" in schema
        
        if has_type or has_properties:
            # Already correct - validate it
            is_valid, errors = validate_schema_against_expected(schema, expected, sample_id)
            if is_valid:
                already_correct += 1
                valid_samples.append(sample)
            else:
                validation_errors.append((sample_id, errors))
                print(f"❌ Sample {sample_id}: Dropping - validation failed:")
                for error in errors:
                    print(f"    {error}")
        else:
            # Needs fixing
            fixed_schema, warnings = fix_schema(schema, expected, sample_id)
            
            if warnings:
                print(f"⚠️  Sample {sample_id}: {', '.join(warnings)}")
            
            # Validate fixed schema
            is_valid, errors = validate_schema_against_expected(fixed_schema, expected, sample_id)
            
            if is_valid:
                sample["schema"] = fixed_schema
                fixed_count += 1
                valid_samples.append(sample)
            else:
                failed_count += 1
                validation_errors.append((sample_id, errors))
                print(f"❌ Sample {sample_id}: Dropping - failed to fix schema:")
                for error in errors:
                    print(f"    {error}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples loaded: {len(samples)}")
    print(f"Already correct: {already_correct}")
    print(f"Fixed: {fixed_count}")
    print(f"Dropped (validation errors): {len(validation_errors)}")
    print(f"Valid samples to write: {len(valid_samples)}")
    
    if validation_errors:
        print("\n❌ DROPPED SAMPLES (validation errors):")
        for sample_id, errors in validation_errors:
            print(f"\n  {sample_id}:")
            for error in errors:
                print(f"    - {error}")
    
    # Create backup
    if fixed_count > 0 or validation_errors:
        print(f"\nCreating backup: {backup_file}")
        import shutil
        shutil.copy2(data_file, backup_file)
        print(f"✅ Backup created")
    
    # Write valid samples only
    print(f"\nWriting {len(valid_samples)} valid samples to: {data_file}")
    with open(data_file, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(valid_samples)} valid samples")
    
    if validation_errors:
        print(f"\n⚠️  Dropped {len(validation_errors)} samples with validation errors")
        print("   All remaining samples have valid schemas that match their expected output")
    else:
        print("\n✅ All schemas are valid!")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

