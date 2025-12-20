#!/usr/bin/env python3
"""Validate JSON schemas in paraloq_data.jsonl.

Checks if schemas have required 'type' or 'properties' fields at root level,
and validates them using jsonschema library.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from jsonschema import Draft202012Validator, validate, ValidationError
    from jsonschema.exceptions import SchemaError
except ImportError:
    print("ERROR: jsonschema library not installed. Install with: pip install jsonschema")
    sys.exit(1)


def check_schema_structure(schema: Dict, sample_id: str) -> Tuple[bool, List[str]]:
    """Check if schema has required structure (type or properties at root).
    
    Args:
        schema: JSON schema dictionary
        sample_id: Sample identifier for error messages
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if schema is a dict
    if not isinstance(schema, dict):
        errors.append(f"Schema is not a dictionary (type: {type(schema).__name__})")
        return False, errors
    
    # Check for root-level 'type' or 'properties'
    has_type = "type" in schema
    has_properties = "properties" in schema
    has_defs = "$defs" in schema
    
    if not has_type and not has_properties:
        errors.append("Missing both 'type' and 'properties' at root level")
        if has_defs:
            errors.append("  Note: Schema has '$defs' but no root 'type' or 'properties'")
    
    # Additional checks
    if has_type:
        type_value = schema["type"]
        if type_value == "object" and not has_properties:
            errors.append("Schema has type='object' but no 'properties' field")
    
    if has_properties and not has_type:
        # This is technically valid (properties implies object type)
        # but some validators prefer explicit type
        pass
    
    return len(errors) == 0, errors


def validate_schema_with_jsonschema(schema: Dict, sample_id: str) -> Tuple[bool, List[str]]:
    """Validate schema using jsonschema library.
    
    Args:
        schema: JSON schema dictionary
        sample_id: Sample identifier for error messages
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Try to create a validator - this will fail if schema is invalid
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        errors.append(f"Invalid JSON Schema: {e.message}")
        if e.path:
            errors.append(f"  Path: {list(e.path)}")
    except Exception as e:
        errors.append(f"Unexpected error validating schema: {type(e).__name__}: {e}")
    
    return len(errors) == 0, errors


def analyze_schema(schema: Dict, sample_id: str) -> Dict:
    """Analyze a single schema and return validation results.
    
    Args:
        schema: JSON schema dictionary
        sample_id: Sample identifier
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "sample_id": sample_id,
        "has_type": "type" in schema,
        "has_properties": "properties" in schema,
        "has_defs": "$defs" in schema,
        "root_keys": list(schema.keys())[:10],  # First 10 keys for inspection
        "structure_valid": False,
        "structure_errors": [],
        "jsonschema_valid": False,
        "jsonschema_errors": [],
    }
    
    # Check structure
    structure_valid, structure_errors = check_schema_structure(schema, sample_id)
    result["structure_valid"] = structure_valid
    result["structure_errors"] = structure_errors
    
    # Validate with jsonschema library
    jsonschema_valid, jsonschema_errors = validate_schema_with_jsonschema(schema, sample_id)
    result["jsonschema_valid"] = jsonschema_valid
    result["jsonschema_errors"] = jsonschema_errors
    
    return result


def main():
    """Main validation function."""
    data_file = Path("src/tasks/structured/.data/paraloq_data.jsonl")
    
    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        print(f"  Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"Validating schemas in: {data_file}")
    print("=" * 80)
    
    results = []
    invalid_structure = []
    invalid_jsonschema = []
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                sample_id = sample.get("id", f"line_{line_num}")
                schema = sample.get("schema", {})
                
                if not schema:
                    print(f"WARNING: Sample {sample_id} has no schema field")
                    continue
                
                result = analyze_schema(schema, sample_id)
                results.append(result)
                
                if not result["structure_valid"]:
                    invalid_structure.append(result)
                
                if not result["jsonschema_valid"]:
                    invalid_jsonschema.append(result)
                    
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected error on line {line_num}: {e}")
    
    # Print summary
    print(f"\nTotal samples analyzed: {len(results)}")
    print(f"Samples with invalid structure (missing type/properties): {len(invalid_structure)}")
    print(f"Samples with invalid JSON Schema: {len(invalid_jsonschema)}")
    print("=" * 80)
    
    # Print detailed results for invalid schemas
    if invalid_structure:
        print("\n❌ SCHEMAS WITH INVALID STRUCTURE (missing 'type' or 'properties'):")
        print("=" * 80)
        for result in invalid_structure:
            print(f"\nSample ID: {result['sample_id']}")
            print(f"  Has 'type': {result['has_type']}")
            print(f"  Has 'properties': {result['has_properties']}")
            print(f"  Has '$defs': {result['has_defs']}")
            print(f"  Root keys: {result['root_keys']}")
            for error in result['structure_errors']:
                print(f"  ERROR: {error}")
    
    if invalid_jsonschema:
        print("\n❌ SCHEMAS WITH INVALID JSON SCHEMA:")
        print("=" * 80)
        for result in invalid_jsonschema:
            print(f"\nSample ID: {result['sample_id']}")
            for error in result['jsonschema_errors']:
                print(f"  ERROR: {error}")
    
    # Print information about JSON Schema requirements
    print("\n" + "=" * 80)
    print("JSON SCHEMA REQUIREMENTS:")
    print("=" * 80)
    print("According to JSON Schema specification (Draft 2020-12):")
    print("  - A schema should have 'type' field to specify the data type")
    print("  - For object types, 'properties' field defines the object structure")
    print("  - '$defs' is used for reusable definitions but doesn't define root structure")
    print("\nSURUS endpoint requires:")
    print("  - At least one of: 'type' OR 'properties' at root level")
    print("  - If 'type'='object', 'properties' should also be present")
    
    # Check if structure issues match jsonschema issues
    structure_only = [r for r in invalid_structure if r["jsonschema_valid"]]
    if structure_only:
        print(f"\n⚠️  {len(structure_only)} schemas are valid JSON Schema but missing type/properties:")
        print("   (These might work with jsonschema library but fail with SURUS endpoint)")
        for result in structure_only[:5]:  # Show first 5
            print(f"   - {result['sample_id']}")
        if len(structure_only) > 5:
            print(f"   ... and {len(structure_only) - 5} more")
    
    # Exit with error code if invalid schemas found
    if invalid_structure or invalid_jsonschema:
        sys.exit(1)
    else:
        print("\n✅ All schemas are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()



