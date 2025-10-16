"""
Schema Complexity Analysis Module

Computes complexity features from JSON schemas to analyze
correlation with model performance.
"""

import logging
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)


def compute_schema_complexity(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute complexity features from a JSON schema.
    
    Args:
        schema: JSON schema dictionary
        
    Returns:
        Dictionary of complexity features
    """
    features = {
        'total_fields': 0,
        'required_fields': 0,
        'optional_fields': 0,
        'max_nesting_depth': 0,
        'has_arrays': False,
        'has_nested_objects': False,
        'field_types_diversity': 0,
        'total_paths': 0,
        'array_fields': 0,
        'object_fields': 0,
    }
    
    # Track unique types
    types_seen = set()
    
    def analyze_properties(props: Dict[str, Any], required: Set[str], depth: int) -> int:
        """Recursively analyze schema properties."""
        nonlocal features, types_seen
        
        field_count = 0
        features['max_nesting_depth'] = max(features['max_nesting_depth'], depth)
        
        for field_name, field_schema in props.items():
            field_count += 1
            features['total_paths'] += 1
            
            # Track if required or optional
            if field_name in required:
                features['required_fields'] += 1
            else:
                features['optional_fields'] += 1
            
            # Get field type
            field_type = field_schema.get('type')
            if field_type:
                types_seen.add(field_type)
            
            # Check for arrays
            if field_type == 'array':
                features['has_arrays'] = True
                features['array_fields'] += 1
                
                # Recursively analyze array items if they're objects
                items_schema = field_schema.get('items', {})
                if isinstance(items_schema, dict) and items_schema.get('type') == 'object':
                    features['has_nested_objects'] = True
                    items_props = items_schema.get('properties', {})
                    items_required = set(items_schema.get('required', []))
                    field_count += analyze_properties(items_props, items_required, depth + 1)
                elif isinstance(items_schema, list):
                    # Handle list of schemas
                    for item in items_schema:
                        if isinstance(item, dict) and item.get('type') == 'object':
                            features['has_nested_objects'] = True
                            items_props = item.get('properties', {})
                            items_required = set(item.get('required', []))
                            field_count += analyze_properties(items_props, items_required, depth + 1)
            
            # Check for nested objects
            elif field_type == 'object':
                features['has_nested_objects'] = True
                features['object_fields'] += 1
                nested_props = field_schema.get('properties', {})
                nested_required = set(field_schema.get('required', []))
                field_count += analyze_properties(nested_props, nested_required, depth + 1)
            
            # Handle anyOf, oneOf, allOf
            for combiner in ['anyOf', 'oneOf', 'allOf']:
                if combiner in field_schema:
                    for sub_schema in field_schema[combiner]:
                        if sub_schema.get('type') == 'object':
                            features['has_nested_objects'] = True
                            sub_props = sub_schema.get('properties', {})
                            sub_required = set(sub_schema.get('required', []))
                            field_count += analyze_properties(sub_props, sub_required, depth + 1)
        
        return field_count
    
    # Start analysis from root
    if schema.get('type') == 'object':
        properties = schema.get('properties', {})
        required = set(schema.get('required', []))
        features['total_fields'] = analyze_properties(properties, required, depth=0)
    
    features['field_types_diversity'] = len(types_seen)
    
    return features


def compute_complexity_score(features: Dict[str, Any]) -> float:
    """
    Compute a single complexity score from features (0-1 scale).
    
    Higher score = more complex schema
    
    Args:
        features: Complexity features from compute_schema_complexity
        
    Returns:
        Normalized complexity score (0-1)
    """
    # Weighted components (normalized to 0-1 range)
    field_score = min(features['total_fields'] / 30.0, 1.0)  # Cap at 30 fields
    depth_score = min(features['max_nesting_depth'] / 4.0, 1.0)  # Cap at depth 4
    diversity_score = min(features['field_types_diversity'] / 6.0, 1.0)  # Cap at 6 types
    
    # Penalty for arrays and nested objects
    structure_penalty = 0.0
    if features['has_arrays']:
        structure_penalty += 0.1
    if features['has_nested_objects']:
        structure_penalty += 0.2
    
    # Weighted combination
    complexity = (
        0.40 * field_score +
        0.30 * depth_score +
        0.15 * diversity_score +
        0.15 * structure_penalty
    )
    
    return min(complexity, 1.0)


def classify_complexity(score: float) -> str:
    """
    Classify complexity score into bins.
    
    Args:
        score: Complexity score from compute_complexity_score
        
    Returns:
        Complexity bin name
    """
    if score < 0.33:
        return 'simple'
    elif score < 0.67:
        return 'medium'
    else:
        return 'complex'





