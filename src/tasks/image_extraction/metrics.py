"""Document extraction metrics with value normalization.

Wraps the structured extraction MetricsCalculator with preprocessing
specific to document extraction use cases:
- DateTime fields compared as dates only
- Numeric string fields (e.g., "0001") compared as integers
- More lenient string comparison thresholds
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..structured.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class DocumentExtractionMetrics:
    """Metrics calculator specialized for document extraction tasks.
    
    Preprocesses values before comparison to handle common document
    extraction variations:
    - Dates/datetimes normalized to date-only strings
    - Padded numeric strings normalized to integers
    - String comparisons use lenient thresholds
    """

    def __init__(self, config: Dict):
        """Initialize document extraction metrics.

        Args:
            config: Configuration dictionary with optional:
                - metrics.numeric_string_fields: List of field names to treat as numeric
                - metrics.partial_matching.string.exact_threshold: Lenient threshold
                - metrics.partial_matching.string.partial_threshold: Lenient threshold
        """
        self.config = config
        
        # Get document-specific config
        metrics_config = config.get("metrics", {})
        self.numeric_string_fields = set(
            metrics_config.get("numeric_string_fields", [
                "punto_de_venta",
                "numero_de_comprobante",
            ])
        )
        # Fields to exclude from evaluation (in schema but not in expected data)
        self.ignored_fields = set(metrics_config.get("ignored_fields", []))
        
        # Apply lenient string thresholds if not explicitly configured
        if "partial_matching" not in metrics_config:
            metrics_config["partial_matching"] = {}
        if "string" not in metrics_config["partial_matching"]:
            metrics_config["partial_matching"]["string"] = {}
        
        string_config = metrics_config["partial_matching"]["string"]
        # More lenient defaults for document extraction
        string_config.setdefault("exact_threshold", 0.85)
        string_config.setdefault("partial_threshold", 0.40)
        
        # Update config and create underlying calculator
        config["metrics"] = metrics_config
        self._calculator = MetricsCalculator(config)

    def calculate_all(
        self,
        prediction: Dict,
        expected: Dict,
        schema: Dict,
        error: str = None,
        error_type: str = None
    ) -> Dict:
        """Calculate metrics with value normalization.

        Args:
            prediction: Model's predicted output
            expected: Expected (ground truth) output
            schema: Target JSON schema
            error: Error message if generation failed
            error_type: Type of error ('connectivity_error' or 'invalid_response')

        Returns:
            Dictionary of metric scores and diagnostics
        """
        # Debug: log schema status
        if not schema or not schema.get("properties"):
            logger.warning(f"Schema is empty or missing properties: {bool(schema)}, keys: {list(schema.keys()) if schema else []}")
        
        if error or prediction is None:
            return self._calculator.calculate_all(prediction, expected, schema, error, error_type)
        
        # Track strict type compliance BEFORE coercion
        # This tells us if the model followed the exact schema format
        strict_type_valid, type_errors = self._check_strict_types(prediction, schema)
        
        # Step 1: Normalize values FIRST (for comparison, doesn't change types)
        try:
            norm_prediction = self._normalize_values(prediction, schema)
            norm_expected = self._normalize_values(expected, schema)
        except Exception as e:
            logger.warning(f"Normalization failed, using original values: {e}")
            norm_prediction = prediction
            norm_expected = expected
        
        # Step 2: Coerce types AFTER normalization to match schema for validation
        # This allows metrics calculation even when model returns wrong types
        try:
            coerced_prediction = self._coerce_types_to_schema(norm_prediction, schema)
            if type_errors:
                logger.info(f"Type coercion: punto_de_venta {norm_prediction.get('punto_de_venta')} ({type(norm_prediction.get('punto_de_venta')).__name__}) -> {coerced_prediction.get('punto_de_venta')} ({type(coerced_prediction.get('punto_de_venta')).__name__})")
        except Exception as e:
            logger.warning(f"Type coercion failed, using normalized prediction: {e}")
            coerced_prediction = norm_prediction
        
        # Step 3: Also coerce expected values for fair comparison
        try:
            coerced_expected = self._coerce_types_to_schema(norm_expected, schema)
        except Exception as e:
            coerced_expected = norm_expected
        
        # Step 4: Filter ignored fields BEFORE schema validation
        # This prevents ignored fields from causing validation errors
        filtered_prediction = self._filter_prediction_fields(
            coerced_prediction, coerced_expected
        )
        
        # Step 5: Create modified schema without ignored_fields in required list
        # This allows filtered prediction to pass validation
        evaluation_schema = self._create_evaluation_schema(schema)
        
        # Step 6: Validate schema with filtered prediction (ignored fields excluded)
        # This ensures structure compliance without penalizing ignored fields
        schema_valid = self._calculator._validate_schema(filtered_prediction, evaluation_schema)
        
        # Calculate metrics on filtered prediction (ignored fields excluded from evaluation)
        metrics = self._calculator.calculate_all(
            filtered_prediction, coerced_expected, evaluation_schema, error, error_type
        )
        
        # Schema validity is already set correctly from filtered prediction validation
        # No need to override - metrics["valid"] is already set by calculate_all
        
        # Debug: Log if still failing
        if not metrics.get("valid"):
            logger.warning(f"Metrics calculator returned valid=False. Error: {metrics.get('error')}")
        
        # Add type compliance metric (did the model return correct types?)
        # This is separate from schema_validity which now passes after coercion
        metrics["strict_type_compliance"] = 1.0 if strict_type_valid else 0.0
        metrics["type_errors"] = type_errors
        
        # Add document-specific score
        metrics["document_extraction_score"] = self._calculate_document_score(metrics)
        
        return metrics

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            
        Returns:
            Aggregated metrics with document_extraction_score
        """
        aggregated = self._calculator.aggregate_metrics(all_metrics)
        
        # Aggregate document extraction scores
        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        if valid_metrics:
            doc_scores = [
                m.get("document_extraction_score", 0.0) 
                for m in valid_metrics
            ]
            aggregated["document_extraction_score"] = sum(doc_scores) / len(doc_scores)
            
            # Aggregate strict type compliance (separate from schema_validity)
            type_compliance_scores = [
                m.get("strict_type_compliance", 1.0)
                for m in valid_metrics
            ]
            aggregated["strict_type_compliance_rate"] = sum(type_compliance_scores) / len(type_compliance_scores)
            
            # Count samples with type errors
            samples_with_type_errors = sum(
                1 for m in valid_metrics if m.get("type_errors")
            )
            aggregated["samples_with_type_errors"] = samples_with_type_errors
        else:
            aggregated["document_extraction_score"] = 0.0
            aggregated["strict_type_compliance_rate"] = 0.0
            aggregated["samples_with_type_errors"] = 0
        
        return aggregated

    def _check_strict_types(self, data: Dict, schema: Dict) -> Tuple[bool, List[str]]:
        """Check if prediction types strictly match schema expectations.
        
        This is checked BEFORE type coercion to track type compliance separately
        from extraction accuracy.
        
        Args:
            data: Prediction dictionary
            schema: JSON schema with type definitions
            
        Returns:
            Tuple of (is_valid, list_of_type_errors)
        """
        if not isinstance(data, dict):
            return False, ["prediction is not a dict"]
        
        properties = schema.get("properties", {})
        type_errors = []
        
        for key, value in data.items():
            if key not in properties:
                continue  # Extra fields handled by schema validation
                
            field_schema = properties.get(key, {})
            expected_type = field_schema.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                type_errors.append(f"{key}: expected string, got {type(value).__name__}")
            elif expected_type == "integer" and not isinstance(value, int):
                # Note: bools are subclass of int in Python, exclude them
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    type_errors.append(f"{key}: expected integer, got {type(value).__name__}")
                elif isinstance(value, float) and not value.is_integer():
                    type_errors.append(f"{key}: expected integer, got float with decimal")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                type_errors.append(f"{key}: expected number, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                type_errors.append(f"{key}: expected boolean, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                type_errors.append(f"{key}: expected array, got {type(value).__name__}")
            elif expected_type == "object" and not isinstance(value, dict):
                type_errors.append(f"{key}: expected object, got {type(value).__name__}")
        
        return len(type_errors) == 0, type_errors

    def _coerce_types_to_schema(self, data: Dict, schema: Dict) -> Dict:
        """Coerce prediction types to match schema expectations.
        
        This handles common model output variations like returning integers
        instead of strings for fields like punto_de_venta.
        
        Args:
            data: Prediction dictionary
            schema: JSON schema with type definitions
            
        Returns:
            Dictionary with coerced types
        """
        if not isinstance(data, dict):
            return data
        
        properties = schema.get("properties", {})
        coerced = {}
        coercion_count = 0
        
        for key, value in data.items():
            field_schema = properties.get(key, {})
            expected_type = field_schema.get("type")
            
            # Coerce int/float to string if schema expects string
            if expected_type == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                # For numeric string fields, preserve leading zeros format
                if key in self.numeric_string_fields:
                    # Pad to match expected format (e.g., punto_de_venta is 4-5 digits)
                    coerced[key] = str(int(value)).zfill(4) if key == "punto_de_venta" else str(int(value))
                else:
                    coerced[key] = str(value)
                coercion_count += 1
                logger.info(f"Coerced {key}: {value} ({type(value).__name__}) -> {coerced[key]} (string)")
            # Coerce string to int if schema expects integer
            elif expected_type == "integer" and isinstance(value, str):
                try:
                    coerced[key] = int(value.lstrip("0") or "0")
                    coercion_count += 1
                except ValueError:
                    coerced[key] = value
            # Coerce float to int if schema expects integer and value is whole number
            elif expected_type == "integer" and isinstance(value, float) and value.is_integer():
                coerced[key] = int(value)
                coercion_count += 1
            # Coerce string to number if schema expects number
            elif expected_type == "number" and isinstance(value, str):
                try:
                    coerced[key] = float(value)
                    coercion_count += 1
                except ValueError:
                    coerced[key] = value
            else:
                coerced[key] = value
        
        if coercion_count > 0:
            logger.info(f"Type coercion: {coercion_count} field(s) coerced")
        
        return coerced

    def _normalize_values(self, data: Dict, schema: Dict) -> Dict:
        """Normalize values based on schema and field configuration.

        Args:
            data: Dictionary of field values
            schema: JSON schema with type/format hints

        Returns:
            Dictionary with normalized values
        """
        if not isinstance(data, dict):
            return data
        
        properties = schema.get("properties", {})
        normalized = {}
        
        for key, value in data.items():
            field_schema = properties.get(key, {})
            normalized[key] = self._normalize_field(key, value, field_schema)
        
        return normalized

    def _normalize_field(self, field_name: str, value: Any, field_schema: Dict) -> Any:
        """Normalize a single field value.

        Args:
            field_name: Name of the field
            value: Field value to normalize
            field_schema: Schema definition for this field

        Returns:
            Normalized value
        """
        if value is None:
            return None
        
        # Check for date/datetime format in schema
        field_format = field_schema.get("format", "")
        if field_format in ("date", "date-time"):
            return self._normalize_datetime(value)
        
        # Check for numeric string fields
        if field_name in self.numeric_string_fields:
            return self._normalize_numeric_string(value)
        
        # Normalize text strings (e.g., "S.A." -> "SA")
        if isinstance(value, str) and field_schema.get("type") == "string":
            # Skip enum fields - they need exact matching
            if "enum" not in field_schema:
                return self._normalize_text(value)
        
        # Return value unchanged for other types
        return value
    
    def _normalize_text(self, value: str) -> str:
        """Normalize text for comparison.
        
        Handles common OCR/extraction variations:
        - "S.A." vs "SA" 
        - "S.R.L." vs "SRL"
        - Extra spaces
        - Different punctuation
        - Case differences ("COMPANY" vs "Company")
        
        Args:
            value: Text string to normalize
            
        Returns:
            Normalized text string (uppercase)
        """
        if not isinstance(value, str):
            return value
        
        # Remove common punctuation that varies in OCR
        # Keep alphanumeric, spaces, and essential punctuation
        normalized = value
        
        # Normalize company suffixes: "S.A." -> "SA", "S.R.L." -> "SRL"
        normalized = re.sub(r'\bS\.A\.?\b', 'SA', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bS\.R\.L\.?\b', 'SRL', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bS\.A\.S\.?\b', 'SAS', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bS\.C\.A\.?\b', 'SCA', normalized, flags=re.IGNORECASE)
        
        # Remove standalone periods and commas (but keep decimal points in numbers)
        normalized = re.sub(r'(?<!\d)\.(?!\d)', '', normalized)  # Period not between digits
        normalized = re.sub(r'(?<!\d),(?!\d)', '', normalized)   # Comma not between digits
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Normalize case (uppercase for consistent comparison)
        normalized = normalized.upper()
        
        return normalized

    def _normalize_datetime(self, value: Any) -> str:
        """Normalize datetime to date-only string (YYYY-MM-DD).

        Args:
            value: Datetime value (string or datetime object)

        Returns:
            Date string in YYYY-MM-DD format, or original if parsing fails
        """
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value.date().isoformat()
        
        if isinstance(value, str):
            # Try various datetime formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    # Handle timezone suffix
                    clean_value = value.replace("Z", "").split("+")[0]
                    dt = datetime.strptime(clean_value, fmt)
                    return dt.date().isoformat()
                except ValueError:
                    continue
            
            # Try ISO format parsing
            try:
                # Handle ISO format with timezone
                clean_value = re.sub(r'[+-]\d{2}:\d{2}$', '', value)
                clean_value = clean_value.replace("Z", "")
                dt = datetime.fromisoformat(clean_value)
                return dt.date().isoformat()
            except ValueError:
                pass
            
            # If already a date-only string, return as-is
            if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                return value
        
        # Return original if parsing fails
        logger.debug(f"Could not parse datetime value: {value}")
        return value

    def _normalize_numeric_string(self, value: Any) -> int:
        """Normalize numeric string to integer.

        Handles padded strings like "0001" -> 1, "00073916" -> 73916

        Args:
            value: String value that represents a number

        Returns:
            Integer value, or original if conversion fails
        """
        if value is None:
            return None
        
        if isinstance(value, int):
            return value
        
        if isinstance(value, str):
            try:
                # Strip leading zeros and convert
                return int(value.lstrip("0") or "0")
            except ValueError:
                logger.debug(f"Could not convert to int: {value}")
                return value
        
        return value

    def _calculate_document_score(self, metrics: Dict) -> float:
        """Calculate document extraction quality score.

        Weights:
        - numeric_precision_rate: 50% (exact numeric matches are critical)
        - field_f1_partial: 35% (overall extraction quality)
        - schema_validity: 15% (structure compliance)

        Args:
            metrics: Per-sample metrics dictionary

        Returns:
            Document extraction score from 0.0 to 1.0
        """
        weights = self.config.get("metrics", {}).get(
            "document_extraction_score", {}
        ).get("weights", {})
        
        numeric_weight = weights.get("numeric_precision", 0.50)
        f1_weight = weights.get("field_f1_partial", 0.35)
        validity_weight = weights.get("schema_validity", 0.15)
        
        # Get component scores
        numeric_precision = metrics.get("numeric_precision_rate", 1.0)
        f1_partial = metrics.get("field_f1_partial", 0.0)
        schema_validity = metrics.get("schema_validity", 0.0)
        
        score = (
            numeric_weight * numeric_precision +
            f1_weight * f1_partial +
            validity_weight * schema_validity
        )
        
        return score

    def _filter_prediction_fields(self, prediction: Dict, expected: Dict) -> Dict:
        """Filter prediction to remove ignored fields from evaluation.
        
        Removes only fields specified in `ignored_fields` config. Other unexpected
        fields remain and will be penalized as spurious/hallucinations.
        
        Args:
            prediction: Prediction dictionary (may contain extra fields)
            expected: Expected dictionary (ground truth fields)
            
        Returns:
            Filtered prediction with ignored fields removed
        """
        if not self.ignored_fields:
            return prediction
        
        
        filtered = {}
        for key, value in prediction.items():
            # Keep field if:
            # 1. It's in expected (will be evaluated)
            # 2. It's NOT in ignored_fields (other unexpected fields still penalized)
            if key in expected or key not in self.ignored_fields:
                filtered[key] = value
            # Otherwise, field is in ignored_fields and not in expected -> skip it
            
        
        if len(filtered) < len(prediction):
            removed = set(prediction.keys()) - set(filtered.keys())
            logger.debug(f"Filtered {len(removed)} ignored field(s) from evaluation: {removed}")
        
        return filtered

    def _create_evaluation_schema(self, schema: Dict) -> Dict:
        """Create a schema for evaluation that doesn't require ignored_fields.
        
        Removes ignored_fields from the required list so that filtered predictions
        can pass validation during metrics calculation.
        
        Args:
            schema: Original JSON schema
            
        Returns:
            Modified schema with ignored_fields removed from required list
        """
        if not self.ignored_fields or "required" not in schema:
            return schema
        
        # Create a copy to avoid modifying original
        import copy
        eval_schema = copy.deepcopy(schema)
        
        # Remove ignored_fields from required list
        if "required" in eval_schema:
            original_required = eval_schema["required"]
            eval_schema["required"] = [
                field for field in original_required
                if field not in self.ignored_fields
            ]
            
            if len(eval_schema["required"]) < len(original_required):
                removed = set(original_required) - set(eval_schema["required"])
                logger.debug(f"Removed {len(removed)} ignored field(s) from schema required list: {removed}")
        
        return eval_schema

