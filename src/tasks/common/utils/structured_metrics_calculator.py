"""Main metrics calculator implementing all metrics from the specification.

This calculator provides comprehensive metrics for structured data extraction,
helping differentiate between different types of failures:

1. Connectivity errors: Infrastructure/network issues (not model quality)
2. Invalid responses: Model can't produce valid JSON
3. Schema validation failures: Model produces JSON but wrong structure
4. Extraction errors: Model produces correct structure but wrong data

The Extraction Quality Score (EQS) is the primary metric, combining:
- Schema validity (can the model follow the structure?)
- Field F1 score (how accurate is the extracted data?)
- Hallucination rate (does the model add spurious fields?)
"""

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from jsonschema import validate, ValidationError

from .partial_matching import PartialMatcher

logger = logging.getLogger(__name__)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DMY_DASH_DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
_DMY_SLASH_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")

def _field_pattern_to_regex(pattern: str) -> str:
    """Convert a field path pattern to a regex.

    Supported:
    - Exact keys like `emisor.cuit`
    - Array wildcard `[]` matching any index, e.g. `items[].codigo` matches `items[0].codigo`
    - `*` wildcard matching any substring (including across separators)
    """
    import re

    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\[\]", r"\[\d+\]")
    escaped = escaped.replace(r"\*", r".*")
    return f"^{escaped}$"


def _matches_any_pattern(key: str, patterns: List[str]) -> bool:
    import re

    for pattern in patterns:
        try:
            if re.match(_field_pattern_to_regex(pattern), key):
                return True
        except re.error:
            continue
    return False


def _digits_only(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    return "".join(ch for ch in s if ch.isdigit())


class MetricsCalculator:
    """Calculate comprehensive metrics for structured data extraction."""

    def __init__(self, config: Dict, strict: bool = False):
        """Initialize metrics calculator.

        Args:
            config: Configuration dictionary with structure:
                {
                    "metrics": {
                        "partial_credit": 0.3,
                        "extraction_quality_score": {
                            "weights": {
                                "schema_validity": 0.15,
                                "field_f1_partial": 0.70,
                                "inverted_hallucination": 0.15
                            }
                        },
                        "partial_matching": {
                            "string": {...},
                            "number": {...}
                        },
                        "normalization": {...}
                    }
                }
            strict: If True, use stricter matching thresholds (default: False)
        """
        self.config = config
        self.partial_matcher = PartialMatcher(config, strict=strict)

    def _resolve_schema_node(self, schema_node: Any, root_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve local $ref pointers and merge local overrides."""
        if not isinstance(schema_node, dict):
            return {}

        resolved = dict(schema_node)
        seen_refs = set()

        while isinstance(resolved, dict) and "$ref" in resolved:
            ref = resolved.get("$ref")
            if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen_refs:
                break
            seen_refs.add(ref)

            target: Any = root_schema
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, dict) or part not in target:
                    target = None
                    break
                target = target[part]

            if not isinstance(target, dict):
                break

            merged = dict(target)
            for key, value in resolved.items():
                if key != "$ref":
                    merged[key] = value
            resolved = merged

        return resolved if isinstance(resolved, dict) else {}

    def _schema_expects_date(self, schema_node: Dict[str, Any]) -> bool:
        """Return True when schema indicates a date-like string field."""
        node = schema_node if isinstance(schema_node, dict) else {}
        if node.get("format") in {"date", "date-time"}:
            return True

        pattern = node.get("pattern")
        if isinstance(pattern, str):
            # Conservative heuristic: treat canonical yyyy-mm-dd style patterns as date fields.
            if "\\d{4}" in pattern and "\\d{2}" in pattern and "-" in pattern:
                return True

        for keyword in ("anyOf", "oneOf", "allOf"):
            variants = node.get(keyword)
            if isinstance(variants, list):
                for variant in variants:
                    if self._schema_expects_date(variant if isinstance(variant, dict) else {}):
                        return True
        return False

    def _canonicalize_date_string(self, value: str) -> str:
        """Canonicalize accepted date formats to YYYY-MM-DD."""
        candidate = value.strip()

        if _ISO_DATE_RE.match(candidate):
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                return candidate
            except ValueError:
                return value

        if _DMY_DASH_DATE_RE.match(candidate):
            try:
                dt = datetime.strptime(candidate, "%d-%m-%Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return value

        if _DMY_SLASH_DATE_RE.match(candidate):
            try:
                dt = datetime.strptime(candidate, "%d/%m/%Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return value

        return value

    def _normalize_structured_value(
        self,
        value: Any,
        schema_node: Any,
        root_schema: Dict[str, Any],
        stats: Dict[str, int],
        *,
        convert_null_strings: bool,
    ) -> Any:
        """Normalize values using schema hints (dates) and policy (null-string coercion)."""
        resolved_schema = self._resolve_schema_node(schema_node, root_schema)

        if isinstance(value, dict):
            properties = resolved_schema.get("properties", {}) if isinstance(resolved_schema, dict) else {}
            additional_properties = (
                resolved_schema.get("additionalProperties") if isinstance(resolved_schema, dict) else None
            )
            normalized = {}
            for key, item in value.items():
                child_schema = properties.get(key, {})
                if not child_schema and isinstance(additional_properties, dict):
                    child_schema = additional_properties
                normalized[key] = self._normalize_structured_value(
                    item,
                    child_schema,
                    root_schema,
                    stats,
                    convert_null_strings=convert_null_strings,
                )
            return normalized

        if isinstance(value, list):
            items_schema = resolved_schema.get("items", {}) if isinstance(resolved_schema, dict) else {}
            return [
                self._normalize_structured_value(
                    item,
                    items_schema,
                    root_schema,
                    stats,
                    convert_null_strings=convert_null_strings,
                )
                for item in value
            ]

        if isinstance(value, str):
            stripped = value.strip()
            if convert_null_strings and stripped.lower() == "null":
                stats["null_string_conversions"] += 1
                return None

            if self._schema_expects_date(resolved_schema):
                canonical = self._canonicalize_date_string(stripped)
                if canonical != value:
                    stats["date_format_conversions"] += 1
                return canonical

        return value

    def _compute_normalization_penalty(self, null_string_conversions: int) -> float:
        """Compute bounded score penalty for null-string coercions."""
        metrics_cfg = self.config.get("metrics", {}) if isinstance(self.config, dict) else {}
        penalty_cfg = metrics_cfg.get("normalization_penalty", {})
        if not isinstance(penalty_cfg, dict):
            penalty_cfg = {}

        per_conversion = float(penalty_cfg.get("null_string_to_null", 0.02))
        max_penalty = float(penalty_cfg.get("max", 0.20))
        penalty = max(0.0, null_string_conversions * max(0.0, per_conversion))
        return min(penalty, max(0.0, max_penalty))

    def calculate_all(
        self,
        prediction: Dict,
        expected: Dict,
        schema: Dict,
        error: str = None,
        error_type: str = None
    ) -> Dict:
        """Calculate all metrics for a single sample.

        Args:
            prediction: Model's predicted output
            expected: Expected (ground truth) output
            schema: Target JSON schema
            error: Error message if generation failed
            error_type: Type of error ('connectivity_error' or 'invalid_response')

        Returns:
            Dictionary of metric scores and diagnostics
        """
        metrics = {
            # Basic status
            "valid": False,
            "error": error,
            "error_type": error_type,
            # Tier 1: Overall Assessment
            "schema_validity": 0.0,
            "hallucination_rate": 0.0,
            # Tier 2: Extraction Accuracy
            "exact_match": False,
            "field_f1_partial": 0.0,
            "field_f1_strict": 0.0,
            "field_precision_partial": 0.0,
            "field_recall_partial": 0.0,
            "field_precision_strict": 0.0,
            "field_recall_strict": 0.0,
            "type_accuracy": 0.0,
            "document_extraction_score": 0.0,
            # Tier 3: Diagnostic
            "match_distribution": {
                "exact": 0,
                "partial": 0,
                "incorrect": 0,
                "missed": 0,
                "spurious": 0
            },
            "composite_scores": [],  # List of all composite scores for diagnostics
            # Schema complexity (for this sample)
            "schema_complexity": {},
            # Normalization diagnostics
            "normalization_penalty": 0.0,
            "null_string_conversions": 0,
            "date_format_conversions": 0,
        }

        # If connectivity error, mark differently and return early
        # (don't calculate quality metrics - these are network issues, not model quality issues)
        if error_type == "connectivity_error":
            metrics["connectivity_error"] = True
            return metrics

        # Check if generation had an error (invalid response or no prediction)
        if error or prediction is None:
            return metrics

        try:
            normalization_stats = {"null_string_conversions": 0, "date_format_conversions": 0}
            normalized_prediction = self._normalize_structured_value(
                prediction,
                schema,
                schema if isinstance(schema, dict) else {},
                normalization_stats,
                convert_null_strings=True,
            )
            normalized_expected = self._normalize_structured_value(
                expected,
                schema,
                schema if isinstance(schema, dict) else {},
                {"null_string_conversions": 0, "date_format_conversions": 0},
                convert_null_strings=False,
            )

            metrics["null_string_conversions"] = normalization_stats["null_string_conversions"]
            metrics["date_format_conversions"] = normalization_stats["date_format_conversions"]
            metrics["normalization_penalty"] = self._compute_normalization_penalty(
                normalization_stats["null_string_conversions"]
            )

            # Check schema validity
            metrics["valid"] = self._validate_schema(normalized_prediction, schema)
            metrics["schema_validity"] = 1.0 if metrics["valid"] else 0.0

            if not metrics["valid"]:
                logger.debug(f"Schema validation failed for prediction type: {type(prediction).__name__}")
                return metrics

            # Calculate schema complexity for this sample
            metrics["schema_complexity"] = self._calculate_schema_complexity(schema)

            # Flatten structures for field-level comparison
            try:
                pred_fields = self._flatten_dict(normalized_prediction)
                exp_fields = self._flatten_dict(normalized_expected)
            except Exception as e:
                logger.error(f"Error flattening dicts - prediction type: {type(prediction).__name__}, expected type: {type(expected).__name__}: {e}")
                metrics["error"] = f"Flattening error: {str(e)}"
                return metrics

            # Compare fields and gather statistics
            try:
                field_results = self._compare_fields(pred_fields, exp_fields)
            except Exception as e:
                logger.error(f"Error comparing fields: {e}")
                metrics["error"] = f"Field comparison error: {str(e)}"
                return metrics

            # Calculate metrics from field results
            try:
                metrics.update(self._aggregate_field_results(field_results, pred_fields, exp_fields))
            except Exception as e:
                logger.error(f"Error aggregating field results: {e}")
                metrics["error"] = f"Aggregation error: {str(e)}"
                return metrics

            # Check exact match
            try:
                metrics["exact_match"] = self._exact_match(normalized_prediction, normalized_expected)
            except Exception as e:
                logger.warning(f"Error checking exact match: {e}")
                metrics["exact_match"] = False

            # Calculate EQS (Extraction Quality Score)
            try:
                metrics["extraction_quality_score"] = self._calculate_eqs(metrics)
            except Exception as e:
                logger.error(f"Error calculating EQS: {e}")
                metrics["extraction_quality_score"] = 0.0

            # Calculate Document Extraction Score (DES) - numeric-critical scoring.
            try:
                metrics["document_extraction_score"] = self._calculate_document_extraction_score(metrics)
            except Exception as e:
                logger.error(f"Error calculating document_extraction_score: {e}")
                metrics["document_extraction_score"] = 0.0

            penalty = metrics.get("normalization_penalty", 0.0)
            if penalty > 0:
                metrics["extraction_quality_score"] = max(
                    0.0, float(metrics.get("extraction_quality_score", 0.0)) - penalty
                )
                metrics["document_extraction_score"] = max(
                    0.0, float(metrics.get("document_extraction_score", 0.0)) - penalty
                )

        except Exception as e:
            logger.error(f"Unexpected error in calculate_all: {e}", exc_info=True)
            metrics["error"] = f"Calculation error: {str(e)}"
            return metrics

        return metrics

    def _validate_schema(self, output: Any, schema: Dict) -> bool:
        """Validate output against JSON schema.

        Args:
            output: Output to validate (should be dict, but may be other types)
            schema: JSON schema for validation

        Returns:
            True if valid, False otherwise
        """
        # Quick type check - if output isn't a dict and schema expects object, it's invalid
        if not isinstance(output, dict):
            logger.debug(f"Output is {type(output).__name__}, not dict - invalid")
            return False

        # Check if schema is empty or missing
        if not schema or not schema.get("properties"):
            logger.warning("Schema is empty or has no properties, skipping validation")
            return True  # Consider valid if no schema to validate against

        try:
            validate(instance=output, schema=schema)
            return True
        except ValidationError as e:
            logger.warning(f"Schema validation failed: {e.message} (path: {list(e.absolute_path)})")
            return False
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            return False

    def _exact_match(self, prediction: Dict, expected: Dict) -> bool:
        """Check if prediction exactly matches expected output.

        Uses value-based comparison instead of JSON string comparison
        to handle float/int differences (1.0 vs 1) and key ordering.
        """
        return self._values_equal(prediction, expected)

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Recursively compare two values for equality.

        Handles:
        - Float/int equivalence (1.0 == 1)
        - Nested dicts and lists
        - String normalization for whitespace
        """
        # Handle None
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        # Handle numeric comparison (1.0 == 1)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Use tolerance for float comparison
            if isinstance(val1, float) or isinstance(val2, float):
                return abs(float(val1) - float(val2)) < 1e-9
            return val1 == val2

        # Handle string comparison (normalize whitespace and case)
        if isinstance(val1, str) and isinstance(val2, str):
            return val1.strip().upper() == val2.strip().upper()

        # Handle dict comparison
        if isinstance(val1, dict) and isinstance(val2, dict):
            if set(val1.keys()) != set(val2.keys()):
                return False
            return all(self._values_equal(val1[k], val2[k]) for k in val1.keys())

        # Handle list comparison
        if isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            return all(self._values_equal(v1, v2) for v1, v2 in zip(val1, val2))

        # Default comparison
        return val1 == val2

    def _compare_fields(
        self,
        pred_fields: Dict[str, Any],
        exp_fields: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Compare predicted and expected fields with partial matching.
        
        Returns:
            Dictionary mapping field keys to comparison results
        """
        ignored_fields = self.config.get("metrics", {}).get("ignored_fields", []) or []
        if isinstance(ignored_fields, str):
            ignored_fields = [ignored_fields]
        ignored_fields = [str(p) for p in ignored_fields if p]

        if ignored_fields:
            pred_fields = {k: v for k, v in pred_fields.items() if not _matches_any_pattern(k, ignored_fields)}
            exp_fields = {k: v for k, v in exp_fields.items() if not _matches_any_pattern(k, ignored_fields)}

        numeric_string_fields = self.config.get("metrics", {}).get("numeric_string_fields", []) or []
        if isinstance(numeric_string_fields, str):
            numeric_string_fields = [numeric_string_fields]
        numeric_string_fields = [str(p) for p in numeric_string_fields if p]

        critical_string_fields = self.config.get("metrics", {}).get("critical_string_fields", []) or []
        if isinstance(critical_string_fields, str):
            critical_string_fields = [critical_string_fields]
        critical_string_fields = [str(p) for p in critical_string_fields if p]

        results = {}

        # Compare expected fields
        for key, exp_value in exp_fields.items():
            if key in pred_fields:
                pred_value = pred_fields[key]
                if numeric_string_fields and _matches_any_pattern(key, numeric_string_fields):
                    pred_digits = _digits_only(pred_value)
                    exp_digits = _digits_only(exp_value)
                    if pred_digits and exp_digits and pred_digits == exp_digits:
                        match_type, score, severity = ("exact", 1.0, "none")
                    else:
                        match_type, score, severity = ("incorrect", 0.0, "critical")
                else:
                    match_type, score, severity = self.partial_matcher.compare_values_with_severity(pred_value, exp_value)
                    if (
                        critical_string_fields
                        and isinstance(exp_value, str)
                        and isinstance(pred_value, str)
                        and match_type != "exact"
                        and _matches_any_pattern(key, critical_string_fields)
                    ):
                        severity = "critical"
                type_match = type(pred_value) == type(exp_value)

                # Check if it's a numeric field
                is_numeric = isinstance(exp_value, (int, float)) and not isinstance(exp_value, bool)
                if numeric_string_fields and _matches_any_pattern(key, numeric_string_fields):
                    is_numeric = True

                results[key] = {
                    "status": "matched",
                    "match_type": match_type,
                    "score": score,
                    "type_match": type_match,
                    "error_severity": severity,
                    "is_numeric": is_numeric,
                    "predicted": pred_value,
                    "expected": exp_value,
                }
            else:
                results[key] = {
                    "status": "missed",
                    "match_type": "missed",
                    "score": 0.0,
                    "type_match": False,
                    "error_severity": "critical",  # Missed fields are critical
                    "is_numeric": isinstance(exp_value, (int, float)) and not isinstance(exp_value, bool),
                    "predicted": None,
                    "expected": exp_value,
                }
                if numeric_string_fields and _matches_any_pattern(key, numeric_string_fields):
                    results[key]["is_numeric"] = True

        # Find spurious fields (in prediction but not expected)
        for key, pred_value in pred_fields.items():
            if key not in exp_fields:
                results[key] = {
                    "status": "spurious",
                    "match_type": "spurious",
                    "score": 0.0,
                    "type_match": False,
                    "error_severity": "critical",  # Spurious fields are critical (hallucination)
                    "is_numeric": isinstance(pred_value, (int, float)) and not isinstance(pred_value, bool),
                    "predicted": pred_value,
                    "expected": None,
                }

        return results

    def _aggregate_field_results(
        self,
        field_results: Dict[str, Dict],
        pred_fields: Dict,
        exp_fields: Dict
    ) -> Dict:
        """Aggregate field-level results into metrics."""
        # Get configurable partial credit (default 0.3)
        partial_credit = self.config.get("metrics", {}).get("partial_credit", 0.3)

        # Counters for distribution
        exact_count = 0
        partial_count = 0
        incorrect_count = 0
        missed_count = 0
        spurious_count = 0
        type_correct_count = 0

        # New counters for enhanced metrics
        critical_errors = 0
        minor_errors = 0
        numeric_fields_total = 0
        numeric_fields_correct = 0

        # For composite score tracking
        composite_scores = []

        # Process each field result
        for field_key, result in field_results.items():
            match_type = result["match_type"]
            if match_type == "exact":
                exact_count += 1
            elif match_type == "partial":
                partial_count += 1
            elif match_type == "incorrect":
                incorrect_count += 1
            elif match_type == "missed":
                missed_count += 1
            elif match_type == "spurious":
                spurious_count += 1

            if result["type_match"]:
                type_correct_count += 1

            # Track composite scores
            if result["score"] > 0:
                composite_scores.append(result["score"])

            # Track error severity
            error_severity = result.get("error_severity", "none")
            if error_severity == "critical":
                critical_errors += 1
            elif error_severity == "minor":
                minor_errors += 1

            # Track numeric field accuracy
            if result.get("is_numeric", False):
                numeric_fields_total += 1
                if match_type == "exact":
                    numeric_fields_correct += 1

        # Calculate F1 scores
        total_expected = len(exp_fields)
        total_predicted = len(pred_fields)

        # Partial F1 (with configurable partial credit)
        correct_partial = exact_count + partial_credit * partial_count
        precision_partial = correct_partial / total_predicted if total_predicted > 0 else 0.0
        recall_partial = correct_partial / total_expected if total_expected > 0 else 0.0
        f1_partial = (2 * precision_partial * recall_partial /
                      (precision_partial + recall_partial)
                      if (precision_partial + recall_partial) > 0 else 0.0)

        # Strict F1 (exact matches only)
        precision_strict = exact_count / total_predicted if total_predicted > 0 else 0.0
        recall_strict = exact_count / total_expected if total_expected > 0 else 0.0
        f1_strict = (2 * precision_strict * recall_strict /
                     (precision_strict + recall_strict)
                     if (precision_strict + recall_strict) > 0 else 0.0)

        # Type accuracy
        type_accuracy = type_correct_count / total_expected if total_expected > 0 else 0.0

        # Hallucination rate
        hallucination_rate = spurious_count / total_predicted if total_predicted > 0 else 0.0

        # NEW METRICS
        # Critical Error Rate
        total_fields = total_expected  # Base on expected fields
        critical_error_rate = critical_errors / total_fields if total_fields > 0 else 0.0

        # Field Accuracy Score (strict, no partial credit)
        field_accuracy_score = exact_count / total_expected if total_expected > 0 else 0.0

        # Numeric Precision Rate
        numeric_precision_rate = numeric_fields_correct / numeric_fields_total if numeric_fields_total > 0 else 1.0

        return {
            "field_f1_partial": f1_partial,
            "field_f1_strict": f1_strict,
            "field_precision_partial": precision_partial,
            "field_recall_partial": recall_partial,
            "field_precision_strict": precision_strict,
            "field_recall_strict": recall_strict,
            "type_accuracy": type_accuracy,
            "hallucination_rate": hallucination_rate,
            "match_distribution": {
                "exact": exact_count,
                "partial": partial_count,
                "incorrect": incorrect_count,
                "missed": missed_count,
                "spurious": spurious_count,
            },
            "composite_scores": composite_scores,
            # New metrics
            "critical_error_rate": critical_error_rate,
            "minor_error_count": minor_errors,
            "critical_error_count": critical_errors,
            "field_accuracy_score": field_accuracy_score,
            "numeric_precision_rate": numeric_precision_rate,
            "numeric_fields_total": numeric_fields_total,
            "numeric_fields_correct": numeric_fields_correct,
        }

    def _calculate_eqs(self, metrics: Dict) -> float:
        """Calculate Extraction Quality Score (EQS).
        
        EQS = 0.15 * schema_validity +
              0.70 * field_f1_partial +
              0.15 * (1 - hallucination_rate)

        New weights emphasize F1 score more heavily to better differentiate
        high-performing models in the 0.9-1.0 range.
        """
        weights = self.config.get("metrics", {}).get("extraction_quality_score", {}).get("weights", {})
        validity_weight = weights.get("schema_validity", 0.15)
        f1_weight = weights.get("field_f1_partial", 0.70)
        anti_halluc_weight = weights.get("inverted_hallucination", 0.15)

        eqs = (validity_weight * metrics["schema_validity"] +
               f1_weight * metrics["field_f1_partial"] +
               anti_halluc_weight * (1 - metrics["hallucination_rate"]))

        return eqs

    def _calculate_document_extraction_score(self, metrics: Dict) -> float:
        """Calculate a numeric-critical document extraction score.

        This score is configurable and intended to emphasize numeric correctness
        (IDs, totals, tax amounts) while being tolerant of small string variations.

        Default:
          0.50 * numeric_precision_rate +
          0.35 * field_f1_partial +
          0.15 * schema_validity
        """
        weights = self.config.get("metrics", {}).get("document_extraction_score", {}).get("weights", {}) or {}

        numeric_weight = float(weights.get("numeric_precision_rate", weights.get("numeric_precision", 0.50)))
        f1_weight = float(weights.get("field_f1_partial", 0.35))
        validity_weight = float(weights.get("schema_validity", 0.15))
        anti_halluc_weight = float(weights.get("inverted_hallucination", 0.0))
        anti_critical_weight = float(weights.get("inverted_critical_error_rate", 0.0))

        score = (
            numeric_weight * float(metrics.get("numeric_precision_rate", 1.0))
            + f1_weight * float(metrics.get("field_f1_partial", 0.0))
            + validity_weight * float(metrics.get("schema_validity", 0.0))
        )

        if anti_halluc_weight:
            score += anti_halluc_weight * (1.0 - float(metrics.get("hallucination_rate", 0.0)))
        if anti_critical_weight:
            score += anti_critical_weight * (1.0 - float(metrics.get("critical_error_rate", 0.0)))

        return max(0.0, min(1.0, score))

    def _calculate_schema_complexity(self, schema: Dict) -> Dict:
        """Calculate complexity metrics for a schema."""
        def count_fields(obj, depth=0):
            """Recursively count fields and track max depth."""
            if not isinstance(obj, dict):
                return 0, depth

            total_fields = 0
            max_depth = depth
            if "properties" in obj:
                total_fields += len(obj["properties"])
                for prop_value in obj["properties"].values():
                    if isinstance(prop_value, dict):
                        sub_fields, sub_depth = count_fields(prop_value, depth + 1)
                        total_fields += sub_fields
                        max_depth = max(max_depth, sub_depth)

            # Check $defs
            if "$defs" in obj:
                for def_value in obj["$defs"].values():
                    sub_fields, sub_depth = count_fields(def_value, depth + 1)
                    total_fields += sub_fields
                    max_depth = max(max_depth, sub_depth)

            return total_fields, max_depth

        total_fields, max_depth = count_fields(schema)
        required_fields = 0
        if "required" in schema:
            required_fields = len(schema.get("required", []))

        return {
            "total_fields": total_fields,
            "required_fields": required_fields,
            "max_nesting_depth": max_depth,
            "has_arrays": "array" in str(schema),
            "has_nested_objects": "$defs" in schema or max_depth > 1,
        }

    def _flatten_dict(self, d: Any, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested dictionary, handling edge cases.

        Args:
            d: Input data (should be dict, but may be list or other types)
            parent_key: Parent key for nested flattening
            sep: Separator for key concatenation

        Returns:
            Flattened dictionary
        """
        # Handle non-dict inputs gracefully
        if d is None:
            return {}

        if not isinstance(d, dict):
            # If it's a list at the top level, try to convert to dict-like structure
            if isinstance(d, list):
                logger.warning("Received list instead of dict for flattening, wrapping as 'root' key")
                d = {"root": d}
            else:
                # For other types, wrap as a single value
                logger.warning(f"Received {type(d).__name__} instead of dict for flattening, wrapping as 'root' key")
                return {"root": d}

        items = []
        try:
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            items.extend(self._flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                        else:
                            items.append((f"{new_key}[{i}]", item))
                else:
                    items.append((new_key, v))
        except Exception as e:
            logger.error(f"Error during dict flattening: {e}, returning empty dict")
            return {}

        return dict(items)

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all samples.

        Args:
            all_metrics: List of per-sample metrics

        Returns:
            Dictionary of aggregated metrics
        """
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "response_rate": 0.0,
                "connectivity_error_rate": 0.0,
                "invalid_response_rate": 0.0,
                "extraction_quality_score": 0.0,
                "schema_validity_rate": 0.0,
                "hallucination_rate": 0.0,
                "exact_match_rate": 0.0,
                "field_f1_partial": 0.0,
                "field_f1_strict": 0.0,
                "field_precision_partial": 0.0,
                "field_recall_partial": 0.0,
                "field_precision_strict": 0.0,
                "field_recall_strict": 0.0,
                "type_accuracy": 0.0,
                "document_extraction_score": 0.0,
                "match_distribution": {},
                "match_distribution_counts": {},
                "composite_score_stats": {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0},
                "sample_level_variance": {
                    "eqs_mean": 0.0, "eqs_stdev": 0.0, "eqs_min": 0.0, "eqs_max": 0.0,
                    "f1_mean": 0.0, "f1_stdev": 0.0, "f1_min": 0.0, "f1_max": 0.0,
                },
                "normalization_penalty": 0.0,
                "null_string_conversions": 0,
                "date_format_conversions": 0,
            }

        total_samples = len(all_metrics)

        # Tier 1: Overall Assessment
        validity_rate = sum(m["valid"] for m in all_metrics) / total_samples
        avg_hallucination = sum(m["hallucination_rate"] for m in all_metrics) / total_samples

        # Tier 2: Extraction Accuracy
        exact_match_rate = sum(m["exact_match"] for m in all_metrics) / total_samples

        valid_metrics = [m for m in all_metrics if m["valid"]]
        if valid_metrics:
            avg_f1_partial = sum(m["field_f1_partial"] for m in valid_metrics) / len(valid_metrics)
            avg_f1_strict = sum(m["field_f1_strict"] for m in valid_metrics) / len(valid_metrics)
            avg_precision_partial = sum(m["field_precision_partial"] for m in valid_metrics) / len(valid_metrics)
            avg_recall_partial = sum(m["field_recall_partial"] for m in valid_metrics) / len(valid_metrics)
            avg_precision_strict = sum(m["field_precision_strict"] for m in valid_metrics) / len(valid_metrics)
            avg_recall_strict = sum(m["field_recall_strict"] for m in valid_metrics) / len(valid_metrics)
            avg_type_accuracy = sum(m["type_accuracy"] for m in valid_metrics) / len(valid_metrics)
            avg_eqs = sum(m["extraction_quality_score"] for m in valid_metrics) / len(valid_metrics)
            avg_des = sum(m.get("document_extraction_score", 0.0) for m in valid_metrics) / len(valid_metrics)

            # New metrics
            avg_critical_error_rate = sum(m.get("critical_error_rate", 0.0) for m in valid_metrics) / len(valid_metrics)
            avg_field_accuracy_score = sum(m.get("field_accuracy_score", 0.0) for m in valid_metrics) / len(valid_metrics)

            # Numeric precision (only count samples with numeric fields)
            metrics_with_numeric = [m for m in valid_metrics if m.get("numeric_fields_total", 0) > 0]
            if metrics_with_numeric:
                avg_numeric_precision = sum(m.get("numeric_precision_rate", 0.0) for m in metrics_with_numeric) / len(metrics_with_numeric)
                total_numeric_fields = sum(m.get("numeric_fields_total", 0) for m in valid_metrics)
                total_numeric_correct = sum(m.get("numeric_fields_correct", 0) for m in valid_metrics)
            else:
                avg_numeric_precision = 1.0  # No numeric fields means 100% precision
                total_numeric_fields = 0
                total_numeric_correct = 0
        else:
            avg_f1_partial = avg_f1_strict = 0.0
            avg_precision_partial = avg_recall_partial = 0.0
            avg_precision_strict = avg_recall_strict = 0.0
            avg_type_accuracy = avg_eqs = 0.0
            avg_des = 0.0
            avg_critical_error_rate = 0.0
            avg_field_accuracy_score = 0.0
            avg_numeric_precision = 0.0
            total_numeric_fields = 0
            total_numeric_correct = 0

        avg_normalization_penalty = (
            sum(float(m.get("normalization_penalty", 0.0)) for m in valid_metrics) / len(valid_metrics)
            if valid_metrics else 0.0
        )
        total_null_string_conversions = sum(int(m.get("null_string_conversions", 0)) for m in all_metrics)
        total_date_format_conversions = sum(int(m.get("date_format_conversions", 0)) for m in all_metrics)

        # Tier 3: Match Distribution (aggregate counts)
        total_dist = defaultdict(int)
        for m in valid_metrics:
            for key, value in m["match_distribution"].items():
                total_dist[key] += value

        # Convert to percentages
        total_fields = sum(total_dist.values())
        dist_percentages = {
            key: (value / total_fields * 100) if total_fields > 0 else 0.0
            for key, value in total_dist.items()
        }

        # Composite score statistics
        all_composite_scores = []
        for m in valid_metrics:
            all_composite_scores.extend(m.get("composite_scores", []))

        if all_composite_scores:
            import statistics
            composite_stats = {
                "mean": statistics.mean(all_composite_scores),
                "median": statistics.median(all_composite_scores),
                "stdev": statistics.stdev(all_composite_scores) if len(all_composite_scores) > 1 else 0.0,
                "min": min(all_composite_scores),
                "max": max(all_composite_scores),
            }
        else:
            composite_stats = {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}

        # Sample-level variance
        if valid_metrics:
            eqs_scores = [m["extraction_quality_score"] for m in valid_metrics]
            f1_scores = [m["field_f1_partial"] for m in valid_metrics]

            import statistics
            sample_variance = {
                "eqs_mean": statistics.mean(eqs_scores),
                "eqs_stdev": statistics.stdev(eqs_scores) if len(eqs_scores) > 1 else 0.0,
                "eqs_min": min(eqs_scores),
                "eqs_max": max(eqs_scores),
                "f1_mean": statistics.mean(f1_scores),
                "f1_stdev": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                "f1_min": min(f1_scores),
                "f1_max": max(f1_scores),
            }

            # NEW: Imperfect Sample Quality (ISQ)
            # Average quality of samples that are NOT exact matches
            imperfect_samples = [m for m in valid_metrics if not m.get("exact_match", False)]
            if imperfect_samples:
                imperfect_eqs_scores = [m["extraction_quality_score"] for m in imperfect_samples]
                imperfect_f1_scores = [m["field_f1_partial"] for m in imperfect_samples]
                imperfect_sample_quality = statistics.mean(imperfect_eqs_scores)
                imperfect_sample_quality_f1 = statistics.mean(imperfect_f1_scores)
                imperfect_count = len(imperfect_samples)
            else:
                imperfect_sample_quality = 1.0  # All samples are perfect
                imperfect_sample_quality_f1 = 1.0
                imperfect_count = 0
        else:
            sample_variance = {
                "eqs_mean": 0.0, "eqs_stdev": 0.0, "eqs_min": 0.0, "eqs_max": 0.0,
                "f1_mean": 0.0, "f1_stdev": 0.0, "f1_min": 0.0, "f1_max": 0.0,
            }
            imperfect_sample_quality = 0.0
            imperfect_sample_quality_f1 = 0.0
            imperfect_count = 0

        # Error count and classification
        error_count = sum(1 for m in all_metrics if m["error"] is not None)
        connectivity_errors = sum(1 for m in all_metrics if m.get("error_type") == "connectivity_error")
        invalid_responses = sum(1 for m in all_metrics if m.get("error_type") == "invalid_response")
        samples_with_response = total_samples - connectivity_errors

        # New metrics: response rate and error type rates
        response_rate = samples_with_response / total_samples if total_samples > 0 else 0.0
        connectivity_error_rate = connectivity_errors / total_samples if total_samples > 0 else 0.0
        invalid_response_rate = invalid_responses / total_samples if total_samples > 0 else 0.0

        # Calculate overall EQS accounting for invalid responses
        # Connectivity errors are excluded (they're infrastructure, not model quality)
        success_rate = 1.0 - invalid_response_rate
        overall_eqs = avg_eqs * success_rate

        return {
            # Sample counts
            "total_samples": total_samples,
            "valid_samples": len(valid_metrics),
            "error_count": error_count,
            "error_rate": error_count / total_samples,

            # Response and error type metrics
            "response_rate": response_rate,
            "connectivity_error_rate": connectivity_error_rate,
            "invalid_response_rate": invalid_response_rate,

            # Tier 1: Overall Assessment
            "extraction_quality_score": avg_eqs,
            "overall_extraction_quality_score": overall_eqs,
            "document_extraction_score": avg_des,
            "schema_validity_rate": validity_rate,
            "hallucination_rate": avg_hallucination,

            # Tier 2: Extraction Accuracy
            "exact_match_rate": exact_match_rate,
            "field_f1_partial": avg_f1_partial,
            "field_f1_strict": avg_f1_strict,
            "field_precision_partial": avg_precision_partial,
            "field_recall_partial": avg_recall_partial,
            "field_precision_strict": avg_precision_strict,
            "field_recall_strict": avg_recall_strict,
            "type_accuracy": avg_type_accuracy,

            # Tier 3: Diagnostic
            "match_distribution": dist_percentages,
            "match_distribution_counts": dict(total_dist),
            "composite_score_stats": composite_stats,
            "sample_level_variance": sample_variance,

            # NEW METRICS
            "critical_error_rate": avg_critical_error_rate,
            "field_accuracy_score": avg_field_accuracy_score,
            "numeric_precision_rate": avg_numeric_precision,
            "numeric_fields_total": total_numeric_fields,
            "numeric_fields_correct": total_numeric_correct,

            # Imperfect Sample Quality (ISQ)
            "imperfect_sample_quality": imperfect_sample_quality,
            "imperfect_sample_quality_f1": imperfect_sample_quality_f1,
            "imperfect_sample_count": imperfect_count,
            # Normalization diagnostics
            "normalization_penalty": avg_normalization_penalty,
            "null_string_conversions": total_null_string_conversions,
            "date_format_conversions": total_date_format_conversions,
        }
