"""Main metrics calculator implementing all metrics from the specification."""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from jsonschema import validate, ValidationError
from .partial_matching import PartialMatcher

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive metrics for structured data extraction."""

    def __init__(self, config: Dict):
        """Initialize metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.partial_matcher = PartialMatcher(config)

    def calculate_all(
        self,
        prediction: Dict,
        expected: Dict,
        schema: Dict,
        error: str = None
    ) -> Dict:
        """Calculate all metrics for a single sample.

        Args:
            prediction: Model's predicted output
            expected: Expected (ground truth) output
            schema: Target JSON schema
            error: Error message if generation failed

        Returns:
            Dictionary of metric scores and diagnostics
        """
        metrics = {
            # Basic status
            "valid": False,
            "error": error,

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
        }

        # Check if generation had an error
        if error or prediction is None:
            return metrics

        # Check schema validity
        metrics["valid"] = self._validate_schema(prediction, schema)
        metrics["schema_validity"] = 1.0 if metrics["valid"] else 0.0

        if not metrics["valid"]:
            return metrics

        # Calculate schema complexity for this sample
        metrics["schema_complexity"] = self._calculate_schema_complexity(schema)

        # Flatten structures for field-level comparison
        pred_fields = self._flatten_dict(prediction)
        exp_fields = self._flatten_dict(expected)

        # Compare fields and gather statistics
        field_results = self._compare_fields(pred_fields, exp_fields)

        # Calculate metrics from field results
        metrics.update(self._aggregate_field_results(field_results, pred_fields, exp_fields))

        # Check exact match
        metrics["exact_match"] = self._exact_match(prediction, expected)

        # Calculate EQS (Extraction Quality Score)
        metrics["extraction_quality_score"] = self._calculate_eqs(metrics)

        return metrics

    def _validate_schema(self, output: Dict, schema: Dict) -> bool:
        """Validate output against JSON schema."""
        try:
            validate(instance=output, schema=schema)
            return True
        except ValidationError as e:
            logger.debug(f"Schema validation failed: {e.message}")
            return False
        except Exception as e:
            logger.debug(f"Schema validation error: {e}")
            return False

    def _exact_match(self, prediction: Dict, expected: Dict) -> bool:
        """Check if prediction exactly matches expected output."""
        pred_str = json.dumps(prediction, sort_keys=True)
        exp_str = json.dumps(expected, sort_keys=True)
        return pred_str == exp_str

    def _compare_fields(
        self,
        pred_fields: Dict[str, Any],
        exp_fields: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Compare predicted and expected fields with partial matching.

        Returns:
            Dictionary mapping field keys to comparison results
        """
        results = {}

        # Compare expected fields
        for key, exp_value in exp_fields.items():
            if key in pred_fields:
                pred_value = pred_fields[key]
                match_type, score = self.partial_matcher.compare_values(pred_value, exp_value)
                type_match = type(pred_value) == type(exp_value)

                results[key] = {
                    "status": "matched",
                    "match_type": match_type,
                    "score": score,
                    "type_match": type_match,
                    "predicted": pred_value,
                    "expected": exp_value,
                }
            else:
                results[key] = {
                    "status": "missed",
                    "match_type": "missed",
                    "score": 0.0,
                    "type_match": False,
                    "predicted": None,
                    "expected": exp_value,
                }

        # Find spurious fields (in prediction but not expected)
        for key, pred_value in pred_fields.items():
            if key not in exp_fields:
                results[key] = {
                    "status": "spurious",
                    "match_type": "spurious",
                    "score": 0.0,
                    "type_match": False,
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
        # Counters for distribution
        exact_count = 0
        partial_count = 0
        incorrect_count = 0
        missed_count = 0
        spurious_count = 0
        type_correct_count = 0

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

        # Calculate F1 scores
        total_expected = len(exp_fields)
        total_predicted = len(pred_fields)

        # Partial F1 (with partial credit)
        correct_partial = exact_count + 0.5 * partial_count
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
        }

    def _calculate_eqs(self, metrics: Dict) -> float:
        """Calculate Extraction Quality Score (EQS).

        EQS = 0.20 * schema_validity +
              0.60 * field_f1_partial +
              0.20 * (1 - hallucination_rate)
        """
        weights = self.config.get("metrics", {}).get("extraction_quality_score", {}).get("weights", {})

        validity_weight = weights.get("schema_validity", 0.20)
        f1_weight = weights.get("field_f1_partial", 0.60)
        anti_halluc_weight = weights.get("inverted_hallucination", 0.20)

        eqs = (validity_weight * metrics["schema_validity"] +
               f1_weight * metrics["field_f1_partial"] +
               anti_halluc_weight * (1 - metrics["hallucination_rate"]))

        return eqs

    def _calculate_schema_complexity(self, schema: Dict) -> Dict:
        """Calculate complexity metrics for a schema."""
        def count_fields(obj, depth=0):
            """Recursively count fields and track max depth."""
            if not isinstance(obj, dict):
                return 0, depth

            total_fields = 0
            max_depth = depth
            has_arrays = False
            has_nested = False

            if "properties" in obj:
                total_fields += len(obj["properties"])
                for prop_value in obj["properties"].values():
                    if isinstance(prop_value, dict):
                        if prop_value.get("type") == "array":
                            has_arrays = True
                        if prop_value.get("type") == "object":
                            has_nested = True
                        sub_fields, sub_depth = count_fields(prop_value, depth + 1)
                        total_fields += sub_fields
                        max_depth = max(max_depth, sub_depth)

            # Check $defs
            if "$defs" in obj:
                for def_value in obj["$defs"].values():
                    sub_fields, sub_depth = count_fields(def_value, depth + 1)
                    total_fields += sub_fields
                    max_depth = max(max_depth, sub_depth)
                    has_nested = True

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

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested dictionary."""
        items = []
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
        return dict(items)

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all samples.

        Args:
            all_metrics: List of per-sample metrics

        Returns:
            Dictionary of aggregated metrics
        """
        if not all_metrics:
            return {}

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
        else:
            avg_f1_partial = avg_f1_strict = 0.0
            avg_precision_partial = avg_recall_partial = 0.0
            avg_precision_strict = avg_recall_strict = 0.0
            avg_type_accuracy = avg_eqs = 0.0

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
        else:
            sample_variance = {
                "eqs_mean": 0.0, "eqs_stdev": 0.0, "eqs_min": 0.0, "eqs_max": 0.0,
                "f1_mean": 0.0, "f1_stdev": 0.0, "f1_min": 0.0, "f1_max": 0.0,
            }

        # Error count
        error_count = sum(1 for m in all_metrics if m["error"] is not None)

        return {
            # Sample counts
            "total_samples": total_samples,
            "valid_samples": len(valid_metrics),
            "error_count": error_count,
            "error_rate": error_count / total_samples,

            # Tier 1: Overall Assessment
            "extraction_quality_score": avg_eqs,
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
        }









