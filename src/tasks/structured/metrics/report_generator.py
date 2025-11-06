"""Generate human-readable reports from metrics."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate text and HTML reports from benchmark results."""

    def __init__(self, config: Dict):
        """Initialize report generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def generate_text_report(
        self,
        model_name: str,
        aggregate_metrics: Dict,
        output_path: Path
    ) -> None:
        """Generate a text-based report card.

        Args:
            model_name: Name of the model
            aggregate_metrics: Aggregated metrics dictionary
            output_path: Path to save the report
        """
        lines = []

        # Header
        total_samples = aggregate_metrics.get('total_samples', 0)
        dataset_name = aggregate_metrics.get('dataset_name', 'structured_extraction')
        lines.append("┌" + "─" * 70 + "┐")
        lines.append(f"│  Model: {model_name:<60} │")
        dataset_line = f"Dataset: {dataset_name} ({total_samples} samples)"
        lines.append(f"│  {dataset_line:<68} │")
        lines.append(f"│  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<58} │")
        lines.append("└" + "─" * 70 + "┘")
        lines.append("")
        
        # Handle empty results gracefully
        if total_samples == 0:
            lines.append("⚠️  WARNING: No samples processed")
            lines.append("")
            with open(output_path, "w") as f:
                f.write("\n".join(lines))
            logger.warning(f"Generated report with no samples: {output_path}")
            return

        # Overall Score
        eqs = aggregate_metrics.get("extraction_quality_score", 0.0)
        eqs_grade = self._grade_eqs(eqs)
        lines.append("OVERALL SCORE")
        lines.append("━" * 72)
        lines.append(f"  Extraction Quality Score (EQS): {eqs:.3f}  {eqs_grade}")
        
        # Confidence interval (approximation based on variance)
        eqs_std = aggregate_metrics.get("sample_level_variance", {}).get("eqs_stdev", 0.0)
        if total_samples > 0:
            ci_lower = max(0.0, eqs - 1.96 * eqs_std / (total_samples ** 0.5))
            ci_upper = min(1.0, eqs + 1.96 * eqs_std / (total_samples ** 0.5))
            lines.append(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        lines.append("")

        # Primary Metrics (NEW LAYOUT)
        lines.append("PRIMARY METRICS")
        lines.append("━" * 72)
        
        # Question 1: How many are perfect?
        exact_match = aggregate_metrics.get("exact_match_rate", 0.0)
        em_count = int(exact_match * aggregate_metrics.get("total_samples", 0))
        total_count = aggregate_metrics.get("total_samples", 0)
        lines.append(f"  {self._emoji(exact_match)} Perfect Extractions:        {exact_match:5.1%}  ({em_count}/{total_count})  Grade: {self._grade_metric(exact_match)}")
        
        # Question 2: How bad are the imperfect ones?
        imperfect_quality = aggregate_metrics.get("imperfect_sample_quality", 0.0)
        imperfect_count = aggregate_metrics.get("imperfect_sample_count", 0)
        if imperfect_count > 0:
            lines.append(f"  {self._emoji(imperfect_quality)} Imperfect Sample Quality:   {imperfect_quality:.3f}             Grade: {self._grade_metric(imperfect_quality)}")
        else:
            lines.append(f"  {self._emoji(1.0)} Imperfect Sample Quality:   N/A (all perfect)    Grade: A+")
        
        # Overall EQS for model comparison
        lines.append(f"  ⭐ Overall Quality (EQS):      {eqs:.3f}             {eqs_grade}")
        lines.append("")
        
        # Additional Key Metrics
        lines.append("ADDITIONAL KEY METRICS")
        lines.append("━" * 72)
        
        validity = aggregate_metrics.get("schema_validity_rate", 0.0)
        valid_count = aggregate_metrics.get("valid_samples", 0)
        lines.append(f"  {self._emoji(validity)} Schema Validity:     {validity:5.1%}  ({valid_count}/{total_count})  Grade: {self._grade_metric(validity)}")
        
        f1_partial = aggregate_metrics.get("field_f1_partial", 0.0)
        lines.append(f"  {self._emoji(f1_partial)} Field F1 (Partial):  {f1_partial:.3f}             Grade: {self._grade_metric(f1_partial)}")
        
        halluc = aggregate_metrics.get("hallucination_rate", 0.0)
        lines.append(f"  {self._emoji(1 - halluc)} Hallucination Rate:  {halluc:5.1%}              Grade: {self._grade_metric(1 - halluc)}")
        lines.append("")

        # Detailed Accuracy
        lines.append("DETAILED ACCURACY")
        lines.append("━" * 72)
        f1_strict = aggregate_metrics.get("field_f1_strict", 0.0)
        precision = aggregate_metrics.get("field_precision_partial", 0.0)
        recall = aggregate_metrics.get("field_recall_partial", 0.0)
        type_acc = aggregate_metrics.get("type_accuracy", 0.0)
        field_accuracy = aggregate_metrics.get("field_accuracy_score", 0.0)
        
        lines.append(f"  Field F1 (Strict):       {f1_strict:.3f}")
        lines.append(f"  Field Accuracy Score:    {field_accuracy:.3f}  (exact matches only, no partial credit)")
        lines.append(f"  Field Precision:         {precision:.3f}")
        lines.append(f"  Field Recall:            {recall:.3f}")
        lines.append(f"  Type Accuracy:           {type_acc:5.1%}")
        lines.append("")
        
        # ERROR ANALYSIS (NEW SECTION)
        lines.append("ERROR ANALYSIS")
        lines.append("━" * 72)
        
        critical_error_rate = aggregate_metrics.get("critical_error_rate", 0.0)
        lines.append(f"  Critical Error Rate:     {critical_error_rate:5.1%}  {self._emoji(1 - critical_error_rate)}")
        lines.append(f"    (wrong numerics, missing fields, type mismatches)")
        
        # Numeric field analysis
        numeric_total = aggregate_metrics.get("numeric_fields_total", 0)
        numeric_correct = aggregate_metrics.get("numeric_fields_correct", 0)
        numeric_precision = aggregate_metrics.get("numeric_precision_rate", 0.0)
        
        if numeric_total > 0:
            lines.append(f"  Numeric Field Precision: {numeric_precision:5.1%}  ({numeric_correct}/{numeric_total})  {self._emoji(numeric_precision)}")
        else:
            lines.append(f"  Numeric Field Precision: N/A (no numeric fields in dataset)")
        
        lines.append("")

        # Match Distribution
        lines.append("MATCH DISTRIBUTION")
        lines.append("━" * 72)
        dist = aggregate_metrics.get("match_distribution", {})
        
        for match_type, percentage in [
            ("Exact", dist.get("exact", 0.0)),
            ("Partial", dist.get("partial", 0.0)),
            ("Incorrect", dist.get("incorrect", 0.0)),
            ("Missed", dist.get("missed", 0.0)),
            ("Spurious", dist.get("spurious", 0.0)),
        ]:
            bar = self._make_bar(percentage, max_width=40)
            lines.append(f"  {match_type:10s} {percentage:5.1f}%  {bar}")
        lines.append("")

        # Composite Score Distribution
        comp_stats = aggregate_metrics.get("composite_score_stats", {})
        if comp_stats.get("mean", 0) > 0:
            lines.append("COMPOSITE SCORE STATISTICS")
            lines.append("━" * 72)
            lines.append(f"  Mean:   {comp_stats.get('mean', 0.0):.3f}")
            lines.append(f"  Median: {comp_stats.get('median', 0.0):.3f}")
            lines.append(f"  StdDev: {comp_stats.get('stdev', 0.0):.3f}")
            lines.append(f"  Range:  [{comp_stats.get('min', 0.0):.3f}, {comp_stats.get('max', 0.0):.3f}]")
            lines.append("")

        # Sample-Level Variance
        variance = aggregate_metrics.get("sample_level_variance", {})
        lines.append("SAMPLE-LEVEL CONSISTENCY")
        lines.append("━" * 72)
        lines.append(f"  EQS:  μ={variance.get('eqs_mean', 0.0):.3f}, "
                    f"σ={variance.get('eqs_stdev', 0.0):.3f}, "
                    f"range=[{variance.get('eqs_min', 0.0):.3f}, {variance.get('eqs_max', 0.0):.3f}]")
        lines.append(f"  F1:   μ={variance.get('f1_mean', 0.0):.3f}, "
                    f"σ={variance.get('f1_stdev', 0.0):.3f}, "
                    f"range=[{variance.get('f1_min', 0.0):.3f}, {variance.get('f1_max', 0.0):.3f}]")
        
        consistency_grade = self._grade_consistency(variance.get('eqs_stdev', 0.0))
        lines.append(f"  Consistency: {consistency_grade}")
        lines.append("")

        # Schema Complexity Analysis
        complexity = aggregate_metrics.get("complexity_analysis", {})
        if complexity:
            lines.append("SCHEMA COMPLEXITY ANALYSIS")
            lines.append("━" * 72)
            
            # Overall complexity distribution
            bins = complexity.get("bins", {})
            if bins:
                lines.append("  Distribution by Complexity:")
                for bin_name in ["simple", "medium", "complex"]:
                    bin_data = bins.get(bin_name, {})
                    count = bin_data.get("count", 0)
                    if count > 0:
                        eqs_bin = bin_data.get("eqs", 0.0)
                        f1_bin = bin_data.get("f1_partial", 0.0)
                        halluc_bin = bin_data.get("hallucination_rate", 0.0)
                        lines.append(f"    {bin_name.capitalize():8s} ({count:3d}): "
                                   f"EQS={eqs_bin:.3f}, F1={f1_bin:.3f}, Halluc={halluc_bin:5.1%}")
                lines.append("")
            
            # Correlations
            corr = complexity.get("correlations", {})
            if corr:
                lines.append("  Correlations with Performance:")
                for feature, r_value in corr.items():
                    strength = self._correlation_strength(r_value)
                    direction = "negative" if r_value < 0 else "positive"
                    lines.append(f"    {feature:25s} r={r_value:+.3f} ({strength} {direction})")
                lines.append("")
        
        # Production Readiness
        lines.append("PRODUCTION READINESS")
        lines.append("━" * 72)
        
        prod_ready = self._assess_production_readiness(aggregate_metrics)
        tier = self._determine_tier(eqs, f1_partial, halluc)
        
        lines.append(f"  Quality Tier: {tier}")
        lines.append(f"  Production Ready: {'✅ YES' if prod_ready else '⚠️  CAUTION'}")
        lines.append("")
        
        if prod_ready:
            lines.append("  Recommendation: DEPLOY WITH CONFIDENCE")
            lines.append("  Model demonstrates strong extraction capability across")
            lines.append("  the benchmark. Suitable for production use.")
        elif eqs >= 0.65:
            lines.append("  Recommendation: DEPLOY WITH MONITORING")
            lines.append("  Model shows acceptable performance but may struggle with")
            lines.append("  complex schemas. Recommend human review for critical extractions.")
        else:
            lines.append("  Recommendation: NOT RECOMMENDED FOR PRODUCTION")
            lines.append("  Model shows significant limitations in structured extraction.")
            lines.append("  Consider fine-tuning or using a more capable model.")
        lines.append("")

        # Performance metrics if available
        if "throughput" in aggregate_metrics:
            lines.append("PERFORMANCE")
            lines.append("━" * 72)
            throughput = aggregate_metrics.get("throughput", 0.0)
            duration = aggregate_metrics.get("total_duration", 0.0)
            lines.append(f"  Throughput:         {throughput:.2f} samples/second")
            lines.append(f"  Total Duration:     {duration:.2f} seconds")
            lines.append(f"  Avg per Sample:     {1/throughput if throughput > 0 else 0:.2f} seconds")
            lines.append("")

        # Write to file
        report_text = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(report_text)
        
        logger.info(f"Generated text report: {output_path}")

    def _grade_eqs(self, eqs: float) -> str:
        """Grade the EQS score."""
        if eqs >= 0.85:
            return "⭐ EXCELLENT"
        elif eqs >= 0.75:
            return "✅ GOOD"
        elif eqs >= 0.65:
            return "⚠️  MODERATE"
        else:
            return "❌ POOR"

    def _grade_metric(self, value: float) -> str:
        """Grade a metric value."""
        if value >= 0.95:
            return "A+"
        elif value >= 0.90:
            return "A"
        elif value >= 0.85:
            return "A-"
        elif value >= 0.80:
            return "B+"
        elif value >= 0.75:
            return "B"
        elif value >= 0.70:
            return "B-"
        elif value >= 0.65:
            return "C+"
        elif value >= 0.60:
            return "C"
        else:
            return "D"

    def _grade_consistency(self, stdev: float) -> str:
        """Grade consistency based on standard deviation."""
        if stdev < 0.05:
            return "Excellent (very consistent)"
        elif stdev < 0.10:
            return "Good (mostly consistent)"
        elif stdev < 0.15:
            return "Moderate (some variance)"
        else:
            return "Poor (highly variable)"

    def _emoji(self, value: float) -> str:
        """Get emoji indicator for a metric."""
        if value >= 0.85:
            return "✅"
        elif value >= 0.70:
            return "⚠️ "
        else:
            return "❌"

    def _make_bar(self, percentage: float, max_width: int = 40) -> str:
        """Create a text-based bar chart."""
        filled = int(percentage / 100.0 * max_width)
        return "█" * filled

    def _assess_production_readiness(self, metrics: Dict) -> bool:
        """Assess if model is production ready."""
        eqs = metrics.get("extraction_quality_score", 0.0)
        validity = metrics.get("schema_validity_rate", 0.0)
        halluc = metrics.get("hallucination_rate", 0.0)
        f1 = metrics.get("field_f1_partial", 0.0)

        return (
            eqs >= 0.75 and
            validity >= 0.95 and
            halluc <= 0.10 and
            f1 >= 0.70
        )

    def _determine_tier(self, eqs: float, f1: float, halluc: float) -> str:
        """Determine quality tier."""
        if eqs >= 0.85 and f1 >= 0.85 and halluc <= 0.05:
            return "Tier 1 (Excellent)"
        elif eqs >= 0.75 and f1 >= 0.75 and halluc <= 0.10:
            return "Tier 2 (Good)"
        elif eqs >= 0.65 and f1 >= 0.65 and halluc <= 0.15:
            return "Tier 3 (Moderate)"
        else:
            return "Tier 4 (Poor)"

    def _correlation_strength(self, r: float) -> str:
        """Classify correlation strength."""
        abs_r = abs(r)
        if abs_r < 0.20:
            return "negligible"
        elif abs_r < 0.40:
            return "weak"
        elif abs_r < 0.60:
            return "moderate"
        elif abs_r < 0.80:
            return "strong"
        else:
            return "very strong"

