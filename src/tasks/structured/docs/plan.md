### 4.5 Metric Summary and Reporting

#### 4.5.1 Primary Metrics Dashboard

**Headline Metrics** (displayed prominently):
```python
headline_metrics = {
    'extraction_quality_score': 0.0-1.0,     # PRIMARY METRIC
    'schema_validity_rate': 0-100%,
    'field_f1_partial': 0.0-1.0,
    'exact_match_rate': 0-100%,
    'hallucination_rate': 0-100%
}
```

#### 4.5.2 Complete Metrics Set

**Tier 1: Summary Metrics**
1. **Extraction Quality Score (EQS)** - Composite overall score
2. **Schema Validity Rate** - Can the model produce valid JSON?
3. **Field F1 (Partial)** - Primary accuracy metric with partial credit
4. **Exact Match Rate (Strict)** - Percentage of perfect extractions

**Tier 2: Core Accuracy Metrics** (Partial Mode)
5. **Field Precision (Partial)** - Reliability of predictions
6. **Field Recall (Partial)** - Completeness of extractions
7. **Field F1 (Strict)** - Accuracy without partial credit
8. **Field F1 (Lenient)** - Accuracy with maximum partial credit
9. **Type Accuracy** - Percentage of correct data types

**Tier 3: Diagnostic Accuracy Metrics**
10. **Match Category Distribution** - Breakdown of exact/partial/incorrect/missed/spurious
11. **Composite Score Distribution** - Quality distribution of partial matches
12. **Hallucination Rate** - Spurious field percentage
13. **Hallucination Severity Breakdown** - Critical/moderate/minor
14. **Field Omission Rate** - Percentage of missed expected fields
15. **Omission Analysis by Field Type** - Which fields are missed most

**Tier 4: Performance Metrics**
16. **Mean Latency** - Average response time
17. **P95 Latency** - 95th percentile response time
18. **P99 Latency** - 99th percentile response time
19. **Throughput (RPS)** - Requests per second
20. **Token Efficiency** - Fields per output token
21. **Success Rate** - Percentage of successful completions

**Tier 5: Advanced Diagnostics**
22. **String Similarity Distribution** - Token overlap / Levenshtein / Containment stats
23. **Schema Complexity Correlation** - Performance vs complexity
24. **Error Pattern Analysis** - Categorized error types
25. **Field-Level Metrics by Type** - Precision/Recall broken down by data type
26. **Field-Level Metrics by Nesting** - Precision/Recall broken down by depth
27. **Required vs Optional Field Performance** - Comparison of required/optional field accuracy

**Optional Advanced Metrics**
28. **Confidence Calibration** - If model provides confidence scores
29. **Output Consistency** - For non-deterministic settings (temperature > 0)
30. **Cost Estimation** - Based on token usage

#### 4.5.3 Metric Aggregation Strategies

**Per-Sample Metrics**:
- Individual EQS score
- Field-level match results
- Individual latency
- Error categorization

**Dataset-Level Aggregation**:
```python
aggregation_methods = {
    'micro': 'Aggregate all fields across all samples, then compute',
    'macro': 'Compute per-sample metric, then average',
    'weighted': 'Weight by number of expected fields per sample'
}
```

**Recommended Aggregation**:
- **Micro** for Precision, Recall, F1 (gives equal weight to each field)
- **Macro** for per-sample EQS (gives equal weight to each sample)
- **Percentiles** for latency (p50, p95, p99)
- **Mean** for rates (validity, hallucination, etc.)

#### 4.5.4 Comparison Metrics (Multi-Model)

When comparing multiple models:

```python
comparison_metrics = {
    'eqs_ranking': ranked_list_of_models,
    'eqs_delta': pairwise_differences,
    'statistical_significance': {
        'model_a_vs_model_b': {
            'p_value': float,
            'effect_size': float,
            'significantly_different': bool
        }
    },
    'pareto_frontier': {
        'accuracy_vs_latency': list_of_models_on_frontier,
        'accuracy_vs_cost': list_of_models_on_frontier
    },
    'best_by_metric': {
        'highest_eqs': model_name,
        'highest_f1_partial': model_name,
        'highest_exact_match': model_name,
        'lowest_hallucination': model_name,
        'fastest_p95': model_name,
        'best_cost_performance': model_name
    },
    'win_rate_matrix': {
        # For each pair of models, percentage of samples where model_a > model_b
        'model_a_vs_model_b': win_rate_percentage
    }
}
```

#### 4.5.5 Statistical Significance Testing

For comparing models, use appropriate statistical tests:

```python
from scipy import stats

def compare_models(model_a_results, model_b_results, metric='f1_partial'):
    """
    Compare two models on a specific metric
    """
    # Get per-sample scores
    scores_a = [sample[metric] for sample in model_a_results]
    scores_b = [sample[metric] for sample in model_b_results]
    
    # Paired t-test (same samples for both models)
    t_statistic, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(scores_a) - np.mean(scores_b)
    pooled_std = np.sqrt((np.std(scores_a)**2 + np.std(scores_b)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_statistic, wilcoxon_p = stats.wilcoxon(scores_a, scores_b)
    
    return {
        'metric': metric,
        'mean_a': np.mean(scores_a),
        'mean_b': np.mean(scores_b),
        'mean_difference': mean_diff,
        't_test': {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': interpret_cohens_d(cohens_d)
        },
        'wilcoxon_test': {
            'statistic': wilcoxon_statistic,
            'p_value': wilcoxon_p
        }
    }

def interpret_cohens_d(d):
    """Cohen's d interpretation"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'
```

#### 4.5.6 Confidence Intervals

Report metrics with confidence intervals for reliability:

```python
def compute_confidence_interval(scores, confidence=0.95):
    """
    Bootstrap confidence interval for a metric
    """
    n_bootstrap = 10000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    
    return {
        'mean': np.mean(scores),
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence
    }
```

**Reporting Format**:
```
Extraction Quality Score: 0.823 [95% CI: 0.812, 0.834]
Field F1 (Partial): 0.856 [95% CI: 0.844, 0.868]
```

#### 4.5.7 Metric Visualization Guidelines

**Required Visualizations**:

1. **Overview Dashboard**:
   - EQS gauge chart (0-1 scale with colored zones)
   - 4-metric radar chart (validity, F1, type accuracy, 1-hallucination)
   - Summary statistics table

2. **Accuracy Breakdown**:
   - Stacked bar chart: exact/partial/incorrect/missed/spurious distribution
   - F1 scores comparison: strict/partial/lenient
   - Precision-Recall curve or scatter plot

3. **Performance Analysis**:
   - Latency histogram with percentile markers
   - Latency vs accuracy scatter plot
   - Throughput over time line chart

4. **Error Analysis**:
   - Error pattern pie chart or tree map
   - Field omission heatmap by field name
   - Hallucination severity breakdown

5. **Complexity Analysis**:
   - F1 vs schema complexity scatter plot
   - Box plots: F1 by complexity bin
   - Correlation heatmap: metrics vs complexity features

6. **Multi-Model Comparison**:
   - Grouped bar chart: key metrics across models
   - Pareto frontier: accuracy vs latency
   - Win rate matrix heatmap
   - Statistical significance indicators

#### 4.5.8 Report Structure

**Executive Summary Section**:
```markdown
# Model Evaluation Report: {model_name}

## Executive Summary

- **Overall Score (EQS)**: 0.823 ⭐ (Excellent - Production Ready)
- **Evaluation Date**: 2025-10-02
- **Dataset**: paraloq/json_data_extraction (1,000 samples)
- **Key Strengths**: High precision (0.89), low hallucination (3.2%)
- **Key Weaknesses**: Struggles with deeply nested objects (F1: 0.67 at depth 3+)
- **Recommendation**: Deploy with confidence for schemas with nesting depth ≤ 2

### At a Glance
| Metric | Value | Grade |
|--------|-------|-------|
| Schema Validity | 98.7% | A+ |
| Field F1 (Partial) | 0.856 | A |
| Exact Match | 67.3% | B+ |
| Hallucination Rate | 3.2% | A+ |
| P95 Latency | 1.2s | A |
```

**Detailed Sections**:
1. Methodology
2. Primary Metrics with visualizations
3. Accuracy Deep Dive
4. Performance Analysis
5. Error Pattern Analysis
6. Schema Complexity Analysis
7. Sample-Level Results (best/worst cases)
8. Recommendations and Next Steps

#### 4.5.9 Metric Selection Guidelines

**For Different Use Cases**:

**Production Deployment Decision**:
- Primary: EQS, Schema Validity, Hallucination Rate
- Secondary: P95 Latency, Success Rate
- Tertiary: Cost per sample

**Model Selection/Comparison**:
- Primary: EQS, Field F1 (Partial)
- Secondary: Field F1 (Strict), Type Accuracy
- Tertiary: Statistical significance tests

**Debugging/Improvement**:
- Primary: Error Pattern Analysis, Omission Analysis
- Secondary: Match Category Distribution, Complexity Correlation
- Tertiary: String Similarity Distribution

**Research/Publication**:
- Primary: Field F1 (Strict, Partial, Lenient)
- Secondary: Precision, Recall, Type Accuracy
- Tertiary: Full breakdown by field type, nesting level

**Cost Optimization**:
- Primary: Token Efficiency, Cost per sample
- Secondary: Throughput, Success Rate
- Tertiary: Accuracy vs Cost tradeoff

#### 4.5.10 Metric Thresholds and SLAs

**Recommended Production Thresholds**:

```python
production_requirements = {
    'minimum': {
        'eqs': 0.75,
        'schema_validity': 0.95,
        'field_f1_partial': 0.70,
        'hallucination_rate_max': 0.10,
        'p95_latency_max': 5000,  # ms
        'success_rate': 0.99
    },
    'target': {
        'eqs': 0.85,
        'schema_validity': 0.98,
        'field_f1_partial': 0.80,
        'hallucination_rate_max': 0.05,
        'p95_latency_max': 2000,
        'success_rate': 0.995
    },
    'excellence': {
        'eqs': 0.90,
        'schema_validity': 0.99,
        'field_f1_partial': 0.90,
        'hallucination_rate_max': 0.02,
        'p95_latency_max': 1000,
        'success_rate': 0.999
    }
}
```

**Evaluation Decision Tree**:
```
if eqs >= 0.90 and hallucination_rate < 0.02:
    recommendation = "Deploy without human review"
elif eqs >= 0.80 and hallucination_rate < 0.05:
    recommendation = "Deploy with spot-check human review"
elif eqs >= 0.70 and hallucination_rate < 0.10:
    recommendation = "Deploy with mandatory human review"
else:
    recommendation = "Not recommended for production - needs improvement"
```# LLM Structured Data Extraction Benchmark - Technical Specification

## 1. Project Overview

### 1.1 Purpose
Build a comprehensive benchmark system to evaluate Large Language Models (LLMs) on structured data extraction tasks. The system will measure how accurately LLMs can extract information from unstructured text into predefined JSON schemas.

### 1.2 Scope
- Evaluate multiple LLM models served via local vLLM server
- Use the paraloq/json_data_extraction dataset from HuggingFace
- Implement multiple evaluation metrics for comprehensive assessment
- Generate detailed reports with per-model and per-sample analytics

### 1.3 Key Requirements
- Support vLLM's structured output capabilities (guided_json parameter)
- Calculate extraction-specific metrics (not just schema validation)
- Handle various JSON schema complexities
- Produce reproducible, comparable results across models
- Efficient processing of large evaluation datasets

---

## 2. Data Architecture

### 2.1 Input Dataset
**Source**: `paraloq/json_data_extraction` from HuggingFace

**Expected Schema**:
```python
{
    "text": str,              # Input paragraph/document
    "schema": dict,           # Target JSON schema
    "expected_output": dict   # Ground truth extraction
}
```

### 2.2 Data Preprocessing
- Load dataset using `datasets` library
- Validate schema format (JSON Schema Draft-07 or later)
- Validate expected outputs against schemas
- Handle missing or malformed entries
- Support sampling for development/testing

### 2.3 Data Splits
- Use all available data for evaluation (no train/test split needed)
- Support configurable subsampling for quick testing
- Track data statistics (schema complexity, field counts, nesting depth)

---

## 3. Model Interface

### 3.1 vLLM Integration
**Connection Method**: OpenAI-compatible API endpoint

**Configuration**:
```python
{
    "base_url": str,           # e.g., "http://localhost:8000/v1"
    "api_key": str,            # API key if required
    "model_name": str,         # Model identifier
    "temperature": float,      # Default: 0.0 for deterministic output
    "max_tokens": int,         # Default: 2048
    "timeout": int             # Request timeout in seconds
}
```

### 3.2 Structured Output Request
**Method**: Use vLLM's `guided_json` parameter or OpenAI's structured output API

```python
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "extraction_result",
            "schema": target_schema,
            "strict": True
        }
    },
    temperature=0.0,
    max_tokens=max_tokens
)
```

### 3.3 Prompt Engineering
**System Prompt Template**:
```
You are a precise data extraction assistant. Extract information from the provided text according to the given JSON schema. Only extract information explicitly stated in the text. If information for a field is not present, omit the field or use null as appropriate.
```

**User Prompt Template**:
```
Text:
{text}

Extract information according to this JSON schema:
{schema}

Output valid JSON matching the schema exactly.
```

### 3.4 Error Handling
- Retry logic for transient failures (max 3 retries with exponential backoff)
- Timeout handling (configurable per request)
- Invalid JSON parsing recovery
- Schema validation failure logging
- Rate limiting support

---

## 4. Evaluation Metrics

### 4.0 Overall Summary Metric

#### 4.0.1 Extraction Quality Score (EQS)
**Definition**: Composite metric that summarizes overall extraction performance across all dimensions

**Calculation**:
```python
EQS = (w1 * Schema_Validity) + 
      (w2 * Field_F1_Partial) + 
      (w3 * Type_Accuracy) + 
      (w4 * (1 - Hallucination_Rate))

# Default weights (configurable):
w1 = 0.15  # Schema validity (must produce valid JSON)
w2 = 0.50  # Field-level accuracy (most important)
w3 = 0.20  # Type correctness (important for downstream use)
w4 = 0.15  # Penalty for hallucinations (reliability)
```

**Interpretation**:
- EQS ≥ 0.90: Excellent - Production-ready for most use cases
- EQS 0.75-0.90: Good - Suitable with human review
- EQS 0.60-0.75: Moderate - Needs significant oversight
- EQS < 0.60: Poor - Not recommended for production

**Reporting**:
- Report EQS as primary headline metric
- Always report with component breakdown
- Include confidence intervals when available
- Compare across models using EQS as primary ranking

### 4.1 Primary Metrics

#### 4.1.1 Schema Validity Rate
**Definition**: Percentage of outputs that are valid JSON conforming to the target schema

**Calculation**:
```python
validity_rate = (valid_outputs / total_outputs) * 100
```

**Implementation**:
- Use `jsonschema` library for validation
- Log validation errors for debugging
- Track by schema complexity level
- Invalid outputs receive 0 for all other metrics

**Importance**: Foundational metric - invalid outputs are unusable

#### 4.1.2 Exact Match (EM) - Strict Mode
**Definition**: Percentage of valid outputs that exactly match the expected output

**Calculation**:
```python
exact_match_strict = (strict_exact_matches / valid_outputs) * 100
```

**Matching Rules**:
- All fields must match exactly (value and type)
- String comparison: case-sensitive by default (configurable)
- Whitespace: normalized (configurable)
- Floating-point: epsilon comparison (configurable, default 1e-6)
- Arrays: order matters by default (configurable)
- Missing fields: treated as non-match

**Importance**: Strictest measure - indicates perfect extractions

#### 4.1.3 Field-Level F1 Score (Multiple Modes)

**Definition**: Harmonic mean of precision and recall at the field level, calculated under different matching strategies

**Mode A: Strict F1** (Exact matching only)
```python
# Per sample
correct_fields = fields_with_exact_match
precision_strict = correct_fields / predicted_fields
recall_strict = correct_fields / expected_fields
f1_strict = 2 * (precision_strict * recall_strict) / (precision_strict + recall_strict)

# Aggregate
micro_f1_strict = aggregate_across_all_fields()
macro_f1_strict = mean(per_sample_f1_scores)
```

**Mode B: Partial F1** (With partial credit)
```python
# Per sample - using SemEval methodology
correct_fields = exact_matches
partial_fields = partial_matches (composite_score >= 0.5)
incorrect_fields = non_matches
missed_fields = expected_but_not_predicted
spurious_fields = predicted_but_not_expected

# Precision and Recall with partial credit
precision_partial = (correct + 0.5 * partial) / (correct + partial + incorrect + spurious)
recall_partial = (correct + 0.5 * partial) / (correct + partial + incorrect + missed)
f1_partial = 2 * (precision_partial * recall_partial) / (precision_partial + recall_partial)

# Aggregate
micro_f1_partial = aggregate_across_all_fields()
macro_f1_partial = mean(per_sample_f1_scores)
```

**Mode C: Lenient F1** (Maximum credit)
```python
# Per sample
correct_fields = exact_matches
partial_fields = partial_matches (composite_score >= 0.3)
incorrect_fields = non_matches
missed_fields = expected_but_not_predicted
spurious_fields = predicted_but_not_expected

# Full credit for any match
precision_lenient = (correct + partial) / (correct + partial + incorrect + spurious)
recall_lenient = (correct + partial) / (correct + partial + incorrect + missed)
f1_lenient = 2 * (precision_lenient * recall_lenient) / (precision_lenient + recall_lenient)
```

**Primary Recommendation**: Use **Partial F1** as the main metric, report all three modes

**Field Matching Strategies**:

1. **Exact Match**: Identical values after normalization
2. **Partial Match**: Composite score from multiple strategies:

```python
def composite_partial_score(predicted, expected, field_type):
    if field_type == "string":
        # Token Overlap F1
        pred_tokens = set(predicted.lower().split())
        exp_tokens = set(expected.lower().split())
        intersection = pred_tokens & exp_tokens
        
        token_precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        token_recall = len(intersection) / len(exp_tokens) if exp_tokens else 0
        token_f1 = 2 * (token_precision * token_recall) / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0
        
        # Normalized Levenshtein Distance
        from Levenshtein import distance
        max_len = max(len(predicted), len(expected))
        lev_score = 1 - (distance(predicted.lower(), expected.lower()) / max_len) if max_len > 0 else 0
        
        # Containment Score
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        if exp_lower in pred_lower:
            containment = 1.0
        elif pred_lower in exp_lower:
            containment = len(pred_lower) / len(exp_lower)
        else:
            containment = 0.0
        
        # Weighted composite
        composite = (0.5 * token_f1 + 0.3 * lev_score + 0.2 * containment)
        
        return composite
    
    elif field_type == "number":
        # Relative difference for numbers
        if expected == 0:
            return 1.0 if predicted == 0 else 0.0
        rel_diff = abs(predicted - expected) / abs(expected)
        return max(0, 1 - rel_diff)
    
    elif field_type == "boolean":
        # Boolean must be exact
        return 1.0 if predicted == expected else 0.0
    
    elif field_type == "array":
        # Jaccard similarity for arrays
        pred_set = set(map(str, predicted))
        exp_set = set(map(str, expected))
        intersection = pred_set & exp_set
        union = pred_set | exp_set
        return len(intersection) / len(union) if union else 0.0
    
    elif field_type == "object":
        # Recursive field matching for nested objects
        return compute_nested_field_match(predicted, expected)
```

**Match Classification**:
- **Exact**: composite_score >= 0.95
- **Partial**: 0.5 <= composite_score < 0.95
- **Incorrect**: 0 < composite_score < 0.5
- **Missing**: field in expected but not in predicted
- **Spurious**: field in predicted but not in expected

**Importance**: Core metric for extraction accuracy - balances precision and recall with nuanced scoring

### 4.2 Secondary Metrics

#### 4.2.1 Field Precision (Multiple Modes)

**Definition**: Ratio of correctly extracted fields to all extracted fields

**Strict Precision**:
```python
precision_strict = exact_matches / total_predicted_fields
```

**Partial Precision** (Recommended):
```python
precision_partial = (exact_matches + 0.5 * partial_matches) / total_predicted_fields
```

**Lenient Precision**:
```python
precision_lenient = (exact_matches + partial_matches) / total_predicted_fields
```

**Breakdown By**:
- Overall precision
- Per-field-type (string, number, boolean, object, array, null)
- Per-nesting-level (depth 0, depth 1, depth 2+)
- Per-field-name (for common fields across samples)
- Per-schema-complexity-bin (simple, medium, complex)

**Importance**: Indicates how reliable predictions are - high precision means few false positives

#### 4.2.2 Field Recall (Multiple Modes)

**Definition**: Ratio of correctly extracted fields to all expected fields

**Strict Recall**:
```python
recall_strict = exact_matches / total_expected_fields
```

**Partial Recall** (Recommended):
```python
recall_partial = (exact_matches + 0.5 * partial_matches) / total_expected_fields
```

**Lenient Recall**:
```python
recall_lenient = (exact_matches + partial_matches) / total_expected_fields
```

**Breakdown By**:
- Overall recall
- Per-field-type
- Per-nesting-level
- Per-field-name
- Per-schema-complexity-bin
- Per-required-vs-optional fields

**Importance**: Indicates completeness - high recall means few missed fields

#### 4.2.3 Type Accuracy

**Definition**: Percentage of extracted fields with correct data types (regardless of value correctness)

**Calculation**:
```python
type_accuracy = fields_with_correct_type / total_extracted_fields * 100
```

**Type Checking**:
- Primitive types: string, number (int/float), boolean, null
- Complex types: object, array
- Schema-specific: enum values, format constraints
- Type coercion awareness: "123" vs 123 (configurable strictness)

**Breakdown By**:
- Per expected type
- Per schema complexity
- Confusion matrix (expected type vs predicted type)

**Importance**: Type errors break downstream processing - critical for API/database integration

#### 4.2.4 Match Category Distribution

**Definition**: Breakdown of how fields were classified

**Categories** (per SemEval methodology):
```python
categories = {
    'exact': count_of_exact_matches,
    'partial': count_of_partial_matches,
    'incorrect': count_of_incorrect_matches,
    'missed': count_of_expected_but_missing,
    'spurious': count_of_unexpected_predictions
}

# As percentages
total = sum(categories.values())
distribution = {k: (v / total * 100) for k, v in categories.items()}
```

**Importance**: Diagnostic metric - shows where the model struggles

#### 4.2.5 Field-Level Metrics by Composite Score

**Definition**: Distribution of composite partial match scores for matched fields

**Calculation**:
```python
# For all partially matched fields
score_bins = {
    'excellent': [0.95, 1.0],   # Nearly perfect
    'good': [0.80, 0.95],        # Minor differences
    'fair': [0.60, 0.80],        # Noticeable differences
    'poor': [0.40, 0.60],        # Significant differences
    'very_poor': [0.0, 0.40]     # Major differences
}

# Count fields in each bin
distribution = compute_score_distribution(all_partial_matches, score_bins)
```

**Importance**: Shows quality distribution of partial matches - helps tune partial credit thresholds

### 4.3 Performance Metrics

#### 4.3.1 Latency

**Measurements**:
- Mean latency (milliseconds)
- Median latency (p50)
- 95th percentile latency (p95)
- 99th percentile latency (p99)
- Min/Max latency

**Tracking**:
- Total request time (end-to-end)
- Breakdown: network + processing + generation
- Correlation with input length (characters/tokens)
- Correlation with schema complexity (field count, nesting depth)
- Correlation with output length

**Percentile-based SLA metrics**:
```python
sla_compliance = {
    'p95_under_2s': p95_latency < 2000,  # 95% under 2 seconds
    'p99_under_5s': p99_latency < 5000,  # 99% under 5 seconds
}
```

**Importance**: Critical for production deployment - impacts user experience

#### 4.3.2 Throughput

**Measurements**:
- Requests per second (RPS)
- Samples per minute
- Tokens per second (input + output combined)
- Fields extracted per second

**Calculations**:
```python
throughput_rps = total_successful_requests / total_time_seconds
throughput_tokens = (total_input_tokens + total_output_tokens) / total_time_seconds
```

**Importance**: Resource utilization and cost estimation

#### 4.3.3 Token Statistics

**Measurements**:
- Mean/median input tokens
- Mean/median output tokens
- Total tokens consumed
- Token efficiency: extracted_fields / output_tokens

**Cost Estimation**:
```python
# Configurable pricing per token
estimated_cost = (input_tokens * input_price_per_1k / 1000) + 
                 (output_tokens * output_price_per_1k / 1000)
```

**Importance**: Budget planning and efficiency optimization

#### 4.3.4 Reliability Metrics

**Measurements**:
- Success rate: percentage of requests that completed
- Parse failure rate: valid HTTP response but unparseable JSON
- Schema validation failure rate: parseable JSON but schema invalid
- Timeout rate: requests exceeding timeout threshold
- Retry rate: requests requiring retries

**Calculation**:
```python
success_rate = successful_requests / total_requests * 100
reliability_score = (successful_requests + 0.5 * retried_successes) / total_requests
```

**Importance**: System robustness - critical for production reliability

### 4.4 Tertiary Metrics (Diagnostic)

#### 4.4.1 String Similarity Distribution

**Definition**: For string fields, distribution of similarity scores across different strategies

**Measurements**:
```python
string_similarities = {
    'token_overlap_f1': {
        'mean': mean_token_f1,
        'median': median_token_f1,
        'distribution': histogram_bins
    },
    'levenshtein_similarity': {
        'mean': mean_lev_score,
        'median': median_lev_score,
        'distribution': histogram_bins
    },
    'containment_score': {
        'mean': mean_containment,
        'median': median_containment,
        'distribution': histogram_bins
    }
}
```

**Importance**: Understanding which partial matching strategy is most relevant for your data

#### 4.4.2 Hallucination Rate

**Definition**: Percentage of extracted fields not present in ground truth

**Calculation**:
```python
hallucination_rate = spurious_fields / total_extracted_fields * 100
```

**Categories**:
- **Type 1**: Completely fabricated values (field name + value both wrong)
- **Type 2**: Correct field name, fabricated value
- **Type 3**: Incorrect field name, extracted value exists elsewhere in text
- **Type 4**: Excessive array elements beyond expected

**Severity Scoring**:
```python
hallucination_severity = {
    'critical': count_fabricated_values,      # Type 1 + Type 2
    'moderate': count_misplaced_values,       # Type 3
    'minor': count_excessive_array_elements   # Type 4
}
```

**Importance**: Trust and reliability - hallucinations are dangerous in production

#### 4.4.3 Field Omission Analysis

**Definition**: Breakdown of which expected fields are most frequently missed

**Measurements**:
```python
omission_analysis = {
    'by_field_name': {
        'field_name': {
            'omission_rate': missed_count / total_occurrences,
            'total_occurrences': count
        }
    },
    'by_field_type': {
        'string': omission_rate,
        'number': omission_rate,
        ...
    },
    'by_required_vs_optional': {
        'required': omission_rate,
        'optional': omission_rate
    },
    'by_nesting_depth': {
        0: omission_rate,
        1: omission_rate,
        2: omission_rate
    }
}
```

**Importance**: Identifies systematic weaknesses - guides prompt engineering

#### 4.4.4 Schema Complexity Correlation

**Definition**: How metric performance varies with schema characteristics

**Schema Complexity Dimensions**:
```python
complexity_features = {
    'total_fields': int,
    'required_fields': int,
    'optional_fields': int,
    'max_nesting_depth': int,
    'has_arrays': bool,
    'has_nested_objects': bool,
    'total_possible_paths': int,  # Flattened field count
    'field_types_diversity': int,  # Number of distinct types
    'enum_fields_count': int,
    'format_constrained_fields': int
}
```

**Correlation Analysis**:
```python
correlations = {
    'f1_vs_total_fields': pearson_correlation,
    'f1_vs_nesting_depth': pearson_correlation,
    'f1_vs_required_fields': pearson_correlation,
    'latency_vs_total_fields': pearson_correlation,
    'latency_vs_max_nesting': pearson_correlation
}
```

**Binned Performance**:
```python
# Group schemas by complexity
complexity_bins = {
    'simple': complexity_score < 0.33,    # <5 fields, depth <=1
    'medium': 0.33 <= complexity_score < 0.67,  # 5-15 fields, depth <=2
    'complex': complexity_score >= 0.67   # >15 fields or depth >2
}

# Report metrics per bin
for bin_name, samples in complexity_bins.items():
    bin_metrics = compute_metrics(samples)
```

**Importance**: Understanding performance boundaries - capacity planning

#### 4.4.5 Error Pattern Analysis

**Definition**: Systematic categorization of extraction errors

**Error Categories**:
```python
error_patterns = {
    'extraction_errors': {
        'wrong_value': count,           # Value exists but incorrect
        'missing_value': count,         # Should extract but didn't
        'hallucinated_value': count,    # Extracted non-existent value
        'partial_value': count          # Extracted subset/superset
    },
    'type_errors': {
        'wrong_type': count,            # e.g., string instead of number
        'type_coercion_failed': count   # e.g., "N/A" instead of null
    },
    'structure_errors': {
        'wrong_nesting': count,         # Field at wrong depth
        'array_vs_object': count,       # Confused array with object
        'missing_required_field': count,
        'extra_field': count
    },
    'boundary_errors': {
        'incomplete_extraction': count,  # "San Francisco" instead of "San Francisco, CA"
        'over_extraction': count         # "San Francisco, CA, USA" instead of "San Francisco, CA"
    }
}
```

**Most Common Errors**:
```python
# Top 10 most common field-level errors
top_errors = sorted(all_errors, key=lambda x: x['frequency'], reverse=True)[:10]
```

**Importance**: Actionable insights for model improvement and prompt engineering

#### 4.4.6 Confidence Calibration (Optional)

**Definition**: If model provides confidence scores, measure calibration quality

**Measurements**:
```python
# Group predictions by confidence bins
confidence_bins = [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]

for bin in confidence_bins:
    predictions_in_bin = filter_by_confidence(predictions, bin)
    actual_accuracy = compute_accuracy(predictions_in_bin)
    expected_confidence = bin.midpoint
    calibration_error = abs(actual_accuracy - expected_confidence)
```

**Expected Calibration Error (ECE)**:
```python
ECE = sum(|accuracy(bin) - confidence(bin)| * weight(bin) for bin in bins)
```

**Importance**: If model provides confidence, validates trustworthiness of confidence scores

#### 4.4.7 Consistency Metrics (Optional)

**Definition**: If running with temperature > 0, measure output consistency

**Measurements**:
```python
# Run same input N times (e.g., N=5)
consistency_metrics = {
    'exact_match_consistency': (identical_outputs / total_runs) * 100,
    'field_level_consistency': mean_field_agreement_across_runs,
    'value_variance': std_dev_of_field_values_across_runs
}
```

**Importance**: For non-deterministic settings, understand output stability

---

## 5. System Architecture

### 5.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Benchmark System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Dataset    │      │    Model     │      │  Metrics  │ │
│  │   Loader     │─────▶│  Interface   │─────▶│ Calculator│ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │    Config    │      │   Result     │      │  Report   │ │
│  │   Manager    │      │   Storage    │      │ Generator │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Module Specifications

#### 5.2.1 Dataset Loader (`dataset_loader.py`)
**Responsibilities**:
- Load data from HuggingFace
- Validate data integrity
- Apply sampling/filtering
- Provide iteration interface

**Key Classes**:
```python
class DatasetLoader:
    def __init__(self, dataset_name: str, split: str, cache_dir: str)
    def load(self) -> Dataset
    def sample(self, n: int, seed: int) -> Dataset
    def validate_schema(self, sample: dict) -> bool
    def get_statistics(self) -> dict
```

#### 5.2.2 Model Interface (`model_interface.py`)
**Responsibilities**:
- Connect to vLLM server
- Format prompts
- Request structured outputs
- Handle errors and retries

**Key Classes**:
```python
class VLLMInterface:
    def __init__(self, config: dict)
    def generate_structured(self, text: str, schema: dict) -> dict
    def batch_generate(self, inputs: List[dict]) -> List[dict]
    def test_connection(self) -> bool
    
class PromptBuilder:
    def __init__(self, system_template: str, user_template: str)
    def build_messages(self, text: str, schema: dict) -> List[dict]
```

#### 5.2.3 Metrics Calculator (`metrics_calculator.py`)
**Responsibilities**:
- Implement all metric calculations
- Normalize and compare outputs
- Aggregate results

**Key Classes**:
```python
class MetricsCalculator:
    def calculate_all(self, prediction: dict, expected: dict, schema: dict) -> dict
    
class SchemaValidator:
    def validate(self, output: dict, schema: dict) -> Tuple[bool, List[str]]
    
class FieldMatcher:
    def match_fields(self, pred: dict, expected: dict) -> dict
    def fuzzy_match_string(self, s1: str, s2: str, threshold: float) -> bool
    
class PerformanceTracker:
    def record_latency(self, duration: float)
    def get_percentiles(self, percentiles: List[int]) -> dict
```

#### 5.2.4 Result Storage (`result_storage.py`)
**Responsibilities**:
- Store per-sample results
- Store aggregate metrics
- Support multiple output formats
- Enable result querying

**Key Classes**:
```python
class ResultStore:
    def __init__(self, output_dir: str)
    def save_sample_result(self, sample_id: str, result: dict)
    def save_aggregate_metrics(self, metrics: dict)
    def load_results(self, run_id: str) -> dict
    
class ResultFormatter:
    def to_json(self, results: dict) -> str
    def to_csv(self, results: dict) -> str
    def to_dataframe(self, results: dict) -> pd.DataFrame
```

#### 5.2.5 Report Generator (`report_generator.py`)
**Responsibilities**:
- Generate HTML/Markdown reports
- Create visualizations
- Compare multiple runs
- Highlight failure cases

**Key Classes**:
```python
class ReportGenerator:
    def generate_html_report(self, results: dict, output_path: str)
    def generate_comparison_report(self, results_list: List[dict], output_path: str)
    
class Visualizer:
    def plot_metric_distribution(self, metrics: dict) -> Figure
    def plot_latency_histogram(self, latencies: List[float]) -> Figure
    def plot_confusion_matrix(self, results: dict) -> Figure
```

#### 5.2.6 Config Manager (`config_manager.py`)
**Responsibilities**:
- Load configuration files
- Validate configuration
- Provide configuration access

**Key Classes**:
```python
class ConfigManager:
    def __init__(self, config_path: str)
    def load_config(self) -> dict
    def validate_config(self) -> bool
    def get(self, key: str, default: Any) -> Any
```

### 5.3 Configuration File Structure

**YAML Configuration** (`config.yaml`):
```yaml
# Model Configuration
model:
  base_url: "http://localhost:8000/v1"
  api_key: null
  model_name: "meta-llama/Llama-3-8B-Instruct"
  temperature: 0.0
  max_tokens: 2048
  timeout: 60
  max_retries: 3

# Dataset Configuration
dataset:
  name: "paraloq/json_data_extraction"
  split: "test"
  cache_dir: "./cache"
  sample_size: null  # null = use all data
  random_seed: 42

# Prompt Configuration
prompts:
  system: |
    You are a precise data extraction assistant. Extract information from the provided text according to the given JSON schema. Only extract information explicitly stated in the text. If information for a field is not present, omit the field or use null as appropriate.
  user: |
    Text:
    {text}

    Extract information according to this JSON schema:
    {schema}

    Output valid JSON matching the schema exactly.

# Metrics Configuration
metrics:
  enabled:
    - schema_validity
    - exact_match
    - field_f1
    - field_precision
    - field_recall
    - type_accuracy
    - latency
  
  field_matching:
    string_fuzzy_threshold: 0.85
    numeric_tolerance: 0.001
    case_sensitive: false
    normalize_whitespace: true
  
  aggregation:
    f1_type: "micro"  # micro or macro

# Output Configuration
output:
  results_dir: "./results"
  save_predictions: true
  save_per_sample: true
  generate_report: true
  report_format: "html"  # html or markdown

# Performance Configuration
performance:
  batch_size: 1  # vLLM typically processes one at a time
  num_workers: 1
  enable_caching: true
  cache_dir: "./cache"
```

---

## 6. Implementation Plan

### 6.1 Phase 1: Core Infrastructure (Week 1)
**Tasks**:
1. Set up project structure and dependencies
2. Implement `DatasetLoader` class
3. Implement `VLLMInterface` class
4. Implement `ConfigManager` class
5. Create basic CLI interface
6. Write unit tests for core components

**Deliverables**:
- Functional data loading from HuggingFace
- Working connection to vLLM server
- Configuration file parsing
- Basic test suite

### 6.2 Phase 2: Metrics Implementation (Week 2)
**Tasks**:
1. Implement `SchemaValidator`
2. Implement `FieldMatcher` with exact matching
3. Implement primary metrics (validity, EM, F1)
4. Implement secondary metrics (precision, recall, type accuracy)
5. Implement performance tracking
6. Write comprehensive metric tests

**Deliverables**:
- Complete metrics calculation pipeline
- Validated metric implementations
- Performance benchmarks for metric calculation

### 6.3 Phase 3: Evaluation Pipeline (Week 2-3)
**Tasks**:
1. Implement main evaluation loop
2. Add batch processing support
3. Implement result storage
4. Add progress tracking and logging
5. Implement error recovery
6. Add intermediate checkpointing

**Deliverables**:
- End-to-end evaluation pipeline
- Robust error handling
- Resume capability for long-running evaluations

### 6.4 Phase 4: Reporting & Visualization (Week 3)
**Tasks**:
1. Implement `ReportGenerator`
2. Create HTML report templates
3. Implement visualizations (metrics distributions, latency plots)
4. Add comparison report for multiple models
5. Create example failure case viewer
6. Generate documentation

**Deliverables**:
- Professional HTML reports
- Comparative analysis tools
- User documentation

### 6.5 Phase 5: Testing & Optimization (Week 4)
**Tasks**:
1. End-to-end system testing
2. Performance optimization
3. Memory profiling and optimization
4. Documentation finalization
5. Create example notebooks
6. Prepare release

**Deliverables**:
- Tested, production-ready system
- Complete documentation
- Example usage notebooks
- Performance benchmarks

---

## 7. Technical Dependencies

### 7.1 Core Libraries
```
# LLM Interface
openai>=1.0.0              # For vLLM OpenAI-compatible API
aiohttp>=3.9.0             # For async HTTP requests

# Data Processing
datasets>=2.14.0           # HuggingFace datasets
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical operations

# Validation & Metrics
jsonschema>=4.19.0         # JSON Schema validation
python-Levenshtein>=0.21.0 # String similarity
scikit-learn>=1.3.0        # Additional metrics

# Configuration
pyyaml>=6.0                # YAML config parsing
python-dotenv>=1.0.0       # Environment variables

# Reporting
jinja2>=3.1.0              # HTML template rendering
matplotlib>=3.7.0          # Plotting
seaborn>=0.12.0            # Statistical visualization
plotly>=5.17.0             # Interactive plots

# CLI & Utilities
click>=8.1.0               # CLI framework
tqdm>=4.66.0               # Progress bars
rich>=13.5.0               # Rich terminal output
loguru>=0.7.0              # Logging
```

### 7.2 Development Dependencies
```
pytest>=7.4.0              # Testing framework
pytest-cov>=4.1.0          # Coverage reporting
pytest-asyncio>=0.21.0     # Async testing
black>=23.7.0              # Code formatting
ruff>=0.0.287              # Linting
mypy>=1.5.0                # Type checking
pre-commit>=3.3.0          # Git hooks
```

### 7.3 Optional Dependencies
```
# For advanced visualizations
jupyterlab>=4.0.0          # Interactive development
ipywidgets>=8.1.0          # Interactive widgets

# For distributed evaluation
ray>=2.7.0                 # Distributed computing

# For experiment tracking
wandb>=0.15.0              # Experiment tracking
mlflow>=2.7.0              # ML lifecycle management
```

---

## 8. Testing Strategy

### 8.1 Unit Tests
**Coverage Target**: >90%

**Test Categories**:
- Data loading and validation
- Schema validation logic
- Field matching algorithms
- Metric calculations
- Configuration parsing
- Prompt building

### 8.2 Integration Tests
**Test Scenarios**:
- End-to-end evaluation pipeline
- vLLM connection and structured output
- Result storage and retrieval
- Report generation

### 8.3 Validation Tests
**Approach**:
- Create synthetic test cases with known correct answers
- Verify metric calculations against hand-computed values
- Test edge cases (empty outputs, missing fields, extra fields)
- Validate against subset of dataset with manual review

---

## 9. Performance Considerations

### 9.1 Optimization Strategies
1. **Batch Processing**: Process multiple samples concurrently where possible
2. **Caching**: Cache dataset and intermediate results
3. **Efficient Serialization**: Use efficient JSON parsing libraries
4. **Memory Management**: Stream results to disk for large evaluations
5. **Async I/O**: Use async HTTP for vLLM requests

### 9.2 Scalability
**Expected Performance**:
- Target: 1000 samples/hour (depending on model speed)
- Memory: <4GB for 10K sample evaluation
- Storage: ~100MB per 1K samples with full results

---

## 10. Deliverables

### 10.1 Code Deliverables
1. Complete Python package with modular architecture
2. CLI tool for running evaluations
3. Configuration templates
4. Example scripts and notebooks

### 10.2 Documentation Deliverables
1. README with quickstart guide
2. API documentation
3. Metrics documentation (how each is calculated)
4. Configuration guide
5. Troubleshooting guide

### 10.3 Report Deliverables
1. HTML evaluation reports
2. CSV exports of metrics
3. Comparative analysis reports
4. Failure case analysis

---

## 11. Success Criteria

### 11.1 Functional Requirements
- ✅ Successfully load and process HuggingFace dataset
- ✅ Connect to vLLM and request structured outputs
- ✅ Calculate all specified metrics accurately
- ✅ Generate comprehensive HTML reports
- ✅ Support multiple model evaluations

### 11.2 Non-Functional Requirements
- ✅ Process ≥1000 samples/hour
- ✅ <0.1% failure rate due to system errors
- ✅ Memory usage <4GB for typical evaluations
- ✅ Code coverage >90%
- ✅ Clear error messages and logging

### 11.3 Usability Requirements
- ✅ Single command evaluation run
- ✅ Clear progress indicators
- ✅ Intuitive configuration
- ✅ Professional, readable reports
- ✅ Easy result comparison across models

---

## 12. Future Enhancements

### 12.1 Short-term (3-6 months)
- Support for additional datasets
- Multi-language support
- Advanced fuzzy matching algorithms
- Real-time evaluation dashboard
- Integration with experiment tracking platforms

### 12.2 Long-term (6-12 months)
- Distributed evaluation across multiple GPUs
- Active learning for identifying difficult samples
- Automatic prompt optimization
- Cost-performance trade-off analysis
- Integration with fine-tuning pipelines

---

## 13. Risk Management

### 13.1 Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| vLLM API changes | High | Low | Abstract API interface, version pinning |
| Dataset format changes | Medium | Low | Robust validation, flexible parsers |
| Metric calculation bugs | High | Medium | Extensive testing, manual validation |
| Performance bottlenecks | Medium | Medium | Profiling, optimization passes |
| Memory issues with large datasets | Medium | Medium | Streaming, chunking, caching strategy |

### 13.2 Project Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Scope creep | Medium | High | Clear requirements, phased approach |
| Timeline overrun | Medium | Medium | Buffer time, MVP-first approach |
| Insufficient testing | High | Medium | Test-driven development, CI/CD |

---

## Appendix A: Example Data Samples

### Sample 1: Simple Extraction
```python
{
    "text": "John Smith is 35 years old and works as a software engineer at TechCorp.",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
            "company": {"type": "string"}
        },
        "required": ["name"]
    },
    "expected_output": {
        "name": "John Smith",
        "age": 35,
        "occupation": "software engineer",
        "company": "TechCorp"
    }
}
```

### Sample 2: Nested Structure
```python
{
    "text": "Dr. Sarah Johnson, contactable at sarah.j@hospital.org or +1-555-0123, specializes in cardiology at Metro General Hospital.",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "title": {"type": "string"},
            "contact": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "phone": {"type": "string"}
                }
            },
            "specialization": {"type": "string"},
            "workplace": {"type": "string"}
        }
    },
    "expected_output": {
        "name": "Sarah Johnson",
        "title": "Dr.",
        "contact": {
            "email": "sarah.j@hospital.org",
            "phone": "+1-555-0123"
        },
        "specialization": "cardiology",
        "workplace": "Metro General Hospital"
    }
}
```

---

## Appendix B: CLI Commands

```bash
# Basic evaluation
python -m benchmark run --model llama-3-8b --config config.yaml

# With sampling
python -m benchmark run --model llama-3-8b --sample 100 --seed 42

# Multiple models comparison
python -m benchmark run --models llama-3-8b,mistral-7b --config config.yaml

# Generate report from existing results
python -m benchmark report --results-dir ./results/run_001 --output report.html

# Compare multiple runs
python -m benchmark compare --runs run_001,run_002,run_003 --output comparison.html

# Validate configuration
python -m benchmark validate-config --config config.yaml

# Test vLLM connection
python -m benchmark test-connection --base-url http://localhost:8000/v1
```

---

## Appendix C: Report Structure

### HTML Report Sections
1. **Executive Summary**
   - Overall scores
   - Model comparison table
   - Key findings

2. **Detailed Metrics**
   - Primary metrics with visualizations
   - Secondary metrics breakdown
   - Performance metrics

3. **Analysis**
   - Metric correlations
   - Schema complexity analysis
   - Error pattern analysis

4. **Sample Results**
   - Top performing samples
   - Worst performing samples
   - Interesting edge cases

5. **Recommendations**
   - Model selection guidance
   - Areas for improvement
   - Configuration suggestions