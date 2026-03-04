"""Freeform text format handler.

This handler encapsulates logic for freeform text generation tasks including:
- Exact match scoring
- F1 score calculation
- BLEU score (optional)
- Text normalization

## CLI Usage

Create freeform generation tasks directly from the CLI:

```bash
# Question answering
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type freeform \
  --dataset-name squad \
  --dataset-split validation \
  --dataset-input-field question \
  --dataset-output-field answer \
  --user-prompt-template "Context: {context}\n\nQuestion: {question}\n\nAnswer:" \
  --limit 10

# Summarization
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type freeform \
  --dataset-name cnn_dailymail \
  --dataset-input-field article \
  --dataset-output-field summary \
  --system-prompt "Summarize the following article concisely." \
  --limit 10

# Local JSONL dataset
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type freeform \
  --dataset-name ./data/qa_pairs.jsonl \
  --dataset-source local \
  --user-prompt-template "Q: {text}\nA:" \
  --limit 10

# Translation
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type freeform \
  --dataset-name wmt14 \
  --dataset-input-field source \
  --dataset-output-field target \
  --system-prompt "Translate the following text from English to Spanish." \
  --limit 10
```

## Dataset Format

**Minimum required fields**:
```jsonl
{"id": "1", "text": "What is AI?", "expected": "AI is artificial intelligence..."}
{"id": "2", "text": "Explain ML", "expected": "ML is machine learning..."}
```

**With custom field names** (use --dataset-*-field flags):
```jsonl
{"sample_id": "1", "question": "What is AI?", "answer": "AI is..."}
```

**With additional context fields**:
```jsonl
{"id": "1", "context": "Background info...", "question": "What is...?", "expected": "Answer..."}
```

Use prompt template to include context:
```bash
--user-prompt-template "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
```

## Metrics

- **ExactMatch**: Binary metric (0 or 1) for exact string match after normalization
- **F1Score**: Token-level F1 score (precision and recall of tokens)
- **BLEU** (optional): N-gram based similarity metric

### Text Normalization

By default, predictions and references are normalized before comparison:
- Convert to lowercase (unless case_sensitive=True)
- Remove extra whitespace
- Strip punctuation (configurable)
- Remove accents (configurable)

This ensures fair comparison while accounting for minor formatting differences.

## Use Cases

**Question Answering**: Compare generated answers to reference answers
**Summarization**: Evaluate summary quality against reference summaries
**Translation**: Compare translations to reference translations
**Creative Writing**: Evaluate generated text against examples
**Code Generation**: Compare generated code to reference implementations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .metrics import ExactMatch, F1Score
from .base import BaseHandler

logger = logging.getLogger(__name__)


class FreeformHandler(BaseHandler):
    """Handler for freeform text generation tasks.
    
    This handler provides support for open-ended text generation evaluation using
    standard metrics like exact match, F1 score, and optional BLEU score.
    
    Class Attributes (in addition to BaseHandler):
        answer_type: Set to "freeform"
        normalize_prediction: Whether to normalize text for comparison (default: True)
        case_sensitive: Whether comparison is case-sensitive (default: False)
    
    Example:
        class MyTranslationTask(FreeformHandler):
            dataset = "org/translation-dataset"
            system_prompt = "You are a translator."
            metrics = [ExactMatch(), F1Score(), BLEUScore()]
    """

    # Freeform defaults
    answer_type = "freeform"
    normalize_prediction: bool = True
    case_sensitive: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the freeform handler."""
        # Apply config overrides BEFORE calling super().__init__
        if config:
            self._apply_config_overrides(config)
        
        super().__init__(config)

        # Set default metrics if not provided
        if not self.metrics:
            self.metrics = [ExactMatch(), F1Score()]
    
    def _apply_config_overrides(self, config: Dict[str, Any]):
        """Apply configuration overrides to handler attributes.
        
        This allows CLI/config-driven tasks to work with the same handler
        as Python-defined tasks.
        """
        dataset_config = config.get("dataset", {})
        
        # Field mappings
        if "input_field" in dataset_config:
            self.text_field = dataset_config["input_field"]
        if "output_field" in dataset_config:
            self.label_field = dataset_config["output_field"]
        
        # Prompts
        if "system_prompt" in config:
            self.system_prompt = config["system_prompt"]
        if "user_prompt_template" in config:
            self.user_prompt_template = config["user_prompt_template"]
        
        # Multimodal support
        if dataset_config.get("multimodal_input"):
            self.requires_multimodal = True

    def preprocess_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """Transform a raw dataset sample to eval format.
        
        Extracts text input and expected text output.
        
        Args:
            raw_sample: Raw sample from dataset
            idx: Sample index
            
        Returns:
            Processed sample with text and expected fields
        """
        text = raw_sample.get(self.text_field)
        expected = raw_sample.get(self.label_field)

        if text is None or expected is None:
            logger.warning(f"Skipping sample {idx}: missing text or expected output")
            return None

        return {
            "id": f"{self.get_task_name()}_{idx}",
            "text": str(text),
            "expected": str(expected),
        }

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts for freeform text generation.
        
        Args:
            sample: Sample with text field
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Use custom template if provided, otherwise simple format
        if hasattr(self, "user_prompt_template"):
            user_prompt = self.user_prompt_template.format(**sample)
        else:
            user_prompt = sample.get("text", "")

        return self.system_prompt, user_prompt

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not self.normalize_prediction:
            return text

        # Basic normalization
        normalized = text.strip()

        if not self.case_sensitive:
            normalized = normalized.lower()

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for freeform text prediction.
        
        Uses configured metrics (ExactMatch, F1Score, etc.) to evaluate the prediction.
        
        Args:
            prediction: Model output text
            expected: Expected output text
            sample: Full sample dict
            error: Error message if any
            error_type: Type of error
            
        Returns:
            Metrics dict with scores from all configured metrics
        """
        if error:
            return self.get_error_metrics(error, error_type)

        # Normalize if configured
        if isinstance(prediction, str) and isinstance(expected, str):
            pred_normalized = self._normalize_text(prediction)
            exp_normalized = self._normalize_text(expected)
        else:
            pred_normalized = prediction
            exp_normalized = expected

        # Calculate metrics using configured metric objects
        metrics_result = {"valid": True}

        for metric in self.metrics:
            try:
                # Use per_sample() for metric protocol, or compute() for ScalarMetric subclasses
                if hasattr(metric, 'per_sample'):
                    metric_output = metric.per_sample(pred_normalized, exp_normalized, sample)
                elif hasattr(metric, 'compute'):
                    metric_output = metric.compute(pred_normalized, exp_normalized, sample)
                else:
                    continue
                    
                if isinstance(metric_output, dict):
                    metrics_result.update(metric_output)
                else:
                    # Single value metric
                    metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
                    metrics_result[metric_name] = metric_output
            except Exception as e:
                logger.warning(f"Error calculating {metric.__class__.__name__}: {e}")
                continue

        return metrics_result

    def get_error_metrics(
        self, error: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return error metrics for failed predictions.
        
        Args:
            error: Error message
            error_type: Type of error
            
        Returns:
            Metrics dict matching structure of successful predictions
        """
        error_metrics = {
            "valid": False,
            "error": error,
            "error_type": error_type,
        }

        # Add zero values for all configured metrics
        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
            error_metrics[metric_name] = 0.0

        return error_metrics

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate freeform text metrics across all samples.
        
        Computes averages for all numeric metrics.
        
        Args:
            all_metrics: List of per-sample metrics
            
        Returns:
            Aggregated metrics dict
        """
        if not all_metrics:
            return {}

        total_samples = len(all_metrics)
        valid_samples = sum(1 for m in all_metrics if m.get("valid", True))

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_count": total_samples - valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
        }

        # Collect all metric names from the metrics objects
        metric_names = set()
        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
            metric_names.add(metric_name)

        # Compute averages for each metric
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics if m.get("valid", True)]
            if values:
                aggregated[metric_name] = sum(values) / len(values)

        return aggregated

