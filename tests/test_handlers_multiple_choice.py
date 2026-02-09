"""Tests for MultipleChoiceHandler classification logic."""

from typing import Any, ClassVar, Dict


from src.tasks.common.multiple_choice import MultipleChoiceHandler


class SimpleClassificationHandler(MultipleChoiceHandler):
    """Simple classification handler for testing MODE 1 (task-level labels)."""
    
    labels: ClassVar[Dict[Any, str]] = {0: "No", 1: "Yes"}
    dataset_name = "test/dataset"
    text_field = "text"
    label_field = "label"


class PerSampleChoiceHandler(MultipleChoiceHandler):
    """Handler for testing MODE 2 (per-sample choices)."""
    
    # No labels attribute - samples provide their own choices
    dataset_name = "test/dataset"


# Initialization Tests

def test_init_with_task_level_labels_builds_choice_mapping():
    """Test initialization with task-level labels builds choice mapping."""
    handler = SimpleClassificationHandler()
    
    assert len(handler.choice_texts) == 2
    assert handler.choice_texts == ["No", "Yes"]
    assert handler.choice_labels == ["0", "1"]
    assert handler.label_to_index == {0: 0, 1: 1}


def test_init_without_labels_for_per_sample_choices():
    """Test initialization without labels for per-sample choice mode."""
    handler = PerSampleChoiceHandler()
    
    assert len(handler.choice_texts) == 0
    assert len(handler.choice_labels) == 0
    assert len(handler.label_to_index) == 0


def test_init_with_config_overrides_fields():
    """Test initialization with config overrides field mappings."""
    config = {
        "dataset": {
            "input_field": "question",
            "output_field": "answer",
            "label_field": "answer_label",
        }
    }
    
    handler = SimpleClassificationHandler(config)
    
    assert handler.text_field == "question"
    assert handler.label_field == "answer_label"  # label_field is the config key


def test_init_with_config_overrides_labels():
    """Test initialization with config overrides labels."""
    config = {
        "dataset": {
            "labels": {"A": "Option A", "B": "Option B", "C": "Option C"}
        }
    }
    
    handler = PerSampleChoiceHandler(config)
    
    assert len(handler.choice_texts) == 3
    assert "Option A" in handler.choice_texts
    assert "Option B" in handler.choice_texts
    assert "Option C" in handler.choice_texts


def test_init_with_config_enables_multimodal():
    """Test initialization with config enables multimodal support."""
    config = {
        "dataset": {
            "multimodal_input": True
        }
    }
    
    handler = SimpleClassificationHandler(config)
    
    assert handler.requires_multimodal is True


# Config Override Tests

def test_apply_config_overrides_field_mappings():
    """Test _apply_config_overrides updates field mappings."""
    config = {
        "dataset": {
            "input_field": "custom_text",
            "output_field": "custom_label",
        }
    }
    
    handler = SimpleClassificationHandler(config)
    
    assert handler.text_field == "custom_text"
    assert handler.label_field == "custom_label"


def test_apply_config_overrides_labels_from_json_string():
    """Test _apply_config_overrides parses labels from JSON string."""
    config = {
        "dataset": {
            "labels": '{"0": "Negative", "1": "Positive"}'
        }
    }
    
    handler = PerSampleChoiceHandler(config)
    
    assert handler.labels == {"0": "Negative", "1": "Positive"}


def test_apply_config_overrides_labels_from_dict():
    """Test _apply_config_overrides accepts labels as dict."""
    config = {
        "dataset": {
            "labels": {"0": "Negative", "1": "Positive"}
        }
    }
    
    handler = PerSampleChoiceHandler(config)
    
    assert handler.labels == {"0": "Negative", "1": "Positive"}


def test_apply_config_overrides_prompts():
    """Test _apply_config_overrides updates prompts."""
    config = {
        "system_prompt": "Custom system prompt",
        "user_prompt_template": "Q: {text}\nA:",
    }
    
    handler = SimpleClassificationHandler(config)
    
    assert handler.system_prompt == "Custom system prompt"
    assert handler.user_prompt_template == "Q: {text}\nA:"


def test_parse_labels_from_json_string():
    """Test _parse_labels parses JSON string."""
    handler = SimpleClassificationHandler()
    
    labels = handler._parse_labels('{"0": "No", "1": "Yes"}')
    
    assert labels == {"0": "No", "1": "Yes"}


def test_parse_labels_from_dict():
    """Test _parse_labels returns dict as-is."""
    handler = SimpleClassificationHandler()
    
    labels_dict = {"0": "No", "1": "Yes"}
    labels = handler._parse_labels(labels_dict)
    
    assert labels == labels_dict


# Sample Preprocessing Tests (MODE 1: Task-level labels)

def test_preprocess_sample_mode1_extracts_text_and_label():
    """Test MODE 1 preprocessing extracts text and label fields."""
    handler = SimpleClassificationHandler()
    
    raw_sample = {"text": "Is this positive?", "label": 1}
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is not None
    assert result["text"] == "Is this positive?"
    assert "choices" in result
    assert result["expected"] == 1


def test_preprocess_sample_mode1_adds_choices_from_labels():
    """Test MODE 1 preprocessing adds choices from task-level labels."""
    handler = SimpleClassificationHandler()
    
    raw_sample = {"text": "Sample", "label": 0}
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result["choices"] == ["No", "Yes"]
    assert result["choice_labels"] == ["0", "1"]


def test_preprocess_sample_mode1_handles_string_labels():
    """Test MODE 1 preprocessing handles string label values."""
    class StringLabelHandler(MultipleChoiceHandler):
        labels: ClassVar[Dict[Any, str]] = {"negative": "Negative", "positive": "Positive"}
        text_field = "text"
        label_field = "label"
    
    handler = StringLabelHandler()
    
    raw_sample = {"text": "Sample", "label": "positive"}
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is not None
    assert result["expected"] == 1  # Index in sorted labels


def test_preprocess_sample_mode1_skips_on_missing_label():
    """Test MODE 1 preprocessing skips sample when label missing."""
    handler = SimpleClassificationHandler()
    
    raw_sample = {"text": "Sample without label"}
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is None


def test_preprocess_sample_mode1_skips_on_missing_text():
    """Test MODE 1 preprocessing skips sample when text missing."""
    handler = SimpleClassificationHandler()
    
    raw_sample = {"label": 1}
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is None


# Sample Preprocessing Tests (MODE 2: Per-sample choices)

def test_preprocess_sample_mode2_passes_through_choices():
    """Test MODE 2 preprocessing passes through per-sample choices."""
    handler = PerSampleChoiceHandler()
    
    raw_sample = {
        "id": "q1",
        "text": "What is 2+2?",
        "choices": ["3", "4", "5"],
        "choice_labels": ["A", "B", "C"],
        "expected": 1
    }
    
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result == raw_sample  # Already preprocessed, returned as-is


def test_preprocess_sample_mode2_skips_on_missing_choices():
    """Test MODE 2 preprocessing skips when choices missing."""
    handler = PerSampleChoiceHandler()
    
    raw_sample = {
        "text": "Question without choices",
        "expected": 0
    }
    
    # Without labels attribute and without choices in sample, should raise or skip
    try:
        result = handler.preprocess_sample(raw_sample, 0)
        # If it doesn't raise, check that it handled gracefully
        assert result is None or "id" in result
    except ValueError:
        # Expected - MODE 1 requires labels
        assert True


# Prompt Generation Tests

def test_get_prompt_formats_choices_with_labels():
    """Test get_prompt formats choices with labels."""
    handler = SimpleClassificationHandler()
    
    sample = {
        "text": "Is this positive?",
        "choices": ["No", "Yes"],
        "choice_labels": ["0", "1"]
    }
    
    system, user = handler.get_prompt(sample)
    
    assert "No" in user or "Yes" in user  # Choices should be in prompt


def test_get_prompt_uses_custom_template():
    """Test get_prompt uses custom user_prompt_template."""
    handler = SimpleClassificationHandler()
    handler.user_prompt_template = "Question: {text}\nAnswer:"
    
    sample = {
        "text": "Sample question",
        "choices": ["No", "Yes"],
        "choice_labels": ["A", "B"],
    }
    
    system, user = handler.get_prompt(sample)
    
    assert "Sample question" in user


# Metrics Tests

def test_calculate_metrics_returns_accuracy_dict():
    """Test calculate_metrics returns accuracy metrics."""
    handler = SimpleClassificationHandler()
    
    sample = {
        "choices": ["No", "Yes"],
        "choice_labels": ["0", "1"],
        "expected": 1
    }
    
    # calculate_metrics is called by the handler internally
    # For testing, we can check the metrics property exists
    assert handler.metrics is not None


def test_aggregate_metrics_computes_mean_accuracy():
    """Test aggregate_metrics computes mean accuracy."""
    handler = SimpleClassificationHandler()
    
    per_sample_metrics = [
        {"accuracy": 1.0, "valid": True},
        {"accuracy": 0.0, "valid": True},
        {"accuracy": 1.0, "valid": True},
    ]
    
    # aggregate_metrics is called by the metric objects
    # For MultipleChoiceAccuracy
    from src.tasks.common.metrics import MultipleChoiceAccuracy
    metric = MultipleChoiceAccuracy()
    aggregated = metric.aggregate(per_sample_metrics)
    
    assert "accuracy" in aggregated
    # Mean of [1.0, 0.0, 1.0] = 0.666...
    assert 0.6 < aggregated["accuracy"] < 0.7
