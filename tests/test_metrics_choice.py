from src.tasks.common.metrics import ExactMatch, F1Score, MultipleChoiceAccuracy
from src.tasks.common.utils.choice_utils import parse_choice_prediction


def test_parse_choice_prediction_from_json_string() -> None:
    idx = parse_choice_prediction(
        '{"answer": "B"}',
        choices=["red", "blue", "green"],
        labels=["A", "B", "C"],
    )
    assert idx == 1


def test_parse_choice_prediction_with_partial_matching_when_non_strict() -> None:
    idx = parse_choice_prediction(
        "I think the answer is very long cho",
        choices=["very long choice text", "another option"],
        strict=False,
    )
    assert idx == 0


def test_multiple_choice_accuracy_invalid_response() -> None:
    metric = MultipleChoiceAccuracy()
    result = metric.per_sample(
        prediction="zzz",
        expected=0,
        sample={"choices": ["yes", "no"], "choice_labels": ["A", "B"]},
    )

    assert result["valid"] is False
    assert result["accuracy"] == 0.0
    assert result["error_type"] == "invalid_response"


def test_exact_match_case_insensitive_and_strip() -> None:
    metric = ExactMatch()
    score = metric.compute("  YES ", "yes", sample={})
    assert score == 1.0


def test_f1_score_with_reference_list() -> None:
    metric = F1Score()
    score = metric.compute("new york city", ["new york", "los angeles"], sample={})
    assert score > 0.0
    assert score <= 1.0
