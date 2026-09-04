"""Tolerant label parsing for classification (`benchy.task.choice`)."""

from __future__ import annotations

import pytest

from benchy.task.choice import parse_label

LABELS = ["urgente", "normal", "bajo"]


class TestLetters:
    @pytest.mark.parametrize("text", ["A", "a", " A ", "A.", "A)", "A: urgente"])
    def test_resolves_letter_to_label(self, text):
        label, error = parse_label(text, LABELS)
        assert error is None
        assert label == "urgente"

    def test_resolves_second_letter(self):
        label, error = parse_label("B", LABELS)
        assert error is None and label == "normal"


class TestIndex:
    def test_zero_based_index(self):
        assert parse_label(0, LABELS) == ("urgente", None)

    def test_string_index(self):
        assert parse_label("2", LABELS) == ("bajo", None)

    def test_out_of_range_index_fails_cleanly(self):
        label, error = parse_label(99, LABELS)
        assert label is None and error is not None

    def test_float_integer_index(self):
        assert parse_label(1.0, LABELS) == ("normal", None)


class TestTextMatch:
    def test_exact_label_text(self):
        assert parse_label("urgente", LABELS) == ("urgente", None)

    def test_case_and_accent_insensitive(self):
        label, error = parse_label("URGENTE", LABELS)
        assert error is None and label == "urgente"

    def test_substring_match(self):
        label, error = parse_label("Yo diria que es urgente, sin dudas.", LABELS)
        assert error is None and label == "urgente"


class TestAnswerMarkers:
    @pytest.mark.parametrize(
        "text",
        ["Answer: normal", "Respuesta: normal", "After thinking, Answer: normal"],
    )
    def test_extracts_after_marker(self, text):
        label, error = parse_label(text, LABELS)
        assert error is None and label == "normal"


class TestJsonWrapped:
    def test_dict_with_label_key(self):
        assert parse_label({"label": "bajo"}, LABELS) == ("bajo", None)

    def test_dict_with_single_key(self):
        assert parse_label({"answer": "bajo"}, LABELS) == ("bajo", None)

    def test_json_string(self):
        assert parse_label('{"label": "bajo"}', LABELS) == ("bajo", None)

    def test_singleton_list(self):
        assert parse_label(["normal"], LABELS) == ("normal", None)


class TestFailureModes:
    def test_none_fails_cleanly(self):
        label, error = parse_label(None, LABELS)
        assert label is None and isinstance(error, str)

    def test_empty_string_fails_cleanly(self):
        label, error = parse_label("", LABELS)
        assert label is None and isinstance(error, str)

    def test_unrelated_text_fails_cleanly(self):
        label, error = parse_label("The weather is nice today.", LABELS)
        assert label is None and isinstance(error, str)

    def test_no_labels_configured(self):
        label, error = parse_label("urgente", [])
        assert label is None and isinstance(error, str)

    def test_never_raises_on_weird_dict(self):
        label, error = parse_label({"a": 1, "b": 2}, LABELS)
        assert label is None and isinstance(error, str)
