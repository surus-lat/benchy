"""Unit tests for src/data_generator.py pure functions.

No API calls. Tests prompt building, validation, and normalization only.
"""

from __future__ import annotations

import pytest

from src.data_generator import _build_prompt, _validate_sample, _normalize


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def _structured_spec(self, fields=None, seed=None):
        spec = {
            "task": {
                "type": "extraction",
                "input": {"type": "text", "description": "an invoice"},
                "output": {
                    "type": "structured",
                    "fields": fields or [
                        {"name": "vendor_name", "type": "string", "description": "Seller name", "required": True},
                        {"name": "amount", "type": "number", "description": "Total amount", "required": True},
                        {"name": "date", "type": "string", "description": "Invoice date", "required": False},
                    ],
                },
            },
        }
        if seed:
            spec["data"] = {"seed_description": seed}
        return spec

    def _freeform_spec(self, output_type="text"):
        return {
            "task": {
                "type": "qa",
                "input": {"type": "text", "description": "a question"},
                "output": {"type": output_type},
            },
        }

    def test_structured_prompt_contains_field_names(self):
        prompt = _build_prompt(self._structured_spec())
        assert "vendor_name" in prompt
        assert "amount" in prompt
        assert "date" in prompt

    def test_structured_prompt_uses_input_key(self):
        prompt = _build_prompt(self._structured_spec())
        assert '"input"' in prompt

    def test_structured_prompt_uses_expected_output_key(self):
        prompt = _build_prompt(self._structured_spec())
        assert '"expected_output"' in prompt

    def test_freeform_prompt_uses_input_key(self):
        prompt = _build_prompt(self._freeform_spec())
        assert '"input"' in prompt

    def test_freeform_prompt_uses_expected_output_key(self):
        prompt = _build_prompt(self._freeform_spec())
        assert '"expected_output"' in prompt

    def test_seed_description_appended(self):
        prompt = _build_prompt(self._structured_spec(seed="invoices from Argentina"))
        assert "invoices from Argentina" in prompt

    def test_no_seed_no_seed_section(self):
        prompt = _build_prompt(self._structured_spec())
        assert "Seed guidance" not in prompt

    def test_json_only_instruction_present(self):
        prompt = _build_prompt(self._structured_spec())
        assert "valid JSON" in prompt
        assert "No markdown" in prompt

    def test_required_vs_optional_labelled(self):
        prompt = _build_prompt(self._structured_spec())
        assert "[required]" in prompt
        assert "[optional]" in prompt


# ---------------------------------------------------------------------------
# _validate_sample
# ---------------------------------------------------------------------------

class TestValidateSample:
    def _fields(self, names=("vendor_name", "amount")):
        return [{"name": n, "required": True} for n in names]

    def test_valid_structured_sample(self):
        sample = {"input": "some text", "expected_output": {"vendor_name": "ACME", "amount": 100}}
        assert _validate_sample(sample, self._fields()) is True

    def test_missing_input_key(self):
        sample = {"expected_output": {"vendor_name": "ACME", "amount": 100}}
        assert _validate_sample(sample, self._fields()) is False

    def test_missing_expected_output_key(self):
        sample = {"input": "some text"}
        assert _validate_sample(sample, self._fields()) is False

    def test_required_field_missing_from_expected_output(self):
        sample = {"input": "text", "expected_output": {"vendor_name": "ACME"}}  # missing "amount"
        assert _validate_sample(sample, self._fields()) is False

    def test_optional_field_missing_is_ok(self):
        fields = [
            {"name": "vendor_name", "required": True},
            {"name": "date", "required": False},
        ]
        sample = {"input": "text", "expected_output": {"vendor_name": "ACME"}}
        assert _validate_sample(sample, fields) is True

    def test_no_fields_only_checks_keys(self):
        sample = {"input": "text", "expected_output": "some answer"}
        assert _validate_sample(sample, []) is True

    def test_expected_output_not_dict_when_fields_required(self):
        sample = {"input": "text", "expected_output": "flat string"}
        assert _validate_sample(sample, self._fields()) is False

    def test_non_dict_sample_returns_false(self):
        assert _validate_sample("not a dict", []) is False


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_output_keys(self):
        sample = {"input": "hello", "expected_output": {"field": "value"}}
        result = _normalize(sample, 0)
        assert set(result.keys()) == {"id", "input", "expected_output"}

    def test_id_is_string(self):
        result = _normalize({"input": "x", "expected_output": "y"}, 42)
        assert result["id"] == "42"

    def test_input_forwarded(self):
        result = _normalize({"input": "my input", "expected_output": "out"}, 0)
        assert result["input"] == "my input"

    def test_expected_output_forwarded(self):
        expected = {"vendor": "ACME", "amount": 99.0}
        result = _normalize({"input": "x", "expected_output": expected}, 0)
        assert result["expected_output"] == expected

    def test_missing_input_defaults_to_empty(self):
        result = _normalize({"expected_output": "y"}, 0)
        assert result["input"] == ""

    def test_missing_expected_output_defaults_to_empty(self):
        result = _normalize({"input": "x"}, 0)
        assert result["expected_output"] == ""
