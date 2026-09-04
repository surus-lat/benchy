"""Heavy parametrization of the JSON-from-prose repair path.

This is the highest-value surface in the module: a benchmark run must
survive whatever a system actually says, not just what it was asked to
say. Every case here must produce a `(value, error)` pair without ever
raising.
"""

from __future__ import annotations

import pytest

from benchy.task.repair import extract_json


class TestNeverRaises:
    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "   ",
            "I'm sorry, I can't help with that.",
            "{",
            "}",
            "{{{{",
            "[[[[",
            "\x00\x01\x02",
        ],
    )
    def test_pathological_input_never_raises(self, text):
        value, error = extract_json(text)
        assert value is None
        assert isinstance(error, str)


class TestCleanJson:
    def test_bare_object(self):
        value, error = extract_json('{"name": "Ana"}')
        assert error is None
        assert value == {"name": "Ana"}

    def test_none_text_fails_cleanly(self):
        value, error = extract_json(None)
        assert value is None and "no text" in error

    def test_empty_string_fails_cleanly(self):
        value, error = extract_json("")
        assert value is None and "empty" in error


class TestFencedBlocks:
    @pytest.mark.parametrize(
        "text",
        [
            '```json\n{"name": "Ana"}\n```',
            '```\n{"name": "Ana"}\n```',
            'Claro! Aqui tienes:\n```json\n{"name": "Ana"}\n```\nEspero que sirva.',
            'The answer is:\n\n```JSON\n{"name": "Ana"}\n```',
        ],
    )
    def test_extracts_from_fence(self, text):
        value, error = extract_json(text)
        assert error is None, error
        assert value == {"name": "Ana"}


class TestProseAroundJson:
    @pytest.mark.parametrize(
        "text",
        [
            'Sure, here you go: {"name": "Ana"} hope that helps!',
            'Answer: {"name": "Ana"}',
            '{"name": "Ana"} -- that is the extracted name.',
        ],
    )
    def test_extracts_embedded_object(self, text):
        value, error = extract_json(text)
        assert error is None, error
        assert value == {"name": "Ana"}


class TestTrailingCommas:
    @pytest.mark.parametrize(
        "text",
        [
            '{"name": "Ana",}',
            '{"name": "Ana", "age": 30,}',
            '{"items": ["a", "b",]}',
        ],
    )
    def test_repairs_trailing_comma(self, text):
        value, error = extract_json(text)
        assert error is None, error
        assert isinstance(value, dict)


class TestSingleQuotes:
    def test_repairs_python_dict_style_quotes(self):
        value, error = extract_json("{'name': 'Ana'}")
        assert error is None, error
        assert value == {"name": "Ana"}

    def test_does_not_touch_valid_json_with_an_apostrophe(self):
        value, error = extract_json('{"name": "O\'Brien"}')
        assert error is None, error
        assert value == {"name": "O'Brien"}


class TestUnescapedNewlines:
    def test_repairs_raw_newline_inside_string(self):
        text = '{"note": "line one\nline two"}'
        value, error = extract_json(text)
        assert error is None, error
        assert value == {"note": "line one\nline two"}


class TestUnquotedKeys:
    def test_repairs_bare_identifier_keys(self):
        value, error = extract_json("{name: 'Ana', age: 30}")
        assert error is None, error
        assert value == {"name": "Ana", "age": 30}


class TestPythonLiterals:
    def test_repairs_true_false_none(self):
        value, error = extract_json('{"active": True, "deleted": False, "note": None}')
        assert error is None, error
        assert value == {"active": True, "deleted": False, "note": None}


class TestArrayVsObject:
    def test_parses_array_when_object_was_not_required_by_extract_json_itself(self):
        # extract_json is shape-agnostic; unwrapping singleton arrays is a
        # Task-level decision (see base.py::_reconcile_shape), not this
        # module's job.
        value, error = extract_json('[{"name": "Ana"}]')
        assert error is None
        assert value == [{"name": "Ana"}]


class TestNested:
    def test_nested_objects_survive_brace_scanning(self):
        text = (
            '{"nombre": "Ana", "tiene_negocio": true, '
            '"negocio": {"descripcion_negocio": "panaderia", '
            '"meses_en_negocio": 12, "cantidad_empleados": 3}}'
        )
        value, error = extract_json(text)
        assert error is None, error
        assert value["negocio"]["cantidad_empleados"] == 3

    def test_braces_inside_string_values_do_not_confuse_the_scanner(self):
        text = '{"template": "use {curly} braces like this", "ok": true}'
        value, error = extract_json(text)
        assert error is None, error
        assert value == {"template": "use {curly} braces like this", "ok": True}
