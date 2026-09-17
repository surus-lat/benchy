"""P2 — strict YAML parsing (spec §1). Syntax only; semantics are the compiler's."""

from __future__ import annotations

import pytest

from benchy.compiler import parse
from benchy.errors import BenchyError

CANONICAL = """
version: "1.0"
ontology_version: "1.0"
benchmark:
  task: extract
  domain: finance
  language: es
"""


def err(text: str) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        parse(text)
    return exc.value


def test_valid_yaml_parses_to_a_mapping():
    doc = parse(CANONICAL)
    assert doc["version"] == "1.0"
    assert doc["benchmark"]["task"] == "extract"


def test_key_order_is_preserved():
    assert list(parse("b: 1\na: 2\nc: 3")) == ["b", "a", "c"]


def test_c02_duplicate_top_level_key_is_rejected():
    e = err("a: 1\na: 2")
    assert e.code == "duplicate_key"
    assert "a" in e.message


def test_duplicate_nested_key_is_rejected():
    assert err("outer:\n  a: 1\n  a: 2").code == "duplicate_key"


def test_duplicate_key_reports_its_line():
    assert "line 2" in err("a: 1\na: 2").message


def test_anchor_is_rejected():
    assert err("a: &anchor 1\nb: 2").code == "invalid_yaml"


def test_alias_is_rejected():
    assert err("a: &x 1\nb: *x").code == "invalid_yaml"


def test_merge_key_is_rejected():
    assert err("a:\n  <<: {b: 1}").code == "invalid_yaml"


@pytest.mark.parametrize("text", ["a: !foo bar", "a: !!python/object:os.system {}"])
def test_custom_tags_are_rejected(text):
    assert err(text).code == "invalid_yaml"


def test_malformed_yaml_is_rejected():
    assert err("a: [1,").code == "invalid_yaml"


def test_malformed_yaml_reports_a_location():
    assert "line" in err("a: [1,").message


@pytest.mark.parametrize("text", ["", "   \n\n", "# only a comment\n"])
def test_empty_document_is_rejected(text):
    assert err(text).code == "invalid_yaml"


@pytest.mark.parametrize("text", ["- a\n- b", "just a string", "42"])
def test_non_mapping_root_is_rejected(text):
    assert err(text).code == "invalid_yaml"


def test_complex_key_is_rejected():
    assert err("? [a, b]\n: value").code == "invalid_yaml"
