"""Field metadata: the optional inert keys `critical` / `derivation`.

A declared field may carry `critical` (bool) and `derivation` ("copied" |
"derived") next to exactly one structural form (`type` / `enum` / `fields`).
The compiler validates them and the IR conserves them verbatim, but
`validate`, `leaves`, `equal` and the scoring dimensions never read them:
metadata is reported, it never moves a score. Schemas without metadata keys
compile to the exact same IR as before.
"""

from __future__ import annotations

import pytest
from conftest import edit

from benchy import types
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError


def fails_schema(node: object) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        types.compile_schema(node)
    return exc.value


# ---------------------------------------------------------------------------
# conservation: declared metadata reaches the IR verbatim
# ---------------------------------------------------------------------------

def test_primitive_field_with_metadata_is_conserved():
    ir = types.compile_schema({"total": {"type": "float", "critical": True, "derivation": "copied"}})
    assert ir["fields"]["total"] == {"type": "float", "critical": True, "derivation": "copied"}


def test_enum_field_with_metadata_is_conserved():
    ir = types.compile_schema({"label": {"enum": ["a", "b"], "critical": True}})
    assert ir["fields"]["label"] == {"type": "enum", "values": ["a", "b"], "critical": True}


def test_absent_metadata_is_never_injected():
    ir = types.compile_schema({"total": {"type": "float", "critical": True}, "note": "string"})
    assert "derivation" not in ir["fields"]["total"]
    assert ir["fields"]["note"] == {"type": "string"}


def test_metadata_on_a_nested_object_field():
    ir = types.compile_schema({
        "party": {"fields": {"name": {"type": "string", "derivation": "derived"}}, "critical": False},
    })
    party = ir["fields"]["party"]
    assert party["critical"] is False
    assert "derivation" not in party
    assert party["fields"]["name"] == {"type": "string", "derivation": "derived"}


def test_compile_benchmark_conserves_metadata_and_scores_normally():
    text = edit(
        program={
            "input": {"text": "string"},
            "output": {"total": {"type": "float", "critical": True, "derivation": "derived"}},
        },
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
    )
    ir = compile_benchmark(text)
    assert ir["program"]["output"]["fields"]["total"] == {
        "type": "float", "critical": True, "derivation": "derived",
    }
    # The weight tree still mirrors the leaves exactly; metadata adds no dimension.
    assert ir["scoring"]["dimensions"] == [{"path": ["total"], "weight": 1.0}]


# ---------------------------------------------------------------------------
# validation: malformed declarations are compile errors
# ---------------------------------------------------------------------------

def test_metadata_requires_exactly_one_structural_form():
    err = fails_schema({"f": {"type": "string", "enum": ["a"], "critical": True}})
    assert err.code == "invalid_schema"
    err = fails_schema({"f": {"critical": True}})
    assert err.code == "invalid_schema"


def test_metadata_rejects_unknown_keys():
    err = fails_schema({"f": {"type": "string", "critical": True, "bogus": 1}})
    assert err.code == "invalid_schema"
    assert "bogus" in err.message


def test_critical_must_be_a_boolean():
    err = fails_schema({"f": {"type": "string", "critical": "yes"}})
    assert err.code == "invalid_schema"
    assert "critical" in err.message


def test_derivation_is_a_closed_vocabulary():
    err = fails_schema({"f": {"type": "string", "derivation": "guessed"}})
    assert err.code == "invalid_schema"
    assert "derivation" in err.message


def test_declared_form_still_validates_the_structure():
    err = fails_schema({"f": {"type": "nope", "critical": True}})
    assert err.code == "invalid_schema"
    err = fails_schema({"f": {"fields": {}, "critical": True}})
    assert err.code == "invalid_schema"


# ---------------------------------------------------------------------------
# inertness: metadata cannot move a score
# ---------------------------------------------------------------------------

def test_metadata_changes_nothing_the_engine_reads():
    plain = types.compile_schema({"total": "float"})
    declared = types.compile_schema({"total": {"type": "float", "critical": True, "derivation": "copied"}})
    node = declared["fields"]["total"]
    assert types.leaves(declared) == types.leaves(plain) == [["total"]]
    assert types.validate(1.5, node, phase="dataset") == 1.5
    assert types.equal(1.5, 1.5, node) is True
    assert types.equal(1.5, 2.5, node) is False
