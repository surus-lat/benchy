"""P4 — compiling `scoring` into explicit IR dimensions (spec §6).

Covers conformance cases C11–C15 and C28.
"""

from __future__ import annotations

import math

import pytest

from benchy.compiler import compile_scoring
from benchy.errors import BenchyError
from benchy.types import compile_schema

FLAT = {"invoice_number": "string", "date": "date", "total": "float"}
NESTED = {"supplier": {"name": "string", "tax_id": "string"}, "total": "float"}


def scoring(weights, aggregator="weighted_mean"):
    return {"weights": weights, "aggregator": aggregator}


def compile_ok(out, weights, **kw):
    return compile_scoring(scoring(weights, **kw), compile_schema(out))


def fails(out, weights, **kw) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        compile_ok(out, weights, **kw)
    return exc.value


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------

def test_flat_weights_compile_to_dimensions():
    ir = compile_ok(FLAT, {"invoice_number": 1, "date": 1, "total": 5})
    assert ir["dimensions"] == [
        {"path": ["invoice_number"], "weight": 1.0},
        {"path": ["date"], "weight": 1.0},
        {"path": ["total"], "weight": 5.0},
    ]


def test_c28_nested_weights_compile_to_path_arrays():
    ir = compile_ok(NESTED, {"supplier": {"name": 1, "tax_id": 0}, "total": 5})
    assert ir["dimensions"] == [
        {"path": ["supplier", "name"], "weight": 1.0},
        {"path": ["supplier", "tax_id"], "weight": 0.0},
        {"path": ["total"], "weight": 5.0},
    ]


def test_fixed_specification_semantics_are_explicit_in_the_ir():
    ir = compile_ok(FLAT, {"invoice_number": 1, "date": 1, "total": 1})
    assert ir["evaluator"] == "exact_match"
    assert ir["instance_aggregator"] == "weighted_mean"
    assert ir["benchmark_aggregator"] == "mean"


def test_dimension_order_follows_the_output_schema_not_the_weight_mapping():
    ir = compile_ok(FLAT, {"total": 5, "date": 1, "invoice_number": 1})
    assert [d["path"] for d in ir["dimensions"]] == [["invoice_number"], ["date"], ["total"]]


def test_c13_a_zero_weight_among_positive_weights_is_valid():
    ir = compile_ok(FLAT, {"invoice_number": 0, "date": 1, "total": 5})
    assert ir["dimensions"][0]["weight"] == 0.0


def test_weights_are_normalized_to_floats():
    ir = compile_ok({"a": "string"}, {"a": 3})
    assert isinstance(ir["dimensions"][0]["weight"], float)


def test_enum_leaf_is_a_dimension():
    ir = compile_ok({"sentiment": {"enum": ["a", "b"]}}, {"sentiment": 1})
    assert ir["dimensions"] == [{"path": ["sentiment"], "weight": 1.0}]


# ---------------------------------------------------------------------------
# coverage — exactly one weight per leaf
# ---------------------------------------------------------------------------

def test_c11_missing_leaf_weight_is_rejected():
    err = fails(FLAT, {"invoice_number": 1, "date": 1})
    assert err.code == "missing_weight"
    assert err.path == ["total"]


def test_c11_missing_nested_leaf_weight_reports_the_full_path():
    err = fails(NESTED, {"supplier": {"name": 1}, "total": 5})
    assert (err.code, err.path) == ("missing_weight", ["supplier", "tax_id"])


def test_c12_extra_leaf_weight_is_rejected():
    err = fails(FLAT, {"invoice_number": 1, "date": 1, "total": 5, "bogus": 1})
    assert (err.code, err.path) == ("extra_weight", ["bogus"])


def test_c12_extra_nested_weight_is_rejected():
    err = fails(NESTED, {"supplier": {"name": 1, "tax_id": 0, "extra": 1}, "total": 5})
    assert (err.code, err.path) == ("extra_weight", ["supplier", "extra"])


def test_weight_on_an_intermediate_object_is_rejected():
    err = fails(NESTED, {"supplier": 1, "total": 5})
    assert err.code == "invalid_weight"
    assert err.path == ["supplier"]


def test_nested_weight_where_a_leaf_was_declared_is_rejected():
    err = fails(FLAT, {"invoice_number": {"deep": 1}, "date": 1, "total": 5})
    assert (err.code, err.path) == ("invalid_weight", ["invoice_number"])


# ---------------------------------------------------------------------------
# weight validity
# ---------------------------------------------------------------------------

def test_c15_negative_weight_is_rejected():
    err = fails(FLAT, {"invoice_number": -1, "date": 1, "total": 5})
    assert (err.code, err.path) == ("invalid_weight", ["invoice_number"])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_weight_is_rejected(bad):
    assert fails(FLAT, {"invoice_number": bad, "date": 1, "total": 5}).code == "invalid_weight"


@pytest.mark.parametrize("bad", ["1", None, True, [1]])
def test_non_numeric_weight_is_rejected(bad):
    assert fails(FLAT, {"invoice_number": bad, "date": 1, "total": 5}).code == "invalid_weight"


def test_c14_all_weights_zero_is_rejected():
    err = fails(FLAT, {"invoice_number": 0, "date": 0, "total": 0})
    assert err.code == "invalid_weight"
    assert "positive" in err.message


# ---------------------------------------------------------------------------
# the scoring section itself
# ---------------------------------------------------------------------------

def test_aggregator_must_be_weighted_mean():
    err = fails(FLAT, {"invoice_number": 1, "date": 1, "total": 1}, aggregator="mean")
    assert err.code == "invalid_value"
    assert err.path == ["scoring", "aggregator"]


def test_scoring_requires_both_keys():
    out = compile_schema(FLAT)
    for section in ({"weights": {}}, {"aggregator": "weighted_mean"}):
        with pytest.raises(BenchyError) as exc:
            compile_scoring(section, out)
        assert exc.value.code == "missing_key"


def test_unknown_scoring_key_is_rejected():
    out = compile_schema({"a": "string"})
    with pytest.raises(BenchyError) as exc:
        compile_scoring({"weights": {"a": 1}, "aggregator": "weighted_mean", "extra": 1}, out)
    assert exc.value.code == "unknown_key"


def test_weights_must_be_a_mapping():
    out = compile_schema({"a": "string"})
    with pytest.raises(BenchyError) as exc:
        compile_scoring({"weights": 1, "aggregator": "weighted_mean"}, out)
    assert exc.value.code == "invalid_weight"
