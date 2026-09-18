"""P1 — the semantic type system, exercised with no YAML and no IR file involved."""

from __future__ import annotations

import pytest

from benchy.errors import BenchyError
from benchy.types import compile_schema, equal, leaves, validate


def compile_err(node):
    with pytest.raises(BenchyError) as exc:
        compile_schema(node)
    return exc.value


def bad(value, node, **kw):
    with pytest.raises(BenchyError) as exc:
        validate(value, node, phase="dataset", **kw)
    return exc.value


# ---------------------------------------------------------------------------
# compile_schema — paper A.1 grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "token",
    ["string", "int", "float", "bool", "date", "time", "datetime", "image", "audio", "document"],
)
def test_every_primitive_token_compiles(token):
    assert compile_schema(token) == {"type": token}


def test_unknown_type_token_is_invalid_schema():
    err = compile_err("varchar")
    assert err.code == "invalid_schema"
    assert "varchar" in err.message


def test_flat_object_compiles_to_fields():
    assert compile_schema({"total": "float", "supplier": "string"}) == {
        "type": "object",
        "fields": {"total": {"type": "float"}, "supplier": {"type": "string"}},
    }


def test_nested_object_compiles_recursively():
    ir = compile_schema({"supplier": {"name": "string", "tax_id": "string"}, "total": "float"})
    assert ir["fields"]["supplier"] == {
        "type": "object",
        "fields": {"name": {"type": "string"}, "tax_id": {"type": "string"}},
    }


def test_enum_declaration_compiles():
    assert compile_schema({"enum": ["positive", "neutral", "negative"]}) == {
        "type": "enum",
        "values": ["positive", "neutral", "negative"],
    }


def test_enum_alongside_siblings_is_an_object_field_not_a_declaration():
    # Key set is not exactly {"enum"}, so this is an object with a field named
    # "enum" whose value is a list -> lists are rejected.
    assert compile_err({"enum": ["a"], "other": "string"}).code == "invalid_schema"


@pytest.mark.parametrize(
    "values",
    [{"enum": "positive"}, {"enum": []}, {"enum": ["a", ""]}, {"enum": ["a", 1]}, {"enum": ["a", "a"]}],
)
def test_malformed_enums_are_rejected(values):
    assert compile_err(values).code == "invalid_schema"


def test_null_field_value_is_rejected():
    err = compile_err({"total": None})
    assert err.code == "invalid_schema"
    assert err.path == ["total"]


def test_c04_variable_length_collection_is_rejected():
    assert compile_err({"items": ["a", "b"]}).code == "invalid_schema"


def test_empty_object_is_rejected():
    assert compile_err({}).code == "invalid_schema"


def test_nested_empty_object_is_rejected_with_path():
    assert compile_err({"supplier": {}}).path == ["supplier"]


def test_non_string_field_name_is_rejected():
    assert compile_err({1: "string"}).code == "invalid_schema"


def test_diagnostic_path_points_at_the_nested_offender():
    assert compile_err({"a": {"b": {"c": "nope"}}}).path == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# leaves — the scoring dimensions
# ---------------------------------------------------------------------------

def test_leaves_of_flat_schema_in_source_order():
    ir = compile_schema({"invoice_number": "string", "date": "date", "total": "float"})
    assert leaves(ir) == [["invoice_number"], ["date"], ["total"]]


def test_leaves_of_nested_schema_are_path_arrays_depth_first():
    ir = compile_schema({"supplier": {"name": "string", "tax_id": "string"}, "total": "float"})
    assert leaves(ir) == [["supplier", "name"], ["supplier", "tax_id"], ["total"]]


def test_leaves_treats_enum_as_a_leaf():
    ir = compile_schema({"sentiment": {"enum": ["a", "b"]}})
    assert leaves(ir) == [["sentiment"]]


# ---------------------------------------------------------------------------
# validate — spec §8 strict conformance, §9 runtime values
# ---------------------------------------------------------------------------

def test_valid_flat_object_round_trips():
    ir = compile_schema({"supplier": "string", "total": "float"})
    assert validate({"supplier": "ACME", "total": 121.0}, ir, phase="dataset") == {
        "supplier": "ACME",
        "total": 121.0,
    }


def test_missing_field_reports_missing_field_with_path():
    ir = compile_schema({"a": "string", "b": "string"})
    err = bad({"a": "x"}, ir)
    assert (err.code, err.path) == ("missing_field", ["b"])


def test_extra_field_reports_extra_field_with_path():
    ir = compile_schema({"a": "string"})
    err = bad({"a": "x", "debug": "y"}, ir)
    assert (err.code, err.path) == ("extra_field", ["debug"])


def test_nested_missing_field_path_is_full_path():
    ir = compile_schema({"supplier": {"name": "string"}})
    assert bad({"supplier": {}}, ir).path == ["supplier", "name"]


def test_object_expected_but_scalar_given():
    ir = compile_schema({"supplier": {"name": "string"}})
    assert bad({"supplier": "ACME"}, ir).code == "wrong_type"


@pytest.mark.parametrize(
    "type_name,value",
    [
        ("string", "x"), ("int", 5), ("float", 1.5), ("float", 5), ("bool", True),
        ("date", "2026-09-13"), ("time", "12:00:00"), ("time", "12:00:00.500"),
        ("datetime", "2026-09-13T12:00:00Z"), ("datetime", "2026-09-13T12:00:00-03:00"),
    ],
)
def test_accepted_scalar_values(type_name, value):
    assert validate(value, {"type": type_name}, phase="dataset") == value


@pytest.mark.parametrize(
    "type_name,value,expected_code",
    [
        ("string", 5, "wrong_type"),
        ("int", True, "wrong_type"),           # bool is not an int (A.6)
        ("int", 5.0, "wrong_type"),            # a real is not an integer
        ("int", "5", "wrong_type"),
        ("float", True, "wrong_type"),         # bool is not a float (A.6)
        ("float", "1.5", "wrong_type"),
        ("float", float("inf"), "invalid_value"),
        ("float", float("nan"), "invalid_value"),
        ("bool", 1, "wrong_type"),
        ("date", "2026-9-13", "invalid_value"),      # non-canonical
        ("date", "20260913", "invalid_value"),
        ("date", "2026-13-45", "invalid_value"),
        ("time", "12:00", "invalid_value"),
        ("datetime", "2026-09-13T12:00:00", "invalid_value"),   # no timezone
        ("datetime", "2026-09-13", "invalid_value"),
    ],
)
def test_rejected_scalar_values(type_name, value, expected_code):
    assert bad(value, {"type": type_name}).code == expected_code


def test_enum_member_accepted_and_non_member_rejected():
    node = {"type": "enum", "values": ["positive", "negative"]}
    assert validate("positive", node, phase="dataset") == "positive"
    assert bad("neutral", node).code == "invalid_enum"
    assert bad(1, node).code == "wrong_type"


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def test_artifact_path_must_be_an_existing_readable_file(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"png")
    assert validate(str(f), {"type": "image"}, phase="dataset") == str(f)
    assert bad(str(tmp_path / "missing.png"), {"type": "image"}).code == "artifact_not_found"
    assert bad(str(tmp_path), {"type": "image"}).code == "artifact_not_found"  # a directory


def test_resolve_hook_rewrites_the_stored_value_in_one_walk(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"png")
    ir = compile_schema({"image": "image"})
    seen: list[tuple[str, tuple[str, ...]]] = []

    def resolve(value, path):
        seen.append((value, path))
        return str(tmp_path / value)

    out = validate({"image": "a.png"}, ir, phase="dataset", resolve=resolve)
    assert out == {"image": str(f)}
    assert seen == [("a.png", ("image",))]


# ---------------------------------------------------------------------------
# equal — paper A.7 exact match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_name,a,b,expected",
    [
        ("string", "ACME SA", "ACME SA", True),
        ("string", "ACME SA", "acme sa", False),      # no case folding
        ("string", "ACME", " ACME ", False),          # no trimming
        ("float", 121.0, 121.0, True),
        ("float", 121, 121.0, True),                  # integers are valid reals
        ("float", 121.0, 121.01, False),              # no tolerance
        ("int", 5, 5, True),
        ("bool", True, True, True),
        ("bool", True, False, False),
        ("date", "2026-09-13", "2026-09-13", True),
        ("date", "2026-09-13", "2026-09-14", False),
        ("time", "12:00:00", "12:00:00.000", True),   # parsed time-of-day equality
        ("time", "12:00:00", "12:00:01", False),
        ("datetime", "2026-09-13T12:00:00Z", "2026-09-13T09:00:00-03:00", True),  # same instant
        ("datetime", "2026-09-13T12:00:00Z", "2026-09-13T12:00:00-03:00", False),
    ],
)
def test_scalar_equality(type_name, a, b, expected):
    assert equal(a, b, {"type": type_name}) is expected


def test_enum_equality():
    node = {"type": "enum", "values": ["a", "b"]}
    assert equal("a", "a", node) is True
    assert equal("a", "b", node) is False


def test_nested_object_equality():
    ir = compile_schema({"supplier": {"name": "string"}, "total": "float"})
    a = {"supplier": {"name": "ACME"}, "total": 1.0}
    assert equal(a, dict(a), ir) is True
    assert equal(a, {"supplier": {"name": "OTHER"}, "total": 1.0}, ir) is False


def test_artifact_equality_is_byte_for_byte(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"different")
    node = {"type": "image"}
    assert equal(str(a), str(b), node) is True
    assert equal(str(a), str(c), node) is False
    assert equal(str(a), str(a), node) is True


def test_artifact_equality_distinguishes_equal_sizes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"aaaa")
    b.write_bytes(b"bbbb")
    assert equal(str(a), str(b), {"type": "document"}) is False
