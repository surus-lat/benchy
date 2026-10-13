# -*- coding: utf-8 -*-
"""canonical parity: the vendored encoder against fixed vectors, and against
the IR the compiler emits today.

The compiler canonicalization is STRUCTURAL (``compile_benchmark`` always
returns the same fixed-key dict; there is no byte-level encoder in the engine
today). The vendored ``benchy.canonical`` supplies that byte-level form: the
golden ``IR_INVOICES_SHA256`` below locks the canonical bytes of the compiled
invoices example, so a change in the IR shape OR in the encoder breaks here.
The IR itself is untouched -- this test only reads it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchy import canonical
from benchy.canonical import canonical_bytes, canonical_text, sha256_of
from benchy.compiler import compile_benchmark

REPO = Path(__file__).resolve().parents[1]

# (name, value, canonical_text(value), sha256_of(value)) -- locked 2026-10-09
CANONICAL_VECTORS = [
    ("simple", {"b": 2, "a": 1}, "{\"a\":1,\"b\":2}",
     "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"),
    ("nested_key_order", {"z": {"y": 1, "x": 2}, "a": [{"b": 1, "a": 2}]},
     "{\"a\":[{\"a\":2,\"b\":1}],\"z\":{\"x\":2,\"y\":1}}",
     "cf1301ce6cf9f8f6e4c4b5f829901061fc0baac14a9f482c4c45e391f036f500"),
    ("unicode_preserved", {"á": "ñ", "lista": ["ü", "é"]},
     "{\"lista\":[\"ü\",\"é\"],\"á\":\"ñ\"}",
     "4a14442e3fd5db1af3b1d57fd060a49f311818ff40b895e8d6a53ac071c5738e"),
    ("floats", {"x": 1.5, "y": -0.25, "z": 1.0},
     "{\"x\":1.5,\"y\":-0.25,\"z\":1.0}",
     "7409cecca46127065ca81a7f47347949e613eeeb28d368a6717d9a16cd75708d"),
    ("bool_null", {"t": True, "f": False, "n": None},
     "{\"f\":false,\"n\":null,\"t\":true}",
     "22e00dc2f7b01420f940fbdbfbdf34fa0667cc6500186495023ba37722cbd05e"),
    ("empty_containers", {"l": [], "d": {}, "s": ""},
     "{\"d\":{},\"l\":[],\"s\":\"\"}",
     "7258cfb0cf2f129792d914aa59202e3abf72f6b50f75836247810a386efe3bff"),
    ("big_int", {"n": 123456789012345678901234567890},
     "{\"n\":123456789012345678901234567890}",
     "03b8f78ef5e8a305f4cda82942db0c6accec4046b40457ae19055ea5e4fff2ac"),
    ("string_escapes", {"s": "a\"b\nc\\d"},
     "{\"s\":\"a\\\"b\\nc\\\\d\"}",
     "2164458fc9dd39c823b716212c30ddb02726b4f527fb670a73bd407482425584"),
]

#: Golden content-id of the compiled examples/invoices IR, canonical-encoded.
IR_INVOICES_SHA256 = "d16e7bd83adb0d2a7ca602dd9980488bf57aaff894ea95bddbcb0d54eceecb40"


@pytest.mark.parametrize("name,value,expected_text,_", CANONICAL_VECTORS)
def test_canonical_text_fixed_vectors(name, value, expected_text, _):
    assert canonical_text(value) == expected_text
    assert canonical_bytes(value) == expected_text.encode("utf-8")


@pytest.mark.parametrize("name,value,_,expected_sha", CANONICAL_VECTORS)
def test_sha256_fixed_vectors(name, value, _, expected_sha):
    assert sha256_of(value) == expected_sha
    assert sha256_of(value) == hashlib.sha256(canonical_bytes(value)).hexdigest()


def test_key_order_does_not_change_bytes_or_id():
    first = {"b": 2, "a": {"y": 1, "x": [1, 2]}}
    second = {"a": {"x": [1, 2], "y": 1}, "b": 2}
    assert canonical_bytes(first) == canonical_bytes(second)
    assert sha256_of(first) == sha256_of(second)


def test_non_finite_numbers_are_rejected():
    for value in (float("nan"), float("inf"), {"x": [float("-inf")]}):
        with pytest.raises(ValueError):
            canonical_bytes(value)
        with pytest.raises(ValueError):
            sha256_of(value)


def test_compiled_ir_is_canonical_deterministic():
    text = (REPO / "examples" / "invoices" / "benchmark.yaml").read_text()
    first = compile_benchmark(text)
    second = compile_benchmark(text)
    assert canonical_bytes(first) == canonical_bytes(second)


def test_compiled_ir_matches_the_golden_content_id():
    text = (REPO / "examples" / "invoices" / "benchmark.yaml").read_text()
    ir = compile_benchmark(text)
    assert sha256_of(ir) == IR_INVOICES_SHA256


def test_canonical_imports_nothing_from_the_package():
    import inspect
    for line in inspect.getsource(canonical).splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "benchy" not in stripped
