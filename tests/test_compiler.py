"""P5 — the whole compiler: YAML -> canonical JSON IR (paper §7, A.9, spec §17).

Covers C01 and C03, plus the determinism and "no unresolved shorthand" properties.
"""

from __future__ import annotations

import json

import pytest

from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from conftest import CANONICAL, edit


def fails(text: str) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        compile_benchmark(text)
    return exc.value


# ---------------------------------------------------------------------------
# C01 — the canonical benchmark
# ---------------------------------------------------------------------------

def test_c01_canonical_benchmark_compiles():
    ir = compile_benchmark(CANONICAL)
    assert ir["version"] == "1.0"
    assert ir["ontology_version"] == "1.0"
    assert ir["benchmark"] == {"task": "extract", "domain": "finance", "language": "es"}


def test_ir_matches_the_documented_shape():
    ir = compile_benchmark(CANONICAL)
    assert set(ir) == {
        "version", "ontology_version", "benchmark", "program", "scoring", "data", "ai-system",
    }
    assert ir["program"]["input"] == {"type": "object", "fields": {"image": {"type": "image"}}}
    assert ir["program"]["output"]["fields"]["total"] == {"type": "float"}
    assert ir["data"] == {"path": "./data/invoices.jsonl", "format": "jsonl"}
    assert ir["ai-system"] == {"type": "external", "id": "invoice-extractor-v7"}


def test_scoring_dimensions_are_resolved_in_output_order():
    ir = compile_benchmark(CANONICAL)
    assert [(d["path"], d["weight"]) for d in ir["scoring"]["dimensions"]] == [
        (["invoice_number"], 1.0), (["date"], 1.0), (["supplier"], 1.0),
        (["subtotal"], 1.0), (["total"], 5.0),
    ]


def test_ir_is_json_serializable_and_deterministic():
    a = json.dumps(compile_benchmark(CANONICAL), sort_keys=True)
    b = json.dumps(compile_benchmark(CANONICAL), sort_keys=True)
    assert a == b


def test_ir_contains_no_unresolved_source_shorthand():
    """Enum shorthand and bare type tokens become explicit IR nodes."""
    ir = compile_benchmark(edit(
        benchmark={"task": "classify", "domain": "retail", "language": "pt"},
        program={"input": {"text": "string"}, "output": {"sentiment": {"enum": ["a", "b"]}}},
        scoring={"weights": {"sentiment": 1}, "aggregator": "weighted_mean"},
    ))
    assert ir["program"]["output"]["fields"]["sentiment"] == {"type": "enum", "values": ["a", "b"]}


def test_data_path_is_kept_verbatim_so_the_ir_stays_portable():
    ir = compile_benchmark(CANONICAL)
    assert ir["data"]["path"] == "./data/invoices.jsonl"


def test_compiling_does_not_touch_the_filesystem_for_data():
    """`data.path` need not exist to compile; it is resolved at run time."""
    compile_benchmark(edit(data={"path": "./nowhere/absent.jsonl"}))


# ---------------------------------------------------------------------------
# root structure (spec §1) and versions (spec §2)
# ---------------------------------------------------------------------------

def test_c03_unknown_root_key_is_rejected():
    err = fails(CANONICAL + "\nextra: 1\n")
    assert err.code == "unknown_key"
    assert "extra" in err.message


@pytest.mark.parametrize(
    "key",
    ["version", "ontology_version", "benchmark", "program", "scoring", "data", "ai-system"],
)
def test_every_root_key_is_required(key):
    assert fails(edit(**{key.replace("-", "_"): None})).code == "missing_key"


# `None` would delete the key (see conftest.edit); absence is covered by
# test_every_root_key_is_required.
@pytest.mark.parametrize("version", ["1.1", "2.0", 1.0, "1", ""])
def test_unsupported_spec_version_is_rejected(version):
    assert fails(edit(version=version)).code == "unsupported_version"


def test_unsupported_ontology_version_is_rejected():
    assert fails(edit(ontology_version="9.9")).code == "unsupported_ontology_version"


# ---------------------------------------------------------------------------
# section structure
# ---------------------------------------------------------------------------

def test_benchmark_section_is_closed():
    assert fails(edit(benchmark={"task": "extract", "domain": "finance", "language": "es", "x": 1})).code == "unknown_key"
    assert fails(edit(benchmark={"task": "extract", "domain": "finance"})).code == "missing_key"


def test_program_section_is_closed():
    assert fails(edit(program={"input": {"a": "string"}})).code == "missing_key"
    assert fails(edit(program={"input": {"a": "string"}, "output": {"b": "string"}, "x": 1})).code == "unknown_key"


def test_program_input_and_output_must_be_non_empty_mappings():
    assert fails(edit(program={"input": {}, "output": {"total": "float"}})).code == "invalid_schema"
    assert fails(edit(program={"input": {"a": "string"}, "output": {}})).code == "invalid_schema"


@pytest.mark.parametrize("anonymous", ["string", {"enum": ["a", "b"]}])
def test_anonymous_root_scalar_program_io_is_rejected(anonymous):
    """Paper §10: anonymous root scalar inputs or outputs are outside the language."""
    assert fails(edit(program={"input": anonymous, "output": {"total": "float"}})).code == "invalid_schema"


def test_data_section_is_closed_and_path_must_be_a_non_empty_string():
    assert fails(edit(data={"path": "./d.jsonl", "format": "csv"})).code == "unknown_key"
    assert fails(edit(data={})).code == "missing_key"
    for bad in ("", 5, None):
        assert fails(edit(data={"path": bad})).code == "invalid_value"


# ---------------------------------------------------------------------------
# ai-system (spec, handoff §3)
# ---------------------------------------------------------------------------

def test_external_ai_system_requires_type_and_id_only():
    assert fails(edit(ai_system={"type": "external"})).code == "missing_key"
    assert fails(edit(ai_system={"type": "external", "id": "x", "provider": "y"})).code == "unknown_key"
    assert fails(edit(ai_system={"type": "external", "id": ""})).code == "invalid_ai_system"


def test_model_ai_system_requires_provider_and_model():
    ir = compile_benchmark(edit(ai_system={"type": "model", "provider": "openai", "model": "m"}))
    assert ir["ai-system"] == {"type": "model", "provider": "openai", "model": "m"}
    assert fails(edit(ai_system={"type": "model", "provider": "openai"})).code == "missing_key"


def test_model_ai_system_accepts_optional_prompt_and_parameters():
    system = {"type": "model", "provider": "openai", "model": "m",
              "prompt": "./p.md", "parameters": {"temperature": 0}}
    assert compile_benchmark(edit(ai_system=system))["ai-system"] == system


def test_absent_optional_keys_are_not_injected_into_the_ir():
    """Compilation does not inject hidden defaults (paper §1)."""
    ir = compile_benchmark(edit(ai_system={"type": "model", "provider": "openai", "model": "m"}))
    assert "prompt" not in ir["ai-system"]
    assert "parameters" not in ir["ai-system"]


def test_unknown_ai_system_type_is_rejected():
    assert fails(edit(ai_system={"type": "wizard", "id": "x"})).code == "invalid_ai_system"
    assert fails(edit(ai_system={"id": "x"})).code == "invalid_ai_system"


# ---------------------------------------------------------------------------
# diagnostic order (handoff §15)
# ---------------------------------------------------------------------------

def test_root_key_check_precedes_the_version_check():
    assert fails(edit(version="9.9") + "\nextra: 1\n").code == "unknown_key"


def test_classification_membership_precedes_program_grammar():
    text = edit(
        benchmark={"task": "extract", "domain": "astrology", "language": "es"},
        program={"input": {"a": "nonsense_type"}, "output": {"total": "float"}},
    )
    assert fails(text).code == "unknown_domain"


def test_program_grammar_precedes_task_program_validation():
    text = edit(
        benchmark={"task": "classify", "domain": "finance", "language": "es"},
        program={"input": {"a": "nonsense_type"}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
    )
    assert fails(text).code == "invalid_schema"


def test_task_program_validation_precedes_weight_coverage():
    text = edit(
        benchmark={"task": "classify", "domain": "finance", "language": "es"},
        program={"input": {"a": "string"}, "output": {"total": "float"}},
        scoring={"weights": {"bogus": 1}, "aggregator": "weighted_mean"},
    )
    assert fails(text).code == "task_program_mismatch"
