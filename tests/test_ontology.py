"""P3 — the shared SURUS ontology: membership and task-to-program validation.

Covers conformance cases C05, C06, C09, C10 (C07/C08 withdrawn with `transcribe`).
"""

from __future__ import annotations

import pytest

from benchy import ontology
from benchy.errors import BenchyError
from benchy.types import compile_schema


def registry():
    return ontology.load("1.0")


def check(task, domain="finance", language="es", *, inp=None, out=None):
    """Validate a classification + program pair, raising BenchyError on rejection."""
    benchmark = {"task": task, "domain": domain, "language": language}
    input_ir = compile_schema(inp or {"text": "string"})
    output_ir = compile_schema(out or {"value": "string"})
    ontology.check_classification(benchmark, registry())
    ontology.check_program(benchmark, input_ir, output_ir, registry())


def fails(**kw) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        check(**kw)
    return exc.value


# ---------------------------------------------------------------------------
# registry loading (spec §2, amendment §4)
# ---------------------------------------------------------------------------

def test_registry_1_0_loads_from_the_packaged_store():
    r = registry()
    assert r["version"] == "1.0"
    assert set(r["tasks"]) == {"extract", "classify", "translate"}


def test_transcribe_is_not_in_ontology_1_0():
    assert "transcribe" not in registry()["tasks"]


def test_declaring_transcribe_fails_loudly_as_an_unknown_task():
    assert fails(task="transcribe").code == "unknown_task"


def test_unsupported_ontology_version_is_rejected():
    with pytest.raises(BenchyError) as exc:
        ontology.load("2.0")
    assert exc.value.code == "unsupported_ontology_version"


def test_benchmark_never_supplies_an_ontology_path():
    # Amendment §4: the runtime resolves the registry by version. `load` takes a
    # version, never a filepath.
    with pytest.raises(BenchyError):
        ontology.load("./some/ontology.yaml")


# ---------------------------------------------------------------------------
# classification membership (spec §3)
# ---------------------------------------------------------------------------

def test_unknown_task_domain_and_language_each_have_their_own_code():
    assert fails(task="summarize").code == "unknown_task"
    assert fails(task="extract", domain="astrology").code == "unknown_domain"
    assert fails(task="extract", language="kl").code == "unknown_language"


def test_registered_classification_passes():
    check("extract", domain="legal", language="pt")


# ---------------------------------------------------------------------------
# extract (spec §5) — no task-specific structural constraint
# ---------------------------------------------------------------------------

def test_extract_accepts_an_arbitrary_fixed_output_schema():
    check("extract", inp={"image": "image"}, out={"n": "string", "d": "date", "t": "float"})


def test_extract_accepts_nested_output():
    check("extract", out={"supplier": {"name": "string"}, "total": "float"})


def test_non_translation_task_rejects_a_source_target_language():
    err = fails(task="extract", language={"source": "es", "target": "en"})
    assert err.code == "unknown_language"


# ---------------------------------------------------------------------------
# classify (spec §5) — exactly one output leaf, and it is enum
# ---------------------------------------------------------------------------

def test_c05_classify_with_one_enum_output_leaf_is_valid():
    check("classify", out={"sentiment": {"enum": ["positive", "negative"]}})


def test_c06_classify_with_a_string_output_is_a_task_program_mismatch():
    assert fails(task="classify", out={"sentiment": "string"}).code == "task_program_mismatch"


def test_classify_with_two_output_leaves_is_a_mismatch():
    out = {"a": {"enum": ["x"]}, "b": {"enum": ["y"]}}
    assert fails(task="classify", out=out).code == "task_program_mismatch"


def test_classify_accepts_any_input_schema():
    check("classify", inp={"image": "image"}, out={"label": {"enum": ["a", "b"]}})


# ---------------------------------------------------------------------------
# translate (spec §5)
# ---------------------------------------------------------------------------

def test_c10_translate_with_source_target_language_is_valid():
    check(
        "translate",
        domain="general",
        language={"source": "es", "target": "en"},
        inp={"text": "string"},
        out={"translation": "string"},
    )


def test_c09_translate_with_a_scalar_language_is_rejected():
    err = fails(task="translate", domain="general", language="es", out={"translation": "string"})
    assert err.code == "task_program_mismatch"
    assert "source" in err.message


def test_translate_language_members_must_be_registered():
    err = fails(
        task="translate", domain="general",
        language={"source": "es", "target": "kl"},
        out={"translation": "string"},
    )
    assert err.code == "unknown_language"


def test_translate_language_object_rejects_extra_or_missing_keys():
    for language in ({"source": "es"}, {"source": "es", "target": "en", "via": "pt"}):
        assert fails(task="translate", domain="general", language=language,
                     out={"translation": "string"}).code == "task_program_mismatch"


def test_translate_requires_at_least_one_string_input_leaf():
    err = fails(
        task="translate", domain="general", language={"source": "es", "target": "en"},
        inp={"audio": "audio"}, out={"translation": "string"},
    )
    assert err.code == "task_program_mismatch"


def test_translate_requires_exactly_one_string_output_leaf():
    base = dict(task="translate", domain="general", language={"source": "es", "target": "en"})
    assert fails(**base, out={"translation": "float"}).code == "task_program_mismatch"
    assert fails(**base, out={"a": "string", "b": "string"}).code == "task_program_mismatch"


def test_translate_accepts_extra_non_string_input_fields():
    check(
        "translate", domain="general", language={"source": "es", "target": "en"},
        inp={"text": "string", "page": "int"}, out={"translation": "string"},
    )
