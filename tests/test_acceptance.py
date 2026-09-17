"""P9 — the CLI, and C33: the engine runs from persisted IR with no source YAML.

This is the build plan's final acceptance test. It is the observable form of paper
A.8's invariant: if the engine can still execute after the YAML is deleted, then it
genuinely compiled the semantics once rather than reinterpreting source.
"""

from __future__ import annotations

import json
import textwrap

import pytest
from conftest import CANONICAL, edit

from benchy.cli import main

BENCHMARK = edit(
    program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
    scoring={"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
)

EXAM = [
    {"input": {"text": "invoice one"}, "expected": {"supplier": "ACME", "total": 121.0}},
    {"input": {"text": "invoice two"}, "expected": {"supplier": "Example", "total": 242.0}},
]

#: A fake AI-system: right on the first example, wrong on every other.
ADAPTER = textwrap.dedent('''
    async def extractor(input_object):
        if input_object["text"] == "invoice one":
            return {"supplier": "ACME", "total": 121.0}
        return {"supplier": "WRONG", "total": 0.0}
''')


@pytest.fixture
def bench(tmp_path):
    (tmp_path / "benchmark.yaml").write_text(BENCHMARK)
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in EXAM))
    (tmp_path / "system.py").write_text(ADAPTER)
    return tmp_path


def cli(*argv) -> int:
    return main([str(a) for a in argv])


def read(path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# benchy compile
# ---------------------------------------------------------------------------

def test_compile_writes_ir(bench):
    assert cli("compile", bench / "benchmark.yaml", "-o", bench / "ir.json") == 0
    ir = read(bench / "ir.json")
    assert ir["version"] == "1.0"
    assert ir["scoring"]["dimensions"][1] == {"path": ["total"], "weight": 3.0}


def test_compile_prints_ir_to_stdout(bench, capsys):
    assert cli("compile", bench / "benchmark.yaml") == 0
    assert json.loads(capsys.readouterr().out)["ontology_version"] == "1.0"


def test_compile_reports_a_structured_diagnostic_and_exits_nonzero(tmp_path, capsys):
    (tmp_path / "bad.yaml").write_text(CANONICAL + "\nextra: 1\n")
    assert cli("compile", tmp_path / "bad.yaml") == 1
    error = json.loads(capsys.readouterr().err)
    assert (error["phase"], error["code"]) == ("compile", "unknown_key")


def test_missing_source_file_is_reported(tmp_path, capsys):
    assert cli("compile", tmp_path / "absent.yaml") == 1
    assert json.loads(capsys.readouterr().err)["code"] == "data_not_found"


# ---------------------------------------------------------------------------
# benchy run
# ---------------------------------------------------------------------------

def test_run_from_yaml(bench, capsys):
    assert cli("run", bench / "benchmark.yaml", "--adapter", f"{bench / 'system.py'}:extractor") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"] == {"examples": 2, "valid": 2, "invalid_outputs": 0, "execution_errors": 0}
    assert result["benchmark_score"] == 0.5  # one perfect example, one entirely wrong


def test_run_resolves_the_workspace_from_the_source_directory(bench):
    """`data.path: ./exam.jsonl` is relative to the benchmark file, not the cwd."""
    assert cli("run", bench / "benchmark.yaml", "--adapter", f"{bench / 'system.py'}:extractor",
               "-o", bench / "result.json") == 0
    assert read(bench / "result.json")["summary"]["examples"] == 2


@pytest.mark.parametrize("spec", ["nomodule", "nomodule:attr", "system.py", ":attr"])
def test_malformed_or_missing_adapter_is_reported(bench, spec, capsys):
    assert cli("run", bench / "benchmark.yaml", "--adapter", spec) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "adapter_not_bound"


def test_adapter_module_without_the_named_attribute_is_reported(bench, capsys):
    assert cli("run", bench / "benchmark.yaml", "--adapter", f"{bench / 'system.py'}:absent") == 1
    assert json.loads(capsys.readouterr().err)["code"] == "adapter_not_bound"


def test_dataset_error_aborts_with_no_benchmark_score(bench, capsys):
    (bench / "exam.jsonl").write_text('{"input": {"text": "a"}, "expected": {"supplier": "x"}}')
    assert cli("run", bench / "benchmark.yaml", "--adapter", f"{bench / 'system.py'}:extractor") == 1
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert (error["phase"], error["code"]) == ("dataset", "missing_field")
    assert captured.out == ""  # no partial result document is emitted


# ---------------------------------------------------------------------------
# C33 — the acceptance test
# ---------------------------------------------------------------------------

def test_c33_engine_runs_from_persisted_ir_with_the_yaml_deleted(bench):
    adapter = f"{bench / 'system.py'}:extractor"

    # 1. compile, and run from source.
    assert cli("compile", bench / "benchmark.yaml", "-o", bench / "ir.json") == 0
    assert cli("run", bench / "benchmark.yaml", "--adapter", adapter, "-o", bench / "from-yaml.json") == 0

    # 2. remove every trace of the source.
    (bench / "benchmark.yaml").unlink()
    assert not (bench / "benchmark.yaml").exists()

    # 3. run again from the persisted IR alone.
    assert cli("run", bench / "ir.json", "--adapter", adapter, "-o", bench / "from-ir.json") == 0

    # 4. the two runs are indistinguishable.
    assert read(bench / "from-ir.json") == read(bench / "from-yaml.json")


def test_c33_ir_alone_carries_every_semantic_the_engine_needs(bench):
    """No ontology lookup, no YAML, no schema inference at execution time."""
    assert cli("compile", bench / "benchmark.yaml", "-o", bench / "ir.json") == 0
    (bench / "benchmark.yaml").unlink()
    ir = read(bench / "ir.json")
    assert ir["program"]["output"]["fields"]["total"] == {"type": "float"}
    assert ir["scoring"]["evaluator"] == "exact_match"
    assert ir["scoring"]["benchmark_aggregator"] == "mean"
    assert ir["data"]["format"] == "jsonl"
