"""Every benchmark under `examples/` compiles and runs, and scores what its README says.

Documentation that is never executed rots. These run the real files through the real
CLI, so an example cannot drift from the engine without going red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchy.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ is not distributed")

BENCHMARKS = sorted(EXAMPLES.glob("*/benchmark.yaml"))


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda p: p.parent.name)
def test_example_compiles(benchmark, capsys):
    assert main(["compile", str(benchmark)]) == 0
    ir = json.loads(capsys.readouterr().out)
    assert ir["version"] == "1.0"


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda p: p.parent.name)
def test_example_runs(benchmark, capsys):
    adapter = f"{benchmark.parent / 'system.py'}:extractor"
    assert main(["run", str(benchmark), "--adapter", adapter]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["execution_errors"] == 0
    assert result["summary"]["invalid_outputs"] == 0


def test_there_is_at_least_one_example():
    assert BENCHMARKS


def test_invoices_example_scores_what_its_docstring_claims(capsys):
    benchmark = EXAMPLES / "invoices" / "benchmark.yaml"
    adapter = f"{EXAMPLES / 'invoices' / 'system.py'}:extractor"
    assert main(["run", str(benchmark), "--adapter", adapter]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["benchmark_score"] == pytest.approx(22 / 27)
    assert [r["score"] for r in result["results"]] == [1.0, 1.0, pytest.approx(4 / 9)]

    # The zero-weight field is still validated and still reported.
    tax_id = next(f for f in result["results"][0]["field_scores"] if f["path"] == ["supplier", "tax_id"])
    assert tax_id["weight"] == 0.0
