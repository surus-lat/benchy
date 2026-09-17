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


def test_readme_quotes_the_engines_actual_size():
    """A number in prose drifts the moment it is not checked."""
    import ast

    readme = EXAMPLES.parent / "README.md"
    if not readme.is_file():
        pytest.skip("README.md is not distributed")

    total = 0
    for path in sorted((EXAMPLES.parent / "benchy").glob("*.py")):
        source = path.read_text()
        docstrings: set[int] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node, clean=False) and isinstance(node.body[0], ast.Expr):
                    docstrings.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
        total += sum(
            1
            for i, line in enumerate(source.splitlines(), 1)
            if line.strip() and not line.strip().startswith("#") and i not in docstrings
        )

    assert f"{total} lines of code" in readme.read_text(), f"README should say {total} lines of code"


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda p: p.parent.name)
def test_example_files_are_committed(benchmark):
    """An example that is not in the repository is a broken promise to every reader.

    `*.jsonl` is ignored repo-wide — correct for datasets and run artifacts, wrong for
    an example's exam. This was caught by a clean-clone run, not by the suite, so it
    is pinned here.
    """
    import subprocess

    for name in ("benchmark.yaml", "exam.jsonl", "system.py"):
        path = benchmark.parent / name
        assert path.is_file(), f"{path} is missing"
        ignored = subprocess.run(
            ["git", "check-ignore", str(path)], capture_output=True, cwd=EXAMPLES.parent
        )
        assert ignored.returncode != 0, f"{path} is gitignored and would not survive a clone"
