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

#: Examples shipping their own `system.py` run offline. Examples whose `ai-system` is
#: a `model` need credentials and a live provider, so they are compiled but not run.
OFFLINE = [b for b in BENCHMARKS if (b.parent / "system.py").is_file()]


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda p: p.parent.name)
def test_example_compiles(benchmark, capsys):
    assert main(["compile", str(benchmark)]) == 0
    ir = json.loads(capsys.readouterr().out)
    assert ir["version"] == "1.0"


@pytest.mark.parametrize("benchmark", OFFLINE, ids=lambda p: p.parent.name)
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


CORE = ["__init__", "errors", "types", "ontology", "compiler", "data", "score",
        "adapter", "run", "cli"]


def _code_lines(path):
    import ast

    source = path.read_text()
    docstrings: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) and isinstance(node.body[0], ast.Expr):
                docstrings.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    return sum(
        1
        for i, line in enumerate(source.splitlines(), 1)
        if line.strip() and not line.strip().startswith("#") and i not in docstrings
    )


def test_readme_quotes_the_engines_actual_size():
    """A number in prose drifts the moment it is not checked.

    The engine and the optional provider adapter are counted separately, because the
    README lists them separately and `providers.py` is deliberately outside the core.
    """
    readme = EXAMPLES.parent / "README.md"
    if not readme.is_file():
        pytest.skip("README.md is not distributed")

    benchy = EXAMPLES.parent / "benchy"
    engine = sum(_code_lines(benchy / f"{name}.py") for name in CORE)
    provider = _code_lines(benchy / "providers.py")

    text = readme.read_text()
    assert f"{engine} lines of code" in text, f"README should say {engine} lines of code"
    assert f"plus {provider} in the optional provider adapter" in text, f"provider is {provider}"


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda p: p.parent.name)
def test_every_file_an_example_needs_is_committed(benchmark):
    """An example that is not in the repository is a broken promise to every reader.

    `*.jsonl` is ignored repo-wide — right for datasets and run artifacts, wrong for an
    example's exam, which once left `examples/invoices` broken in every clone while the
    suite stayed green. Artifacts the exam points at count too.

    Note what `git check-ignore` does and does not prove: a *tracked* file is never
    reported as ignored, so this fires only for files not yet committed. That is exactly
    where the bug lives — a new file silently skipped by `git add -A`, never committed,
    and missing from every clone. Files already tracked are safe by construction.
    """
    import subprocess

    needed = [benchmark, benchmark.parent / "exam.jsonl"]
    if benchmark in OFFLINE:
        needed.append(benchmark.parent / "system.py")
    for line in (benchmark.parent / "exam.jsonl").read_text().splitlines():
        if line.strip():
            needed += [
                benchmark.parent / value
                for value in json.loads(line)["input"].values()
                if isinstance(value, str) and "/" in value
            ]

    for path in needed:
        assert path.is_file(), f"{path} is missing"
        ignored = subprocess.run(["git", "check-ignore", str(path)], capture_output=True, cwd=EXAMPLES.parent)
        assert ignored.returncode != 0, f"{path} is gitignored and would not survive a clone"
