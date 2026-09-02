"""nb_tests — defines 'broken' for the s05 engine (benchmark-as-directory)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nb import bench  # noqa: E402

HELLO = ROOT / "bench" / "hello"


def test_benchmark_is_data_locatable_by_ontology_path():
    bench_md = bench.load(HELLO)
    assert bench_md["task"]["task"] == "sentiment"


def test_hello_runs_offline_good_stub_scores_1():
    result = bench.run(HELLO, bench.load_system(HELLO, "good"))
    assert result["score"] == 1.0


def test_hello_runs_offline_dumb_stub_scores_half():
    result = bench.run(HELLO, bench.load_system(HELLO, "dumb"))
    assert result["score"] == 0.5


def test_as_loss_ranks_dumb_worse_than_good():
    good = bench.run(HELLO, bench.load_system(HELLO, "good"))
    dumb = bench.run(HELLO, bench.load_system(HELLO, "dumb"))
    assert bench.as_loss(dumb) > bench.as_loss(good)


def test_artifact_has_per_case_scores_and_aggregate():
    result = bench.run(HELLO, bench.load_system(HELLO, "dumb"))
    assert len(result["cases"]) == 6
    per = [c["score"] for c in result["cases"]]
    assert sum(per) == 3
    assert result["score"] == pytest.approx(0.5)


def test_zero_user_python():
    """The benchmark directory must contain ONLY data files — no .py."""
    for p in HELLO.rglob("*"):
        assert p.suffix != ".py", f"benchmark contains python: {p}"


def test_dumb_case_scores_prove_discrimination():
    """Dumb gets the 3 pos cases right, the 3 neg cases wrong."""
    result = bench.run(HELLO, bench.load_system(HELLO, "dumb"))
    wrong = [c for c in result["cases"] if c["score"] == 0]
    assert len(wrong) == 3
    assert all(c["expected"] == "neg" for c in wrong)


def test_cli_runs(tmp_path, capsys, monkeypatch):
    import subprocess
    import sys as _sys
    monkeypatch.chdir(tmp_path)
    code = subprocess.call(
        [_sys.executable, str(ROOT / "nb" / "bench.py"), str(HELLO), "good"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert code == 0
    saved = json.loads((tmp_path / "runs/hello/good.json").read_text())
    assert saved["score"] == 1.0