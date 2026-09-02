"""nb tests — the definition of broken. pytest nb_tests (explicit arg, always)."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "nb"))
from benchy import Benchmark, grade, invoke, ok, exam  # noqa: E402

HERE = Path(__file__).parent.parent / "bench" / "hello"


@pytest.fixture
def bench():
    return Benchmark.load(HERE)


@pytest.fixture
def good(bench):
    return bench.systems["good"]


@pytest.fixture
def dumb(bench):
    return bench.systems["dumb"]


# ---- acceptance: the hello exam discriminates ----
def test_good_scores_one(bench, good):
    assert bench.run(good)["score"] == 1.0


def test_dumb_scores_half(bench, dumb):
    a = bench.run(dumb)
    assert a["score"] == 0.5
    assert len(a["cases"]) == 6
    assert [r["score"] for r in a["cases"]] == [1, 1, 1, 0, 0, 0]


def test_loss_ranks_stubs(bench, good, dumb):
    loss = bench.as_loss()
    assert loss(dumb) > loss(good)


def test_artifact_contract(bench, dumb):
    a = bench.run(dumb)
    assert a["ont"] == "/sentiment"
    assert a["score"] == 0.5 and a["loss"] == 0.5
    assert len(a["cases"]) == 6
    assert all(set(c) >= {"id", "in", "want", "got", "score"} for c in a["cases"])


# ---- task pillar ----
def test_task_enum_gates_pred():
    spec = {"out": {"enum": ["pos", "neg"]}}
    assert ok(spec, "pos") and not ok(spec, "meh")


def test_invalid_prediction_scores_zero(bench):
    # a system emitting junk outside the enum is graded 0, not crashed
    a = bench.run({"const": "junk"})
    assert a["score"] == 0.0


# ---- system pillar: one protocol, many shapes ----
def test_callable_system(bench):
    assert bench.run(lambda inp, ctx=None: "pos")["score"] == 0.5


def test_rule_system_is_data(bench, good):
    assert invoke(good, "this works great") == "pos"
    assert invoke(good, "UNIQUE") == "neg"          # default


def test_py_escape_hatch(bench, tmp_path):
    f = tmp_path / "sys.py"
    f.write_text("def predict(inp, ctx=None):\n    return 'pos'\n")
    a = bench.run({"py": str(f) + ":predict"})
    assert a["score"] == 0.5


# ---- scoring pillar ----
def test_fields_scoring_weights_importance():
    s = {"fields": {"total": 3, "tax": 1}}
    assert grade(s, {"total": 1, "tax": 2}, {"total": 1, "tax": 2}) == 1.0
    assert grade(s, {"total": 1, "tax": 2}, {"total": 1, "tax": 9}) == 0.75  # 3/4
    assert grade(s, {"total": 1}, {"total": 2}) == 0.0


# ---- data pillar: ontology, resume, fan-out ----
def test_load_by_ontology_path(bench):
    assert Benchmark.load("/sentiment").ont == "/sentiment"


def test_limit(bench, dumb):
    assert len(bench.run(dumb, limit=2)["cases"]) == 2


def test_resume_keeps_graded_work(bench, dumb, tmp_path):
    out = tmp_path / "run.json"
    bench.run(dumb, limit=2, out=out)
    n = 0
    # cheat: the artifact exists mid-run; resume must keep those 2 rows
    assert len(json.loads(out.read_text())["cases"]) == 2
    a = bench.run(dumb, out=out)
    assert len(a["cases"]) == 6
    assert out.exists()


def test_resume_is_true_resume(bench, dumb, tmp_path, monkeypatch):
    # prove graded rows are NOT re-invoked: the system is replaced by one that
    # would fail every case; resume must still return the graded artifact.
    out = tmp_path / "run.json"
    bench.run(dumb, limit=6, out=out)
    calls = []
    def spy(inp, ctx=None):
        calls.append(inp)
        return "pos"
    a = bench.run(spy, out=out)   # same artifact path, new system
    assert len(calls) == 0           # every case already graded: zero re-takes
    assert a["score"] == 0.5


def test_workers_fan_out(bench, dumb):
    assert bench.run(dumb, workers=3)["score"] == 0.5


def test_concurrent_matches_serial(bench, dumb):
    serial = bench.run(dumb)
    concur = bench.run(dumb, workers=6)
    assert serial["cases"] == concur["cases"]


# ---- CLI ----
def test_cli_run(capsys, monkeypatch, bench):
    import benchy
    monkeypatch.setattr("sys.argv", ["benchy", "run", str(HERE), "dumb", "--out", str(Path("/tmp/cli.json"))])
    rc = benchy.main(["run", str(HERE), "dumb", "--out", "/tmp/cli.json"])
    assert rc == 0
    a = json.loads(capsys.readouterr().out)
    assert a["score"] == 0.5
    assert json.loads(Path("/tmp/cli.json").read_text())["score"] == 0.5


def test_cli_loss(capsys):
    import benchy
    assert benchy.main(["loss", str(HERE), "dumb"]) == 0
    assert float(capsys.readouterr().out.strip()) == 0.5