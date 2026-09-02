"""The hello acceptance bar (search/GOLEM.md): run offline, end to end —
CLI-first: the three commands ARE the spec; these tests pin every word."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _cli(*args, cwd=ROOT, env_nb=True):
    import os
    env = {**os.environ, "PYTHONPATH": str(ROOT)} if env_nb else None
    return subprocess.run([PY, "-m", "nb", *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def _run(system, *extra):
    r = _cli("run", "/sentiment", system, *extra)
    assert r.returncode == 0, r.stderr
    return r


# --- the vision loop: create -> run -> grade -> export as loss -------------

def test_run_good_scores_one_and_writes_artifact():
    r = _run("good")
    out = ROOT / "runs" / f"sentiment-good.json"
    a = json.loads(out.read_text())
    assert a["score"] == 1.0 and a["system"] == "good"
    assert len(a["cases"]) == 6
    assert a["benchmark"] == "/sentiment"


def test_run_dumb_scores_half():
    r = _run("dumb")
    a = json.loads((ROOT / "runs" / "sentiment-dumb.json").read_text())
    assert a["score"] == 0.5


def test_artifact_interprets_alone():
    a = json.loads((ROOT / "runs" / "sentiment-dumb.json").read_text())
    for c in a["cases"]:
        assert set(c) >= {"input", "expected", "prediction", "score"}


def test_limit_is_the_smoke_valve():
    _run("dumb", "--limit", "4")  # 3 pos + 1 neg expected -> dumb scores 0.75
    a = json.loads((ROOT / "runs" / "sentiment-dumb.json").read_text())
    assert len(a["cases"]) == 4 and a["score"] == 0.75


def test_as_loss_ranks_dumb_above_good():
    sys.path.insert(0, str(ROOT))
    from nb.exam import locate
    exam = locate(ROOT / "bench", "/sentiment")
    sys.path.insert(0, str(ROOT / "bench" / "hello"))
    import systems
    loss = exam.as_loss()
    assert loss(systems.dumb) > loss(systems.good)


def test_report_reads_the_graded_run():
    _run("dumb")
    r = _cli("report", "runs/sentiment-dumb.json")
    assert r.returncode == 0, r.stderr
    assert "score 0.500" in r.stdout and "loss 0.500" in r.stdout
    assert "fail" in r.stdout and "pass" in r.stdout


def test_new_scaffolds_then_runs():
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copytree(ROOT / "bench", td / "bench")
        r = _cli("new", "sentiment_es", cwd=td)
        assert r.returncode == 0, r.stderr
        f = td / "bench" / "sentiment_es" / "benchmark.json"
        assert json.loads(f.read_text())["path"] == "/sentiment_es"
        # the scaffold is immediately runnable (zero cases -> loud failure)
        r = _cli("run", "/sentiment_es", "todo", cwd=td)
        assert r.returncode != 0 and "no cases" in r.stderr


def test_unknown_benchmark_path_is_loud():
    r = _cli("run", "/nope", "good")
    assert r.returncode != 0 and "no benchmark" in r.stderr


def test_unknown_system_is_loud():
    r = _cli("run", "/sentiment", "nonexistent")
    assert r.returncode != 0


def test_unknown_benchmark_key_is_loud():
    sys.path.insert(0, str(ROOT))
    from nb.exam import locate
    bad = ROOT / "bench" / "bad"
    bad.mkdir(exist_ok=True)
    (bad / "benchmark.json").write_text(json.dumps(
        {"path": "/bad", "task": {}, "cases": [], "verbosity": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        locate(ROOT / "bench", "/bad")
    shutil.rmtree(bad) if False else None
    import shutil as _sh
    _sh.rmtree(bad)