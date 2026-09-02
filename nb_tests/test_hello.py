"""The hello acceptance bar (search/GOLEM.md): run offline, end to end —
CLI-first: the three commands ARE the spec; these tests pin every word."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# cycle 9: the scaffold teaches a BLANK task object — any typed example is a
# type-lie the engine does not check; honesty = presence, not semantics
EXAM_DATA = '{"task": {}, "cases": []}'


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
    assert a["score"] == 1.0
    assert len(a["cases"]) == 6


def test_artifact_interprets_alone_it_carries_its_own_identity():
    # the artifact IS the report (c4): it must name WHO took and WHICH exam
    # — a JSON that leans on its filename does not interpret alone.
    _run("good")
    a = json.loads((ROOT / "runs" / "sentiment-good.json").read_text())
    assert a["system"] == "good" and a["benchmark"] == "/sentiment"


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


def test_artifact_names_its_own_scope():
    # c10 (c2's open probe, test-first): a smoke run must not masquerade as
    # a full run — the artifact interprets alone, so it carries the exam's
    # size; graded prefix vs full exam is readable from the JSON itself,
    # with no "absent key means full" convention to trust
    _run("dumb", "--limit", "4")
    a = json.loads((ROOT / "runs" / "sentiment-dumb.json").read_text())
    assert len(a["cases"]) == 4 and a["total"] == 6


def test_artifact_carries_the_loss_stdout_only_echoes():
    # c13: the ack line computed the loss IN THE CLI (1 - score) — a second
    # address of as_loss's inner formula — and it was the one stdout fact the
    # artifact lacked (the report test had to re-derive it too: three
    # addresses for one formula).  Grading now composes the loss ONCE; stdout
    # carries nothing the artifact doesn't.
    r = _run("dumb")
    a = json.loads((ROOT / "runs" / "sentiment-dumb.json").read_text())
    assert a["loss"] == 0.5
    assert "loss=0.50" in r.stdout


def test_as_loss_ranks_dumb_above_good():
    import importlib.util
    sys.path.insert(0, str(ROOT))
    from nb.exam import locate
    exam = locate(ROOT / "bench", "/sentiment")
    spec = importlib.util.spec_from_file_location("systems", ROOT / "bench" / "sentiment" / "systems.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loss = exam.as_loss()
    assert loss(mod.dumb) > loss(mod.good)


def test_report_reads_the_graded_run():
    # the artifact IS the report: it interprets alone (per-case detail + loss)
    _run("dumb")
    a = json.loads((ROOT / "runs" / "sentiment-dumb.json").read_text())
    assert a["score"] == 0.5
    fails = [c for c in a["cases"] if not c["score"]]
    assert len(fails) == 3 and a["loss"] == 0.5


def test_new_scaffolds_then_runs():
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copytree(ROOT / "bench", td / "bench")
        r = _cli("new", "sentiment_es", cwd=td)
        assert r.returncode == 0, r.stderr
        f = td / "bench" / "sentiment_es" / "benchmark.json"
        # the scaffold is pure data: {task, cases} — the directory IS the
        # ontology path (no `path` field; cycle 6)
        assert json.loads(f.read_text()) == json.loads(EXAM_DATA)
        # the scaffold is immediately runnable (zero cases -> loud failure)
        r = _cli("run", "/sentiment_es", "todo", cwd=td)
        assert r.returncode != 0 and "no cases" in r.stderr


def test_unknown_benchmark_path_is_loud():
    r = _cli("run", "/nope", "good")
    assert r.returncode != 0 and "no benchmark" in r.stderr


def test_unknown_system_is_loud():
    r = _cli("run", "/sentiment", "nonexistent")
    assert r.returncode != 0


def test_missing_task_is_loud():
    # an exam without its first pillar is not an exam — locate must enforce
    # the {task, cases} format its own error message claims (c9: the
    # scaffold task-lie probe; a teaching scaffold for an unenforced
    # format is a false claim)
    sys.path.insert(0, str(ROOT))
    from nb.exam import locate
    bad = ROOT / "bench" / "bad"
    bad.mkdir(exist_ok=True)
    (bad / "benchmark.json").write_text(json.dumps(
        {"cases": [{"input": "x", "expected": "y"}]}))
    with pytest.raises(ValueError, match="task"):
        locate(ROOT / "bench", "/bad")
    import shutil as _sh
    _sh.rmtree(bad)


def test_unknown_benchmark_key_is_loud():
    sys.path.insert(0, str(ROOT))
    from nb.exam import locate
    bad = ROOT / "bench" / "bad"
    bad.mkdir(exist_ok=True)
    (bad / "benchmark.json").write_text(json.dumps(
        {"path": "/bad", "task": {}, "cases": [], "verbosity": 1}))
    # c9: one honest message diagnoses missing AND junk keys (exact-set check)
    with pytest.raises(ValueError, match="exactly"):
        locate(ROOT / "bench", "/bad")
    shutil.rmtree(bad) if False else None
    import shutil as _sh
    _sh.rmtree(bad)