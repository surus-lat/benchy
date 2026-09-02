"""nb_tests — defines "broken" for the loss-first engine.

Run: /Users/dobleefe/benchy/.venv/bin/python -m pytest nb_tests -q
(the root pyproject sets testpaths=["tests"] — the OLD tests — so always
name nb_tests explicitly; the old tests/ is off-limits and must never run.)

Systems enter as plain callables. Stubs are loaded with stdlib importlib —
the engine has no system loader; a system IS its file, and Python already
knows how to import files.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nb import bench as nb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HELLO = ROOT / "bench" / "hello" / "bench.json"
SYSTEMS = ROOT / "bench" / "hello" / "systems"


def solve_from(path):
    """Load a system program file -> its `solve` callable (stdlib importlib)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve


GOOD = solve_from(SYSTEMS / "good.py")
DUMB = solve_from(SYSTEMS / "dumb.py")


def test_hello_benchmark_is_data():
    spec = json.loads(HELLO.read_text())
    assert spec["path"] == "/sentiment"
    assert len(spec["cases"]) == 6


def test_load_returns_a_callable():
    loss = nb.load("/sentiment")
    assert callable(loss)


def test_loss_ranks_stubs():
    loss = nb.load("/sentiment")
    assert loss(DUMB) > loss(GOOD)


def test_good_stub_scores_one():
    loss = nb.load("/sentiment")
    assert loss(GOOD) == 0.0


def test_dumb_stub_scores_half():
    loss = nb.load("/sentiment")
    loss(DUMB)
    # the 0.5 proves the scoring discriminates: per-case scores are real
    assert [c["score"] for c in loss.trace["cases"]] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert loss.trace["score"] == 0.5


def test_loss_is_pure_float():
    loss = nb.load("/sentiment")
    assert isinstance(loss(lambda t: "pos"), float)
    assert loss(lambda t: "pos") == 0.5  # always-pos: 3/6 wrong


def test_trace_is_the_receipt_of_last_eval():
    loss = nb.load("/sentiment")
    loss(DUMB)
    trace = loss.trace
    # the artifact is SELF-CONTAINED evidence: the exam (in+want), the answers
    # (got), the verdicts (score), the aggregate. A reader (human or a
    # software-3.0 optimizer) needs nothing else to interpret it. Cycle 10
    # tried deleting `want` (derivable from bench.json): a failed case became
    # "wrong, but about what?" — an unanchored projection. BARE_METAL.
    assert set(trace) == {"score", "cases"}
    assert len(trace["cases"]) == 6
    for c in trace["cases"]:
        assert set(c) == {"in", "want", "got", "score"}


def test_loss_score_is_one_minus_trace_score():
    loss = nb.load("/sentiment")
    for prog in (GOOD, DUMB):
        got = loss(prog)
        assert got == pytest.approx(1.0 - loss.trace["score"])


def test_trace_is_json_artifact(tmp_path):
    loss = nb.load("/sentiment")
    loss(DUMB)
    art = tmp_path / "artifact.json"
    art.write_text(json.dumps(loss.trace))
    back = json.loads(art.read_text())
    assert back["score"] == 0.5
    assert len(back["cases"]) == 6


def test_missing_path_raises():
    with pytest.raises(LookupError):
        nb.load("/nope")