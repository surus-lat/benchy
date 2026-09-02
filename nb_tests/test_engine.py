"""nb_tests — defines "broken" for the loss-first engine.

Run: /Users/dobleefe/benchy/.venv/bin/python -m pytest nb_tests -q
(the root pyproject sets testpaths=["tests"] — the OLD tests — so always
name nb_tests explicitly; the old tests/ is off-limits and must never run.)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nb import bench as nb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HELLO = ROOT / "bench" / "hello" / "bench.json"


def test_hello_benchmark_is_data():
    spec = json.loads(HELLO.read_text())
    assert spec["path"] == "/sentiment"
    assert len(spec["cases"]) == 6


def test_load_returns_a_callable():
    loss = nb.load("/sentiment")
    assert callable(loss)


def test_loss_ranks_stubs():
    loss = nb.load("/sentiment")
    good = nb.system("bench/hello/systems/good.py")
    dumb = nb.system("bench/hello/systems/dumb.py")
    assert loss(dumb) > loss(good)


def test_good_stub_scores_one():
    loss = nb.load("/sentiment")
    assert loss(nb.system("bench/hello/systems/good.py")) == 0.0


def test_dumb_stub_scores_half():
    loss = nb.load("/sentiment")
    loss(nb.system("bench/hello/systems/dumb.py"))
    # the 0.5 proves the scoring discriminates: per-case scores are real
    assert [c["score"] for c in loss.trace["cases"]] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert loss.trace["score"] == 0.5


def test_loss_is_pure_float():
    loss = nb.load("/sentiment")
    assert isinstance(loss(lambda t: "pos"), float)
    assert loss(lambda t: "pos") == 0.5  # always-pos: 3/6 wrong


def test_trace_is_the_receipt_of_last_eval():
    loss = nb.load("/sentiment")
    loss(nb.system("bench/hello/systems/dumb.py"))
    trace = loss.trace
    assert set(trace) >= {"path", "score", "cases"}
    assert trace["path"] == "/sentiment"
    assert len(trace["cases"]) == 6
    for c in trace["cases"]:
        assert set(c) >= {"in", "want", "got", "score"}


def test_loss_score_is_one_minus_trace_score():
    loss = nb.load("/sentiment")
    for name in ("good", "dumb"):
        got = loss(nb.system(f"bench/hello/systems/{name}.py"))
        assert got == pytest.approx(1.0 - loss.trace["score"])


def test_trace_is_json_artifact(tmp_path):
    loss = nb.load("/sentiment")
    loss(nb.system("bench/hello/systems/dumb.py"))
    art = tmp_path / "artifact.json"
    art.write_text(json.dumps(loss.trace))
    back = json.loads(art.read_text())
    assert back["score"] == 0.5
    assert len(back["cases"]) == 6


def test_missing_path_raises():
    with pytest.raises(LookupError):
        nb.load("/nope")