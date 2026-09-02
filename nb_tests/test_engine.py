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


def loss_of(b, system):
    return b.as_loss()(system)


def test_hello_benchmark_is_data():
    spec = json.loads(HELLO.read_text())
    assert spec["path"] == "/sentiment"
    assert len(spec["cases"]) == 6
    assert spec["task"]["out"] == ["pos", "neg"]


def test_loss_ranks_stubs():
    b = nb.load("/sentiment")
    good = nb.system("bench/hello/systems/good.py")
    dumb = nb.system("bench/hello/systems/dumb.py")
    assert loss_of(b, dumb) > loss_of(b, good)


def test_good_stub_scores_one():
    r = nb.load("/sentiment").run(nb.system("bench/hello/systems/good.py"))
    assert r["score"] == 1.0


def test_dumb_stub_scores_half():
    r = nb.load("/sentiment").run(nb.system("bench/hello/systems/dumb.py"))
    assert r["score"] == 0.5
    # the 0.5 proves the scoring discriminates: per-case scores are real
    assert [c["score"] for c in r["cases"]] == [1.0, 1.0,  1.0, 0.0, 0.0, 0.0]


def test_receipt_is_the_loss_evidence():
    b = nb.load("/sentiment")
    for name in ("good", "dumb"):
        sys_prog = nb.system(f"bench/hello/systems/{name}.py")
        receipt = b.run(sys_prog)
        assert receipt["score"] == 1.0 - loss_of(b, sys_prog)


def test_receipt_has_per_case_scores_and_aggregate():
    r = nb.load("/sentiment").run(nb.system("bench/hello/systems/dumb.py"))
    assert len(r["cases"]) == 6
    for c in r["cases"]:
        assert set(c) >= {"in", "want", "got", "score"}
    assert r["aggregate"] == "mean"


def test_artifact_json_roundtrip(tmp_path):
    r = nb.load("/sentiment").run(nb.system("bench/hello/systems/dumb.py"))
    art = tmp_path / "artifact.json"
    art.write_text(json.dumps(r))
    back = json.loads(art.read_text())
    assert back["score"] == 0.5
    assert len(back["cases"]) == 6


def test_as_loss_is_pure_float():
    b = nb.load("/sentiment")
    L = b.as_loss()
    assert isinstance(L(nb.system("bench/hello/systems/good.py")), float)
    assert isinstance(L(nb.system("bench/hello/systems/dumb.py")), float)


def test_load_by_ontology_path():
    b = nb.load("/sentiment")
    assert b.spec["path"] == "/sentiment"


def test_system_is_just_a_callable():
    b = nb.load("/sentiment")
    assert b.as_loss()(lambda text: "neg") == 1.0  # always-neg: 3/6 wrong
    assert b.as_loss()(lambda text: "pos") == 0.5