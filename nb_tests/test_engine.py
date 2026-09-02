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
    # cycle 13 probe: moving the receipt into the RETURN value (a tuple) broke
    # this test and 6 others. The vision exports the benchmark as "a new loss
    # function" for software-3.0 optimizers — an optimizer calls loss(system)
    # and must receive a scalar, nothing else. The receipt lives at loss.trace:
    # per-instance state, invisible to the calling convention. BARE_METAL.
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


def test_two_benchmarks_traces_are_independent():
    # cycle 12 escalation: could loss be a module-level function with shared
    # state instead of a per-load closure? Two benchmarks evaluated in
    # interleaved order must keep independent receipts — shared state
    # clobbers. The closure is the only honest home for trace.
    loss_a = nb.load("/sentiment")
    spec = json.loads(HELLO.read_text())
    tmp = tmp_bench({**spec, "path": "/sentiment_twin"})
    try:
        loss_b = nb.load("/sentiment_twin")
        loss_a(DUMB)
        loss_b(GOOD)
        assert loss_a.trace["score"] == 0.5
        assert loss_b.trace["score"] == 1.0
    finally:
        import shutil
        shutil.rmtree(tmp)


def test_scoring_is_data_and_guards_itself():
    # cycle 11 probe: deleting the scoring key + check stayed green — the
    # honesty guard was UNGUARDED. Scoring is a PILLAR (law #6: benchmark =
    # task + data + scoring); a benchmark that fails to name a scoring the
    # engine implements must fail loud, never silently exact-match.
    spec = json.loads(HELLO.read_text())
    assert spec["scoring"] == {"compare": "exact", "aggregate": "mean"}

    import copy
    bad = copy.deepcopy(spec)
    bad.pop("scoring")  # missing
    (ROOT / "bench" / "probe_missing" / "bench.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "bench" / "probe_missing" / "bench.json").write_text(json.dumps(bad))
    try:
        with pytest.raises(LookupError):
            nb.load("/sentiment")  # two matches now; the bad one must fail loud
    finally:
        import shutil
        shutil.rmtree(ROOT / "bench" / "probe_missing")

    bogus = tmp_bench({"path": "/probe_unknown", "scoring": {"compare": "fuzzy"}})
    try:
        with pytest.raises(LookupError):
            nb.load("/probe_unknown")  # unknown vocab: loud, not silent 0.0
    finally:
        shutil.rmtree(bogus)


def tmp_bench(spec):
    d = ROOT / "bench" / "probe_tmp"
    d.mkdir(parents=True, exist_ok=True)
    (d / "bench.json").write_text(json.dumps(spec))
    return d