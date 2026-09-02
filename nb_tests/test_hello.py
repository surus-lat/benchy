# tests: the hello acceptance bar (GOLEM.md) + loud-check laws. these define
# "broken". pytest only, engine imported via nb/ (stdlib engine).
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nb import as_loss, invoke, load, locate, run

HERE = Path(__file__).parent
ROOT = HERE.parent / "bench"
HELLO = ROOT / "hello" / "exam.json"


@pytest.fixture
def exam():
    return load(HELLO)


# --- locate / the ontology path ------------------------------------------------

def test_locate_by_ontology_path(exam):
    assert locate(ROOT, "/sentiment") == exam


def test_locate_missing_raises():
    with pytest.raises(FileNotFoundError):
        locate(ROOT, "/nope")


# --- the two stubs: scoring discriminates ---------------------------------------

def test_good_scores_one(exam):
    assert run(exam, exam["systems"]["good"])["score"] == 1.0


def test_dumb_scores_half(exam):
    assert run(exam, exam["systems"]["dumb"])["score"] == 0.5


def test_as_loss_ranks_stubs(exam):
    # cycle 5 deletion attempt: broke the vision contract (GOLEM law 6:
    # `loss = benchmark.as_loss()` is unbreakable; the loss-export IS the
    # headline feature). loss is not derivable-de-noise — it is the product.
    loss = as_loss(exam)
    assert loss(exam["systems"]["dumb"]) > loss(exam["systems"]["good"])
    assert loss(exam["systems"]["good"]) == 0.0


# --- the artifact: self-contained per-case evidence -----------------------------

def test_artifact_has_per_case_and_aggregate(exam):
    rep = run(exam, exam["systems"]["dumb"])
    assert len(rep["cases"]) == 6
    for c in rep["cases"]:
        assert {"id", "input", "want", "got", "score"} <= set(c)
    assert rep["score"] == sum(c["score"] for c in rep["cases"]) / 6
    # interprets alone: one dumb case shows want vs got without the exam file
    c = rep["cases"][3]
    assert c["want"] == "neg" and c["got"] == "pos" and c["score"] == 0.0


# --- the scoring lens: declared policy, loud -------------------------------------
# (cycle 6: grade fused into run — the declared policy is validated at load,
# the only entry to exam data, so grading is just its application)

def test_scoring_discriminates_via_run():
    # the mean of exact matches; dumb vs good proves the policy discriminates
    exam_dumb = run(load(HELLO), load(HELLO)["systems"]["dumb"])
    assert exam_dumb["cases"][0]["score"] == 1.0
    wrong = [c for c in exam_dumb["cases"] if c["want"] != c["got"]]
    assert all(c["score"] == 0.0 for c in wrong)


def test_run_unknown_policy_raises():
    exam_bad = json.loads(json.dumps(load(HELLO)))
    exam_bad["scoring"] = {"match": "fuzzy"}
    with pytest.raises(ValueError):
        run(exam_bad, exam_bad["systems"]["dumb"])


# --- the compiler lens: systems are specs in data --------------------------------

def test_invoke_const_and_keyword():
    assert invoke({"kind": "const", "value": "pos"}, "anything") == "pos"
    kw = {"kind": "keyword", "any": ["great"], "then": "pos", "else": "neg"}
    assert invoke(kw, "this works great") == "pos"
    assert invoke(kw, "never again") == "neg"


def test_invoke_unknown_kind_raises():
    with pytest.raises(ValueError):
        invoke({"kind": "neural-net"}, "x")


# --- loud checks: unknown keys raise, nothing silently ignored -------------------

def _mutate(exam, fn):
    data = json.loads(json.dumps(exam))
    fn(data)
    p = ROOT / "hello" / "_tmp_exam.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    try:
        load(p)
    finally:
        p.unlink()


def test_unknown_top_key_raises(exam):
    with pytest.raises(ValueError, match="unknown"):
        _mutate(exam, lambda d: d.update({"shuffle": True}))


def test_unknown_sample_key_raises(exam):
    with pytest.raises(ValueError, match="unknown"):
        _mutate(exam, lambda d: d["samples"][0].update({"weight": 2}))


def test_missing_sample_key_raises(exam):
    with pytest.raises(ValueError, match="missing"):
        _mutate(exam, lambda d: d["samples"][0].pop("expected"))


def test_expected_outside_declared_space_raises(exam):
    with pytest.raises(ValueError, match="not in task"):
        _mutate(exam, lambda d: d["samples"][0].update({"expected": "zzz"}))


def test_typo_expected_is_not_silently_a_new_class(exam):
    # the datacentric trap: inferred enum would absorb "negg" as a 3rd class.
    # the declared task must reject it loudly.
    with pytest.raises(ValueError, match="not in task"):
        _mutate(exam, lambda d: d["samples"][0].update({"expected": "negg"}))


def test_unknown_scoring_policy_raises(exam):
    with pytest.raises(ValueError, match="unknown"):
        _mutate(exam, lambda d: d.update({"scoring": {"match": "rouge"}}))


# --- offline end-to-end: the CLI, no pytest needed --------------------------------

def test_cli_end_to_end(tmp_path):
    out = tmp_path / "artifact.json"
    r = subprocess.run([sys.executable, "-m", "nb", str(ROOT), "/sentiment", str(out)],
                       capture_output=True, text=True, cwd=HERE.parent)
    assert r.returncode == 0, r.stderr
    art = json.loads(out.read_text(encoding="utf-8"))
    assert art["path"] == "/sentiment"
    assert set(art["systems"]) == {"good", "dumb"}
    assert art["systems"]["good"]["score"] == 1.0
    assert art["systems"]["dumb"]["score"] == 0.5
    assert "score=1.0" in r.stdout and "score=0.5" in r.stdout