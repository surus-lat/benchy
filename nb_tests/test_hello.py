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
    result = bench.run(HELLO, "good")
    assert result["score"] == 1.0


def test_hello_runs_offline_dumb_stub_scores_half():
    result = bench.run(HELLO, "dumb")
    assert result["score"] == 0.5


def test_as_loss_ranks_dumb_worse_than_good():
    good = bench.run(HELLO, "good")
    dumb = bench.run(HELLO, "dumb")
    assert bench.as_loss(dumb) > bench.as_loss(good)


def test_artifact_has_per_case_scores_and_aggregate():
    result = bench.run(HELLO, "dumb")
    assert set(result) == {"benchmark", "task", "cases", "score"}
    assert len(result["cases"]) == 6
    per = [c["score"] for c in result["cases"]]
    assert sum(per) == 3
    assert result["score"] == pytest.approx(0.5)


def test_scoring_format_is_loud_against_dead_keys(tmp_path):
    """scoring.json carrying keys the engine does not interpret must
    raise, not silently score — unread schema keys are noise (cycle 5)."""
    for f in ("task.json", "scoring.json", "cases.jsonl"):
        (tmp_path / f).write_text((HELLO / f).read_text(encoding="utf-8"),
                                  encoding="utf-8")
    (tmp_path / "systems").mkdir()
    for s in (HELLO / "systems").glob("*.json"):
        (tmp_path / "systems" / s.name).write_text(s.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
    scoring = json.loads((tmp_path / "scoring.json").read_text(encoding="utf-8"))
    scoring["aggregate"] = "mean"  # a key the engine no longer reads
    (tmp_path / "scoring.json").write_text(json.dumps(scoring), encoding="utf-8")
    with pytest.raises(ValueError):
        bench.run(tmp_path, "dumb")


def test_system_can_be_passed_as_data_dict_too():
    sysdata = bench.load(HELLO, "dumb")["system"]
    assert bench.run(HELLO, sysdata)["score"] == 0.5


def test_zero_user_python():
    """The benchmark directory must contain ONLY data files — no .py."""
    for p in HELLO.rglob("*"):
        assert p.suffix != ".py", f"benchmark contains python: {p}"


def test_dumb_case_scores_prove_discrimination():
    """Dumb gets the 3 pos cases right, the 3 neg cases wrong."""
    result = bench.run(HELLO, "dumb")
    wrong = [c for c in result["cases"] if c["score"] == 0]
    assert len(wrong) == 3
    assert all(c["expected"] == "neg" for c in wrong)


EXTRACT = ROOT / "bench" / "extract"


def test_weighted_scoring_is_pure_data_and_ranks_critical_above_nice():
    """C6 probe: hierarchy-of-importance in pure data. 'critical' hits only
    the 5-weight field; 'nice' hits two 1-weight fields. Weights must rank
    critical ABOVE nice — 1 critical field > 2 nice-to-have fields."""
    critical = bench.run(EXTRACT, "critical")
    nice = bench.run(EXTRACT, "nice")
    assert critical["score"] == pytest.approx(5 / 7)
    assert nice["score"] == pytest.approx(2 / 7)
    assert critical["score"] > nice["score"]
    assert bench.as_loss(critical) < bench.as_loss(nice)


def test_weights_must_name_every_expected_field(tmp_path):
    """Loud check: weights naming a subset of fields must raise."""
    for f in ("task.json", "scoring.json", "cases.jsonl"):
        (tmp_path / f).write_text((EXTRACT / f).read_text(encoding="utf-8"),
                                  encoding="utf-8")
    (tmp_path / "systems").mkdir()
    for s in (EXTRACT / "systems").glob("*.json"):
        (tmp_path / "systems" / s.name).write_text(s.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
    scoring = json.loads((tmp_path / "scoring.json").read_text(encoding="utf-8"))
    scoring["weights"] = {"total": 5}  # subset — loud check must bite
    (tmp_path / "scoring.json").write_text(json.dumps(scoring), encoding="utf-8")
    with pytest.raises(ValueError):
        bench.run(tmp_path, "critical")


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