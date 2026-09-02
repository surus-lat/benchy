"""the hello acceptance bar — defines "broken" for this search.

Every invariant of search/GOLEM.md's hello section, as executable law.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nb import as_loss, compile, grade, run

BENCH = Path(__file__).resolve().parents[1] / "bench" / "hello"
GOOD = {"name": "good", "kind": "keyword", "pos": ["great", "excelente", "loved"], "default": "neg"}
DUMB = {"name": "dumb", "kind": "keyword", "pos": [], "default": "pos"}


def bench() -> dict:
    # the file layer, inlined: path -> file -> value (engine stays pure)
    return json.loads((BENCH / "sentiment.json").read_text(encoding="utf-8"))


def system_specs() -> dict:
    raw = json.loads((BENCH / "systems.json").read_text(encoding="utf-8"))
    return {s["name"]: s for s in raw}


class TestPillarData:
    def test_benchmark_locatable_by_ontology_path(self):
        b = bench()
        assert b["path"] == "/sentiment"

    def test_six_cases_with_expected_labels(self):
        b = bench()
        assert len(b["cases"]) == 6
        assert all(c["expected"] in ("pos", "neg") for c in b["cases"])

    def test_output_declared_as_choices(self):
        b = bench()
        assert set(b["task"]["output"]["choices"]) == {"pos", "neg"}


class TestPillarSystem:
    def test_compiler_makes_system_callable(self):
        invoke = compile(GOOD)
        assert callable(invoke)
        assert invoke("this works great") == "pos"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            compile({"kind": "nope"})


class TestPillarScoring:
    def test_good_stub_scores_perfect(self):
        assert run(bench(), system_specs()["good"])["score"] == 1.0

    def test_grade_seam_accepts_any_callable(self):
        # the escape hatch: any invoke(text)->pred, not just engine-compiled
        # specs (real APIs, workflows, cached runs) must be gradable
        from nb import grade
        art = grade(bench(), lambda t: "pos")
        assert art["score"] == 0.5

    def test_dumb_stub_scores_half(self):
        assert run(bench(), system_specs()["dumb"])["score"] == 0.5

    def test_score_is_mean_of_per_case(self):
        art = run(bench(), system_specs()["dumb"])
        per_case = [r["score"] for r in art["cases"]]
        assert art["score"] == sum(per_case) / len(per_case)

    def test_artifact_has_per_case_and_aggregate(self):
        art = run(bench(), system_specs()["good"])
        assert art["benchmark"] == "/sentiment"
        assert len(art["cases"]) == 6
        assert all({"case", "input", "expected", "prediction", "score"} == set(r) for r in art["cases"])

    def test_unknown_rule_rejected(self):
        b = bench()
        b["scoring"]["rule"] = "vibes"
        with pytest.raises(ValueError):
            run(b, GOOD)


class TestVisionInvariants:
    def test_as_loss_ranks_stubs(self):
        loss = as_loss(bench())
        assert loss(DUMB) > loss(GOOD)

    def test_system_is_the_argument_not_a_field(self):
        # run(benchmark, system): neither benchmark nor artifact carries a system
        b = bench()
        assert "system" not in b
        assert "system" not in run(b, GOOD)


class TestCLI:
    def test_cli_runs_offline_and_writes_artifact(self, tmp_path):
        out = tmp_path / "artifact.json"
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(BENCH), "/sentiment", str(out), "dumb"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode == 0, proc.stderr
        art = json.loads(out.read_text(encoding="utf-8"))
        assert art["score"] == 0.5

    def test_cli_defaults_to_first_system(self, tmp_path):
        out = tmp_path / "artifact.json"
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(BENCH), "/sentiment", str(out)],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(out.read_text(encoding="utf-8"))["score"] == 1.0

    def test_cli_bad_usage_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(BENCH)],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode != 0