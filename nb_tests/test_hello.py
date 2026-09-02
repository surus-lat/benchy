"""the hello acceptance bar — defines "broken" for this search.

Every invariant of search/GOLEM.md's hello section, as executable law.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nb.engine import as_loss, compile, grade, run

BENCH = Path(__file__).resolve().parents[1] / "bench" / "hello" / "bundle"
GOOD = {"name": "good", "kind": "keyword", "pos": ["great", "excelente", "loved"], "default": "neg"}
DUMB = {"name": "dumb", "kind": "keyword", "pos": [], "default": "pos"}


def bench() -> dict:
    # the file layer, inlined: the bundle directory -> benchmark value
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
        from nb.engine import grade
        art = grade(bench(), lambda t: "pos")
        assert art["score"] == 0.5

    def test_dumb_stub_scores_half(self):
        assert run(bench(), system_specs()["dumb"])["score"] == 0.5

    def test_dumb_stub_is_always_pos(self):
        # c14: GOLEM.md's acceptance bar says the dumb stub is "always pos",
        # but only the 0.5 SCORE was pinned — the behavior was claimed but
        # untested. pinned before the default-deletion probe: the dumb stub
        # predicts pos on every case (the constant-pos function).
        art = run(bench(), system_specs()["dumb"])
        assert all(r["prediction"] == "pos" for r in art["cases"])

    def test_score_is_mean_of_per_case(self):
        art = run(bench(), system_specs()["dumb"])
        per_case = [r["score"] for r in art["cases"]]
        assert art["score"] == sum(per_case) / len(per_case)

    def test_artifact_has_per_case_and_aggregate(self):
        art = run(bench(), system_specs()["good"])
        assert art["benchmark"] == "/sentiment"
        assert len(art["cases"]) == 6
        # c9: position in the list IS the case id — no explicit index field.
        # c16 grew the row by `error` (donated failure contract — not
        # derivable: it carries the failure type + message).
        assert all({"input", "expected", "prediction", "score", "error"} == set(r) for r in art["cases"])

    def test_unknown_rule_rejected(self):
        b = bench()
        b["scoring"]["rule"] = "vibes"
        with pytest.raises(ValueError):
            run(b, GOOD)

    def test_unknown_aggregate_rejected(self):
        # c11: the aggregate refusal was claimed but untested (3 trees old
        # lesson). pinned first, then deletion-probed.
        b = bench()
        b["scoring"]["aggregate"] = "median"
        with pytest.raises(ValueError):
            run(b, GOOD)

    def test_exam_key_outside_declared_output_rejected(self):
        # c8: the task declaration is load-bearing, not decoration
        b = bench()
        b["cases"][0]["expected"] = "meh"
        with pytest.raises(ValueError):
            run(b, GOOD)


def flaky():
    """a real (cloud) system is not data — it fails after the first call."""
    n = [0]

    def invoke(text: str) -> str:
        n[0] += 1
        if n[0] > 1:
            raise ConnectionError("transient")
        return "pos"

    return invoke


class TestOldBenchyDonations:
    # c16, donated from the old benchy run-loop contract (.staging/benchy/
    # benchmark.py): one bad sample never aborts the run — a system failure
    # is evidence, not a crash. Deliberate divergence: the old system
    # EXCLUDED errored samples from the aggregate; here a failed case scores
    # 0 and is INCLUDED — one scalar, the optimizer sees reliability too.

    def test_flaky_system_does_not_abort_exam(self):
        art = grade(bench(), flaky())
        assert len(art["cases"]) == 6  # every case was taken and graded

    def test_failure_is_visible_evidence_not_silence(self):
        art = grade(bench(), flaky())
        assert art["cases"][0]["error"] is None
        assert "ConnectionError" in art["cases"][1]["error"]

    def test_failed_case_scores_zero_not_excluded(self):
        art = grade(bench(), flaky())
        assert art["score"] == 1 / 6  # /6, not /5 — failures count against
        assert 1.0 - art["score"] > as_loss(bench())(GOOD)

    def test_error_key_present_on_every_row(self):
        art = run(bench(), GOOD)
        assert all("error" in r and r["error"] is None for r in art["cases"])

    def test_empty_exam_refused_not_zero_division(self):
        # c17, donated from the old status vocabulary (no_samples): an empty
        # exam is a broken exam — refusal, never a mid-run ZeroDivisionError.
        b = bench()
        b["cases"] = []
        with pytest.raises(ValueError, match="empty exam"):
            grade(b, lambda t: "pos")

    def test_cli_refuses_mismatched_ontology_path(self, tmp_path):
        # c18, donated from the old spine (OntologyPath: registry key ==
        # on-disk layout; old load_benchmark resolved BY ontology): a file
        # whose declared path != requested path is a broken exam install —
        # running it would silently write artifacts under the wrong identity.
        broken = tmp_path / "broken"
        broken.mkdir()
        shutil.copy(BENCH / "sentiment.json", broken / "other.json")
        shutil.copy(BENCH / "systems.json", broken / "systems.json")
        out = tmp_path / "artifact.json"
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(broken), "/other", str(out), "dumb"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
        assert "declares" in proc.stderr
        assert not out.exists()  # no artifact under a wrong identity


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

    def test_cli_ack_line_carries_exam_system_score(self, tmp_path):
        # c12: the stdout ack was claimed but untested (CLI tests checked
        # exit code + artifact file only). pinned before the deletion probe.
        out = tmp_path / "artifact.json"
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(BENCH), "/sentiment", str(out), "dumb"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode == 0, proc.stderr
        assert "score=0.5" in proc.stdout
        assert "system=dumb" in proc.stdout
        assert "/sentiment" in proc.stdout

    def test_cli_defaults_to_first_system(self, tmp_path):
        # c10: the silent first-system default is GONE — a CLI call without
        # a system name must refuse (exit 2), never silently pick systems[0]
        out = tmp_path / "artifact.json"
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(BENCH), "/sentiment", str(out)],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
        assert not out.exists()

    def test_cli_bad_usage_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "nb", str(BENCH)],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        assert proc.returncode != 0