"""The hello acceptance bar (search/GOLEM.md): run offline, end to end.

These tests are the definition of "broken" for the s06 engine.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELLO = ROOT / "bench" / "hello"
sys.path.insert(0, str(ROOT))


def _exam():
    from nb.exam import load
    return load(HELLO)


def _stub(name):
    sys.path.insert(0, str(HELLO))
    import stubs
    return getattr(stubs, name)


def test_good_stub_scores_one():
    assert _exam().run(_stub("good"))["score"] == 1.0


def test_dumb_stub_scores_half():
    assert _exam().run(_stub("dumb"))["score"] == 0.5


def test_loss_ranks_dumb_above_good():
    loss = _exam().as_loss()
    assert loss(_stub("dumb")) > loss(_stub("good"))


def test_artifact_is_graded_json():
    art = _exam().run(_stub("dumb"))
    assert art["benchmark"] == "/sentiment"
    assert len(art["cases"]) == 6
    for page in art["cases"]:
        assert {"input", "expected", "prediction", "score"} <= set(page)
    assert art["score"] == sum(p["score"] for p in art["cases"]) / 6


def test_locatable_by_ontology_path():
    from nb.exam import locate
    exam = locate(ROOT / "bench", "/sentiment")
    assert exam.path == "/sentiment" and len(exam.cases) == 6


def test_benchmark_is_pure_data():
    data = json.loads((HELLO / "benchmark.json").read_text())
    assert data["task"]["output"]["enum"] == ["pos", "neg"]
    assert data["scoring"]["kind"] == "exact_match"
    assert [c["expected"] for c in data["cases"]] == ["pos"] * 3 + ["neg"] * 3


def test_systems_conform_structurally():
    from nb.core import System
    assert isinstance(_stub("good"), System)
    assert isinstance(_stub("dumb"), System)


def test_any_invoked_program_takes_the_exam():
    class AlwaysNeg:
        def invoke(self, x):
            return "neg"

    assert _exam().run(AlwaysNeg())["score"] == 0.5


def test_scorer_swap_needs_zero_engine_lines():
    from nb.exam import Exam

    def anti(case, prediction):
        return 0.0 if prediction == "pos" else 1.0

    exam = _exam()
    swapped = Exam(exam.cases, anti)
    assert swapped.run(_stub("good"))["score"] == 0.5
    assert swapped.run(_stub("dumb"))["score"] == 0.0


def test_cli_runs_offline_end_to_end(tmp_path):
    hello = tmp_path / "hello"
    shutil.copytree(HELLO, hello)
    r = subprocess.run([sys.executable, "-m", "nb", str(hello), "good"],
                       capture_output=True, text=True, cwd=ROOT, check=False)
    assert r.returncode == 0, r.stderr
    art = json.loads((hello / "artifact_good.json").read_text())
    assert art["score"] == 1.0 and len(art["cases"]) == 6