"""nb tests — hello acceptance bar + the runner contract: concurrency, retries,
kill/resume, zero lost work, artifact interprets alone, exit codes."""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nb import Exam, locate

ROOT = Path(__file__).resolve().parents[1]


def hello():
    return Exam(locate("/sentiment"))


def spec(name):
    return json.loads((ROOT / "bench/sentiment/systems" / f"{name}.json").read_text())


# ---- the hello acceptance bar (GOLEM.md) ----

def test_locate_finds_the_exam_by_ontology_path():
    d = locate("/sentiment")
    assert d.name == "sentiment"  # the directory IS the ontology address
    assert (d / "exam.json").is_file()


def test_good_scores_1_and_dumb_scores_0_5():
    e = hello()
    assert e.run(spec("good"))["score"] == 1.0
    assert e.run(spec("dumb"))["score"] == 0.5


def test_dumb_beats_random_proves_scoring_discriminates():
    e = hello()
    art = e.run(spec("dumb"))
    per_case = [c["score"] for c in art["cases"]]
    assert sorted(per_case) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_artifact_interprets_alone():
    art = hello().run(spec("good"))
    for c in art["cases"]:
        assert set(c) >= {"id", "input", "want", "got", "score", "status"}
    assert art["score"] == 1.0
    assert art["total"] == 6
    assert art["errors"] == 0


def test_as_loss_ranks_the_stubs():
    e = hello()
    assert e.as_loss(spec("dumb")) > e.as_loss(spec("good"))
    assert e.as_loss(spec("good")) == 0.0


# ---- the runner contract: concurrency, retries, resume ----

def big_exam(dir, n, sleep=0.0, tag=""):
    """Generate an n-case exam fixture (test data, not engine input)."""
    cases = [{"id": f"c{i:04d}", "input": f"text {i} {i%7}{tag}", "want": "pos" if i % 2 else "neg"}
             for i in range(n)]
    p = dir / "exam.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"path": "/big", "out": ["pos", "neg"], "cases": cases}))
    return Exam(p.parent)


def test_1000_cases_concurrently_against_flaky_stub(tmp_path):
    e = big_exam(tmp_path, 1000)
    sleep = 0.01  # per attempt; stands in for real per-case exam-taker latency
    sys_spec = {"kind": "flaky", "script": "FP", "sleep": sleep,
                "of": {"kind": "always", "value": "pos"}}
    t0 = time.monotonic()
    art = e.run(sys_spec, out=tmp_path / "a.json", workers=16)
    dt = time.monotonic() - t0
    assert len(art["cases"]) == 1000
    assert art["errors"] == 0
    # every case failed its first attempt -> retries actually fired
    assert all(c["tries"] == 2 for c in art["cases"])
    assert art["score"] == 0.5  # always-pos: right on the pos-wanting half only
    # concurrency is load-bearing: the serial floor is n*tries*sleep = 20s of
    # pure sleep (sleep never undersleeps), so a serial runner can NOT pass.
    serial_floor = 2 * 1000 * sleep
    assert dt < serial_floor * 0.75, f"not concurrent enough: {dt:.1f}s"


def test_flaky_exhausting_retries_is_loud_not_silent(tmp_path):
    e = big_exam(tmp_path, 4)
    art = e.run({"kind": "flaky", "script": "F",
                 "of": {"kind": "always", "value": "pos"}}, tries=3)
    assert art["errors"] == 4
    assert all(c["status"] == "error" and "RuntimeError" in c["error"] for c in art["cases"])


def test_resume_after_real_kill_loses_zero_work(tmp_path):
    e = big_exam(tmp_path, 1000)
    sys_spec = {"kind": "flaky", "script": "FP", "sleep": 0.01,
                "of": {"kind": "always", "value": "pos"}}
    sp = tmp_path / "sys.json"
    sp.write_text(json.dumps(sys_spec))
    out = tmp_path / "killed.json"
    proc = subprocess.Popen(
        [sys.executable, "-m", "nb", str(tmp_path), str(sp), "-o", str(out),
         "--workers", "8", "--tries", "3"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # wait until some but not all cases are durable, then SIGKILL mid-run
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if out.exists():
            n = len(json.loads(out.read_text())["cases"])
            if 10 < n < 900:
                break
        time.sleep(0.02)
    else:
        proc.kill()
        pytest.fail("artifact never progressed mid-run; kill window not found")
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    killed = json.loads(out.read_text())  # atomic write: never torn, even mid-rewrite
    assert 10 < len(killed["cases"]) < 900
    kept = {c["id"] for c in killed["cases"]}
    assert len(kept) == len(killed["cases"])  # no duplicate ids
    # resume: same artifact path, same exam+system -> only the missing re-run
    art = e.run(sys_spec, out=out, workers=8)
    assert len(art["cases"]) == 1000
    assert art["errors"] == 0
    assert {c["id"] for c in art["cases"]} == {f"c{i:04d}" for i in range(1000)}
    ids = [c["id"] for c in art["cases"]]
    assert len(ids) == len(set(ids))
    for c in art["cases"]:
        if c["id"] in kept:
            assert c["status"] == "ok"  # kept, not re-run


def test_resume_refuses_mismatched_exam_or_system(tmp_path):
    e = big_exam(tmp_path, 4)
    out = tmp_path / "a.json"
    e.run({"kind": "always", "value": "pos"}, out=out)
    with pytest.raises(ValueError, match="different exam"):
        big_exam(tmp_path / "other", 4, tag="X").run({"kind": "always", "value": "pos"}, out=out)
    with pytest.raises(ValueError, match="different exam"):
        e.run({"kind": "always", "value": "neg"}, out=out)
    # an edited exam (same ids, changed want) must not silently reuse stale evidence
    p = tmp_path / "exam.json"
    spec = json.loads(p.read_text())
    spec["cases"][0]["want"] = "pos"  # c0000 originally wants neg
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="different exam"):
        Exam(tmp_path).run({"kind": "always", "value": "pos"}, out=out)
    # an edited scoring block (weights change) makes every kept score stale too
    spec = json.loads(p.read_text())
    spec["scoring"] = {"weights": {"critical": 9.0}}
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="different exam"):
        Exam(tmp_path).run({"kind": "always", "value": "pos"}, out=out)


def test_kill_during_heavy_write_never_leaves_torn_artifact(tmp_path):
    # c4 judge: an external reader (another process, a dashboard, the agent
    # reading the artifact) may observe the artifact at ANY moment mid-run;
    # it must ALWAYS be complete JSON. a plain write_text truncates on every
    # O(n^2) rewrite -> any reader during the ms-wide refill windows sees
    # torn/empty bytes -> a kill there = lost work (forbidden by the bar).
    # tmp+rename only ever exposes complete states (rename is atomic).
    n = 40
    cases = [{"id": f"h{i:03d}", "input": "x" * 100000 + str(i),
              "want": "pos" if i % 2 else "neg"} for i in range(n)]
    (tmp_path / "exam.json").write_text(json.dumps(
        {"path": "/heavy", "out": ["pos", "neg"], "cases": cases}))
    sp = tmp_path / "sys.json"
    sp.write_text(json.dumps({"kind": "always", "value": "pos"}))
    out = tmp_path / "heavy.json"
    proc = subprocess.Popen(
        [sys.executable, "-m", "nb", str(tmp_path), str(sp), "-o", str(out)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    torn = polls = done_n = 0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if out.exists():
            # a complete artifact is never size-0 and always ends with '}'
            try:
                with out.open("rb") as f:
                    if f.seek(0, 2) == 0:
                        torn += 1
                    else:
                        f.seek(-1, 2)
                        if f.read(1) != b"}":
                            torn += 1
                    polls += 1
                    if polls % 25 == 0:  # progress check every 25 cheap polls
                        f.seek(0)
                        done_n = f.read().count(b'"id": "h')
                        if done_n >= n - 15:  # late but not finished: kill window
                            break
            except OSError:
                torn += 1  # reader saw the file in a transitional state
        # busy-poll: no sleep, maximize the chance of catching a torn write
    else:
        proc.kill()
        pytest.fail("heavy run never progressed; kill window not found")
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    assert polls > 200, f"poller barely observed the run ({polls} polls)"
    assert torn == 0, f"artifact was torn/empty for a reader {torn}/{polls} polls"
    killed = json.loads(out.read_text())  # atomic: complete even mid-rewrite
    assert 0 < len(killed["cases"]) < n
    # resume from the surviving complete artifact: zero lost work
    art = Exam(tmp_path).run(json.loads(sp.read_text()), out=out)
    assert len(art["cases"]) == n
    assert art["errors"] == 0


def test_weighted_scoring_is_data_not_code(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "exam.json").write_text(json.dumps({
        "path": "/w", "out": ["x", "y"],
        "scoring": {"weights": {"critical": 3.0, "nice": 1.0}},
        "cases": [{"id": "w1", "input": "in",
                   "want": {"critical": "x", "nice": "y"}}]}))
    e = Exam(d)
    art = e.run({"kind": "always", "value": {"critical": "x", "nice": "x"}})
    assert art["cases"][0]["score"] == 0.75  # 3 of 4 weight on critical, right
    assert art["score"] == 0.75


def test_loud_reject_of_want_outside_declared_out(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "exam.json").write_text(json.dumps({"path": "/bad", "out": ["pos", "neg"],
                              "cases": [{"id": "b", "input": "x", "want": "meh"}]}))
    with pytest.raises(ValueError, match="outside declared out"):
        Exam(d)


def test_cli_runs_and_exit_codes(tmp_path):
    out = tmp_path / "cli.json"
    r = subprocess.run([sys.executable, "-m", "nb", "/sentiment", "good", "-o", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "score=1.000" in r.stdout
    art = json.loads(out.read_text())
    assert art["score"] == 1.0 and art["errors"] == 0
    sp = tmp_path / "dead.json"
    sp.write_text(json.dumps({"kind": "flaky", "script": "F",
                              "of": {"kind": "always", "value": "pos"}}))
    r2 = subprocess.run([sys.executable, "-m", "nb", "/sentiment", str(sp), "-o", str(tmp_path / "e.json")],
                         cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode == 1  # errors -> non-zero, the artifact contract


def test_unknown_system_kind_and_unknown_path_are_loud(tmp_path):
    with pytest.raises(ValueError, match="unknown system kind"):
        hello().run({"kind": "quantum"})
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        locate("/nonexistent")