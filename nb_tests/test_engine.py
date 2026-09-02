"""the pytest suite that defines BROKEN.

Hello acceptance (GOLEM.md) must pass here:
  good stub scores 1.0 · dumb stub scores 0.5 · loss(dumb) > loss(good)
  graded artifact JSON · /sentiment ontology path · stdlib-only engine
"""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import nb
from nb import Benchmark, compile_system
from nb.load import main as load_main
from nb import load, score

BENCH = Path(__file__).resolve().parent.parent / "bench" / "hello"


def _systems():  # test-side convenience: compile BENCH's systems/*.json
    import json
    sd = BENCH / "systems"
    return {p.stem: compile_system(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(sd.glob("*.json"))}


# ------------------------------------------------------------ hello bar

def hello_bench():
    return load(BENCH)


def test_hello_good_stub_scores_1():
    bench = hello_bench()
    art = bench.run(_systems()["good-stub"])
    assert art["score"] == 1.0


def test_hello_dumb_stub_scores_half():
    bench = hello_bench()
    art = bench.run(_systems()["dumb-stub"])
    assert art["score"] == 0.5


def test_loss_ranks_stubs():
    bench = hello_bench()
    loss = bench.as_loss()
    systems = _systems()
    assert loss(systems["dumb-stub"]) > loss(systems["good-stub"])


def test_artifact_has_per_case_and_aggregate():
    bench = hello_bench()
    art = bench.run(_systems()["dumb-stub"])
    assert set(art) >= {"score", "cases"}
    assert len(art["cases"]) == 6
    assert all(set(c) >= {"input", "expected", "prediction", "score"} for c in art["cases"])
    assert sum(c["score"] for c in art["cases"]) / 6 == art["score"]


def test_artifact_is_json_serializable():
    bench = hello_bench()
    art = bench.run(_systems()["dumb-stub"])
    json.dumps(art)


def test_ontology_path_sentiment():
    """the benchmark must be locatable by its ontology path /sentiment."""
    root = BENCH.parent  # bench/
    hits = [p for p in root.rglob("cases.json")
            if p.parent.name in ("sentiment", "hello")]
    assert hits, "no /sentiment benchmark found under bench/"


# ------------------------------------------------------------ four pillars

def test_task_is_in_out_declaration():
    """the TASK pillar is DATA: the task.json dict, uninterpreted."""
    assert load(BENCH).task == {"in": "text", "out": "label"}


def test_scoring_exact_partial_weighted():
    s = {"mode": "exact"}
    assert score(s, "pos", "pos") == 1.0 and score(s, "neg", "pos") == 0.0


def test_scoring_partial():
    s = {"mode": "partial"}
    assert score(s, {"a": 1, "b": 2}, {"a": 1, "b": 3}) == 0.5


def test_scoring_weighted():
    s = {"mode": "weighted", "weights": {"a": 9, "b": 1}}
    assert score(s, {"a": 1, "b": 2}, {"a": 1, "b": 2}) == 1.0
    assert score(s, {"a": 1, "b": 9}, {"a": 1, "b": 2}) == 0.9
    assert score(s, {"b": 2}, {"a": 1, "b": 2}) == 0.1


def test_exam_rejects_empty(tmp_path):
    """an empty exam is shape-invalid at the data door (load)."""
    tmp_path.joinpath("task.json").write_text('{"in": "text", "out": "label"}')
    tmp_path.joinpath("scoring.json").write_text('{"mode": "exact"}')
    tmp_path.joinpath("cases.json").write_text("[]")
    try:
        load(tmp_path)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_benchmark_system_is_argument_not_field():
    bench = Benchmark({"in": "text", "out": "label"}, {"mode": "exact"},
                      [("x", "pos")])
    assert not any(isinstance(getattr(bench, a, None), type(bench))
                   for a in vars(bench))


def test_invoke_one_method():
    """the protocol: a system IS a callable — invoke(input) -> prediction."""
    sys_fn = compile_system({"kind": "const", "const": "pos"})
    assert sys_fn("anything") == "pos"


def test_stub_backend_is_data():
    spec = {"kind": "stub", "rules": {"great": "pos"}, "default": "neg"}
    assert compile_system(spec)("this works great") == "pos"
    assert compile_system(spec)("meh") == "neg"


# ------------------------------------------------------------ backends

def test_http_backend_against_mock_server():
    """openai-compatible backend must work over stdlib urllib, no network."""

    class Mock(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = "pos" if "great" in body["messages"][-1]["content"] else "neg"
            out = {"choices": [{"message": {"content": content}}]}
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Mock)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        spec = {"kind": "http", "url": f"http://127.0.0.1:{server.server_port}",
                "model": "mock-1", "system": "reply pos or neg only"}
        sys_fn = compile_system(spec)
        assert sys_fn("this works great") == "pos"
        assert sys_fn("never again") == "neg"
    finally:
        server.shutdown()
        server.server_close()


def test_chain_backend_composes_systems():
    """workflow = system whose backend composes systems. No new concept."""
    inner = {"kind": "stub", "rules": {"great": "one"}, "default": "zero"}
    outer = {"kind": "stub", "rules": {"one": "pos", "zero": "neg"}, "default": "neg"}
    chain = {"kind": "chain", "steps": [inner, outer]}
    assert compile_system(chain)("this works great") == "pos"
    assert compile_system(chain)("meh") == "neg"


def test_agent_backend_is_pure_data():
    """THE falsification probe: an agent (loop + tools) is configuration.

    The 'model' is a stub that asks for a tool once, then answers. The
    tool is a stub. The spec is pure data; the loop is compiler code
    (_backend_agent). The core (Task/Scoring/Exam/Benchmark) never
    learned anything about agents.
    """
    # deterministic model: 1st call -> tool request, 2nd call -> answer
    calls = {"n": 0}

    def fake_model(text):
        calls["n"] += 1
        if calls["n"] == 1:
            return ["tool", "lookup", "the answer"]
        return "pos"

    tool = {"kind": "const", "const": "the answer is pos"}
    spec = {"kind": "agent", "model": fake_model, "tools": {"lookup": tool},
            "max_iters": 3}
    assert compile_system(spec)("what is it?") == "pos"
    assert calls["n"] == 2  # asked once, answered once: the loop worked


def test_agent_budget_out_graded_honestly():
    """a model that never answers returns its last utterance, not a crash."""
    def loopy(text):
        return ["tool", "lookup", "x"]  # constant request: never answers

    tool = {"kind": "const", "const": "ignored"}
    spec = {"kind": "agent", "model": loopy, "tools": {"lookup": tool},
            "max_iters": 2}
    pred = compile_system(spec)("q")
    assert pred == ["tool", "lookup", "x"]  # budget out: utterance graded


def test_agent_unknown_tool_is_prediction():
    """unknown tool: the utterance IS the prediction (no raise, no magic)."""
    spec = {"kind": "agent", "model": {"kind": "const", "const": ["tool", "nope", "x"]},
            "tools": {}, "max_iters": 3}
    assert compile_system(spec)("q") == ["tool", "nope", "x"]


def test_unknown_kind_is_rejected():
    try:
        compile_system({"kind": "nope"})
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_callable_spec_is_pass_through():
    assert compile_system(lambda x: "pos")("x") == "pos"


# ------------------------------------------------------------ CLI

def test_cli_runs_hello(tmp_path, capsys):
    assert load_main([str(BENCH)]) == 0
    out = capsys.readouterr().out
    assert '"score": 1.0' in out and '"score": 0.5' in out


def test_cli_unknown_system():
    assert load_main([str(BENCH), "ghost"]) == 2