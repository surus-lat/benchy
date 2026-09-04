"""The acceptance bar of the bare-metal search, replayed over ONE yaml.

Covers: the 4 pillars in a single file, offline stubs, weighted scoring
as pure data, loss identity, self-contained artifact, the cloud-first
http backend (mocked openai-compatible server, stdlib only), loud
errors, retries.
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nb

BENCH = Path(__file__).parent / "bench"


# ------------------------------------------------------------- offline core

def test_sentiment_good_stub_scores_one():
    r = nb.run(BENCH / "sentiment.yaml")
    assert r["score"] == 1.0
    assert r["loss"] == 0.0


def test_sentiment_constant_pos_scores_half():
    dumb = {"kind": "stub", "rules": {}, "default": "pos"}
    r = nb.run(BENCH / "sentiment.yaml", dumb)
    assert r["score"] == 0.5


def test_loss_ranks_stubs():
    dumb = {"kind": "stub", "rules": {}, "default": "pos"}
    assert nb.as_loss(BENCH / "sentiment.yaml") < \
        nb.as_loss(BENCH / "sentiment.yaml", dumb)


def test_weighted_scoring_is_pure_data():
    """s05's C6 verdict replayed: critical field (5/7) beats two minors (2/7)."""
    critical = {"kind": "regex", "fields": {"total": r"\$(\d+)"}, "miss": ""}
    minors = {"kind": "regex",
              "fields": {"vendor": r"from (\w+)", "date": r"dated ([\w-]+)"},
              "miss": ""}
    rc = nb.run(BENCH / "extract.yaml", critical)
    rm = nb.run(BENCH / "extract.yaml", minors)
    assert rc["score"] > rm["score"]
    assert rc["score"] == 5 / 7 and rm["score"] == 2 / 7


def test_artifact_is_self_contained():
    """s01's BARE_METAL: the receipt is interpretable ALONE."""
    r = nb.run(BENCH / "sentiment.yaml")
    for c in r["cases"]:
        assert set(c) >= {"input", "expected", "predicted", "score"}
    assert r["system"] == "default"


def test_named_systems_map():
    bench = nb.load(BENCH / "sentiment.yaml")
    bench["systems"] = {"dumb": {"kind": "stub", "rules": {}, "default": "pos"}}
    assert nb.run(bench, "dumb")["score"] == 0.5


def test_limit_param():
    assert len(nb.run(BENCH / "sentiment.yaml", limit=2)["cases"]) == 2


def test_two_benchmarks_independent():
    a, b = nb.load(BENCH / "sentiment.yaml"), nb.load(BENCH / "sentiment.yaml")
    ra, rb = nb.run(a), nb.run(b)
    assert ra["score"] == rb["score"] and ra is not rb


# ------------------------------------------------------------- loud errors

def test_every_bench_yaml_loads_and_compiles():
    """every yaml in bench/ parses and its system spec compiles offline
    (compilation builds the closure; only invoke would hit the network)."""
    yamls = sorted(BENCH.glob("*.yaml"))
    assert len(yamls) >= 3
    for path in yamls:
        bench = nb.load(path)
        assert set(bench) == {"task", "scoring", "system", "cases"}
        assert callable(nb.compile_system(bench["system"]))
        assert bench["cases"], f"{path.name} has no cases"


def test_unknown_scoring_raises():
    bench = nb.load(BENCH / "sentiment.yaml")
    bench["scoring"] = {"match": "vibes"}
    try:
        nb.run(bench)
        raise AssertionError("must raise")
    except ValueError as e:
        assert "scoring" in str(e)


def test_unknown_system_kind_raises():
    bench = nb.load(BENCH / "sentiment.yaml")
    bench["system"] = {"kind": "quantum"}
    try:
        nb.run(bench)
        raise AssertionError("must raise")
    except ValueError as e:
        assert "quantum" in str(e)


def test_dead_scoring_keys_raise():
    """s05's loud guard: unread keys in scoring are not silently accepted."""
    bench = nb.load(BENCH / "sentiment.yaml")
    bench["scoring"] = {"match": "exact", "aggregate": "mean"}
    try:
        nb.run(bench)
        raise AssertionError("must raise")
    except ValueError:
        pass


# ------------------------------------------------- cloud-first (mocked cloud)

def _mock_cloud(handler_cls):
    """an openai-compatible server on localhost; returns (server, url)."""
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


class _ChatEcho(BaseHTTPRequestHandler):
    """a sentiment oracle: neg iff the user message carries a negative cue."""
    NEG = ("broken", "porqueria", "never again")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]
        content = "neg" if any(w in user for w in self.NEG) else "pos"
        out = {"choices": [{"message": {"content": content}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())
    def log_message(self, *a):
        pass


def test_http_backend_scores_a_cloud_taker():
    """the whole cloud path: yaml spec -> compiled invoke -> mocked API."""
    server, url = _mock_cloud(_ChatEcho)
    try:
        bench = nb.load(BENCH / "sentiment.yaml")
        bench["system"] = {"kind": "http", "url": url, "model": "mock-1"}
        r = nb.run(bench)
        assert r["score"] == 1.0  # echo model is a perfect sentiment oracle
    finally:
        server.shutdown()


class _ChatFlakyOnce(_ChatEcho):
    hits = 0
    def do_POST(self):
        _ChatFlakyOnce.hits += 1
        if _ChatFlakyOnce.hits == 1:
            self.send_response(429)  # transient: the proctor must retry
            self.end_headers()
            return
        _ChatEcho.do_POST(self)


def test_http_backend_retries_transient_failures():
    server, url = _mock_cloud(_ChatFlakyOnce)
    try:
        bench = nb.load(BENCH / "sentiment.yaml")
        bench["system"] = {"kind": "http", "url": url, "model": "mock-1",
                           "retries": 1, "timeout": 5}
        assert nb.run(bench)["score"] == 1.0
        assert _ChatFlakyOnce.hits == 7  # 6 cases + 1 retried request
    finally:
        server.shutdown()


class _ChatJSON(_ChatEcho):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]
        total = re.search(r"\$(\d+)", user).group(1)
        vendor = re.search(r"from (\w+)", user).group(1)
        date = re.search(r"dated ([\w-]+)", user).group(1)
        out = {"choices": [{"message": {"content": json.dumps(
            {"total": total, "vendor": vendor, "date": date})}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


def test_http_json_parse_unfences_and_extracts():
    """structured output from a cloud taker, graded by weighted fields."""
    server, url = _mock_cloud(_ChatJSON)
    try:
        bench = nb.load(BENCH / "extract.yaml")
        bench["system"] = {"kind": "http", "url": url, "model": "mock-1",
                           "parse": "json"}
        assert nb.run(bench)["score"] == 1.0
    finally:
        server.shutdown()


# ------------------------------------------------------------- composition

def test_chain_is_config_not_code():
    """a workflow is just a chain: each link's output is the next's input."""
    bench = nb.load(BENCH / "sentiment.yaml")
    bench["system"] = {"kind": "chain", "steps": [
        {"kind": "regex", "if": "broken|porqueria|never again",
         "then": "bad", "else": "good"},          # stage 1: signal
        {"kind": "stub", "rules": {"bad": "neg", "good": "pos"},
         "default": "pos"}]}                      # stage 2: label
    assert nb.run(bench)["score"] == 1.0