"""End-to-end properties of the engine, on extraction and classification.

These answer three questions about the *engine*, not about any benchmark's realism:

    1. can one benchmark run against any kind of AI-system?
    2. does the scoring function actually express relative importance?
    3. can the engine reach the data — artifacts, big exams, and nothing outside?

All of it runs without an API key. Where a real model would only add noise, these use a
deterministic stand-in: the point is to check the engine's arithmetic and plumbing, and a
model's mood would obscure both. The live equivalents against Together and Bedrock live in
`examples/`.
"""

from __future__ import annotations

import json
import threading
import tracemalloc
import urllib.request
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from benchy import data
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run
from conftest import edit

# --- the two benchmark shapes that matter: extraction and classification ------------

EXTRACT_FIELDS = {"invoice_number": "string", "date": "date", "supplier": "string",
                  "subtotal": "float", "total": "float"}
EXTRACT_ROW = {"invoice_number": "A-001", "date": "2026-09-13", "supplier": "ACME SA",
               "subtotal": 100.0, "total": 121.0}
LABELS = ["facturacion", "envio", "devolucion", "otro"]


def extraction(weights=None, *, system=None, path="./exam.jsonl"):
    return compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": EXTRACT_FIELDS},
        scoring={"weights": weights or dict.fromkeys(EXTRACT_FIELDS, 1), "aggregator": "weighted_mean"},
        data={"path": path},
        ai_system=system or {"type": "external", "id": "system-under-test"},
    ))


def classification(*, system=None):
    return compile_benchmark(edit(
        benchmark={"task": "classify", "domain": "retail", "language": "es"},
        program={"input": {"message": "string"}, "output": {"intent": {"enum": LABELS}}},
        scoring={"weights": {"intent": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system=system or {"type": "external", "id": "system-under-test"},
    ))


def write_exam(root, rows, name="exam.jsonl"):
    (root / name).write_text("\n".join(json.dumps(r) for r in rows))
    return root


@pytest.fixture
def extraction_exam(tmp_path):
    return write_exam(tmp_path, [{"input": {"text": "Factura A-001"}, "expected": EXTRACT_ROW}])


@pytest.fixture
def classification_exam(tmp_path):
    return write_exam(tmp_path, [
        {"input": {"message": "Me cobraron dos veces"}, "expected": {"intent": "facturacion"}},
        {"input": {"message": "El paquete no llego"}, "expected": {"intent": "envio"}},
    ])


# ---------------------------------------------------------------------------
# 1. one benchmark, any kind of AI-system
# ---------------------------------------------------------------------------
#
# VISION's claim is that the primitive is the AI-system, not the model: a bare model, a
# node, a composed workflow and a served program are the same kind of thing from outside.
# Each case below is a *structurally different* system taking the identical benchmark.

def test_a_plain_function_is_an_ai_system(extraction_exam):
    async def go():
        return await run(extraction(), extraction_exam, lambda _: dict(EXTRACT_ROW))

    import asyncio

    assert asyncio.run(go())["benchmark_score"] == 1.0


async def test_an_ai_node_carries_its_own_prompt_and_parameters(extraction_exam):
    """A node is a model plus fixed task behaviour. Here the behaviour is state the
    adapter holds, which is exactly what distinguishes it from a bare callable."""

    class Node:
        def __init__(self, prompt, temperature):
            self.prompt, self.temperature, self.seen = prompt, temperature, []

        async def invoke(self, input_object):
            self.seen.append((self.prompt, self.temperature))
            return dict(EXTRACT_ROW)

    node = Node(prompt="Sos un extractor de facturas.", temperature=0)
    assert (await run(extraction(), extraction_exam, node))["benchmark_score"] == 1.0
    assert node.seen == [("Sos un extractor de facturas.", 0)]


async def test_a_workflow_of_several_models_is_one_ai_system(extraction_exam):
    """Two models and deterministic code between them. The engine sees one system."""
    calls: list[str] = []

    def draft(_):
        calls.append("model-a")
        return dict(EXTRACT_ROW, total=999.0)          # gets the money wrong

    def money(_):
        calls.append("model-b")
        return {"subtotal": 100.0, "total": 121.0}

    async def workflow(input_object):
        out = dict(draft(input_object))
        second = money(input_object)
        if abs(second["subtotal"] * 1.21 - second["total"]) < 0.01:   # adjudicate
            out.update(second)
        return out

    result = await run(extraction(), extraction_exam, workflow)
    assert result["benchmark_score"] == 1.0, "the composition should fix what one model got wrong"
    assert calls == ["model-a", "model-b"], "both models are actually consulted"


async def test_an_ai_program_behind_http_is_an_ai_system(extraction_exam):
    """A bespoke route, not an OpenAI-compatible one. The engine never learns it is HTTP."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(EXTRACT_ROW).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/extract"

    def over_http(input_object):
        request = urllib.request.Request(url, data=json.dumps(input_object).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        assert (await run(extraction(), extraction_exam, over_http))["benchmark_score"] == 1.0
    finally:
        server.shutdown()
        server.server_close()


async def test_classification_takes_the_same_four_shapes(classification_exam):
    """Whatever works for extraction works for classification: the task is not the seam."""

    class Node:
        async def invoke(self, input_object):
            return {"intent": "facturacion" if "cobraron" in input_object["message"] else "envio"}

    for system in (lambda i: {"intent": "facturacion" if "cobraron" in i["message"] else "envio"}, Node()):
        assert (await run(classification(), classification_exam, system))["benchmark_score"] == 1.0


# ---------------------------------------------------------------------------
# 2. the scoring function expresses importance
# ---------------------------------------------------------------------------

def wrong_about_total(_):
    """Right on four fields, wrong on `total`, every time — so the score is computable."""
    return dict(EXTRACT_ROW, total=EXTRACT_ROW["total"] + 1.0)


@pytest.mark.parametrize(
    "weights,expected,meaning",
    [
        ({"invoice_number": 1, "date": 1, "supplier": 1, "subtotal": 1, "total": 1}, Fraction(4, 5), "all equal"),
        ({"invoice_number": 1, "date": 1, "supplier": 1, "subtotal": 1, "total": 5}, Fraction(4, 9), "total matters 5x"),
        ({"invoice_number": 0, "date": 0, "supplier": 0, "subtotal": 0, "total": 1}, Fraction(0), "only total counts"),
        ({"invoice_number": 1, "date": 1, "supplier": 1, "subtotal": 1, "total": 0}, Fraction(1), "total ignored"),
    ],
    ids=["equal", "total-heavy", "total-only", "total-zero"],
)
async def test_weights_change_the_score_exactly(extraction_exam, weights, expected, meaning):
    """Same system, same exam — only the declared importance moves."""
    result = await run(extraction(weights), extraction_exam, wrong_about_total)
    assert result["benchmark_score"] == pytest.approx(float(expected), abs=1e-12), meaning


async def test_a_zero_weight_field_is_still_validated(extraction_exam):
    """Weight 0 means "does not count", not "not checked" — a wrong *type* there is still
    an invalid_output."""
    weights = dict.fromkeys(EXTRACT_FIELDS, 1) | {"total": 0}
    result = await run(extraction(weights), extraction_exam, lambda _: dict(EXTRACT_ROW, total="lots"))
    (only,) = result["results"]
    assert only["status"] == "invalid_output"
    assert only["error"]["code"] == "wrong_type"


async def test_field_scores_expose_the_arithmetic(extraction_exam):
    """A reader must be able to see *why* a score is what it is."""
    weights = dict.fromkeys(EXTRACT_FIELDS, 1) | {"total": 5}
    (only,) = (await run(extraction(weights), extraction_exam, wrong_about_total))["results"]
    scored = {tuple(f["path"]): (f["score"], f["weight"]) for f in only["field_scores"]}
    assert scored[("total",)] == (0, 5.0)
    assert scored[("supplier",)] == (1, 1.0)
    assert sum(w for _, w in scored.values()) == 9.0


# ---------------------------------------------------------------------------
# 3. reaching the data
# ---------------------------------------------------------------------------

def artifact_benchmark(path="./data/exam.jsonl"):
    return compile_benchmark(edit(
        program={"input": {"scan": "image"}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
        data={"path": path},
        ai_system={"type": "external", "id": "system-under-test"},
    ))


async def test_artifacts_resolve_from_the_exam_not_the_working_directory(tmp_path):
    """The exam says `assets/scan.png`; the adapter is handed an absolute path."""
    nested = tmp_path / "data" / "assets"
    nested.mkdir(parents=True)
    (nested / "scan.png").write_bytes(b"\x89PNG")
    write_exam(tmp_path / "data", [{"input": {"scan": "assets/scan.png"}, "expected": {"total": 1.0}}])

    seen: list[str] = []

    def record(input_object):
        seen.append(input_object["scan"])
        return {"total": 1.0}

    assert (await run(artifact_benchmark(), tmp_path, record))["benchmark_score"] == 1.0
    assert seen == [str(nested / "scan.png")]


def test_the_exam_is_streamed_not_loaded(tmp_path):
    """Memory must not grow with the size of the exam."""
    rows = [{"input": {"text": f"row-{i}"}, "expected": EXTRACT_ROW} for i in range(500)]
    write_exam(tmp_path, rows)

    tracemalloc.start()
    try:
        peaks = []
        for count, _ in enumerate(data.examples(extraction(), tmp_path), start=1):
            if count in (1, 500):
                peaks.append(tracemalloc.get_traced_memory()[1])
        assert count == 500
        first, last = peaks
        assert last < first * 3, f"peak memory grew from {first} to {last} bytes across the exam"
    finally:
        tracemalloc.stop()


async def test_a_dataset_error_aborts_and_emits_no_score(tmp_path):
    """Even with a perfectly good first row. A bad exam never becomes a number."""
    write_exam(tmp_path, [
        {"input": {"text": "fine"}, "expected": EXTRACT_ROW},
        {"input": {"text": "broken"}, "expected": {"invoice_number": "A-002"}},
    ])
    with pytest.raises(BenchyError) as exc:
        await run(extraction(), tmp_path, lambda _: dict(EXTRACT_ROW))
    assert (exc.value.phase, exc.value.code) == ("dataset", "missing_field")


@pytest.mark.parametrize("reference", ["../../outside.png", "/etc/hosts"], ids=["traversal", "absolute"])
async def test_the_exam_cannot_reach_outside_the_workspace(tmp_path, reference):
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (tmp_path / "outside.png").write_bytes(b"secret")
    write_exam(workspace / "data", [{"input": {"scan": reference}, "expected": {"total": 1.0}}])

    with pytest.raises(BenchyError) as exc:
        await run(artifact_benchmark(), workspace, lambda _: {"total": 1.0})
    assert exc.value.code == "path_escape"


async def test_a_symlink_out_of_the_workspace_is_refused(tmp_path):
    """Confinement follows links: the real path is what must stay inside."""
    workspace = tmp_path / "workspace"
    (workspace / "data" / "assets").mkdir(parents=True)
    (tmp_path / "secret.png").write_bytes(b"secret")
    (workspace / "data" / "assets" / "link.png").symlink_to(tmp_path / "secret.png")
    write_exam(workspace / "data", [{"input": {"scan": "assets/link.png"}, "expected": {"total": 1.0}}])

    with pytest.raises(BenchyError) as exc:
        await run(artifact_benchmark(), workspace, lambda _: {"total": 1.0})
    assert exc.value.code == "path_escape"


async def test_the_exam_can_live_anywhere_under_the_workspace(tmp_path):
    """`data.path` resolves from the workspace root, so layout is the author's choice."""
    (tmp_path / "exams" / "q3").mkdir(parents=True)
    write_exam(tmp_path / "exams" / "q3", [{"input": {"text": "x"}, "expected": EXTRACT_ROW}])
    benchmark = extraction(path="exams/q3/exam.jsonl")
    assert (await run(benchmark, tmp_path, lambda _: dict(EXTRACT_ROW)))["benchmark_score"] == 1.0
