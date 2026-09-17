"""Phase 11 — the OpenAI-compatible provider adapter.

Exercised against a local HTTP server speaking the chat-completions shape, so these
need no API key and no network. That proves schema generation, message construction,
transport and response handling; it does not prove any real provider's quirks.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from benchy import providers
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run
from conftest import edit

TEXT = edit(
    program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
    scoring={"weights": {"supplier": 1, "total": 1}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
    ai_system={"type": "model", "provider": "openai", "model": "test-model"},
)
IMAGE = edit(
    program={"input": {"image": "image"}, "output": {"total": "float"}},
    scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
    ai_system={"type": "model", "provider": "openai", "model": "test-model"},
)


class Provider:
    """A stand-in chat-completions endpoint that records what it was sent."""

    def __init__(self, reply="{}", status=200):
        self.reply, self.status, self.requests = reply, status, []
        handler = self._handler()
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "payload": json.loads(body)})
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {"choices": [{"message": {"content": outer.reply}}]}
                self.wfile.write(json.dumps(payload).encode())

        return Handler

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider():
    p = Provider()
    yield p
    p.close()


@pytest.fixture
def workspace(tmp_path):
    rows = [{"input": {"text": "factura"}, "expected": {"supplier": "ACME", "total": 121.0}}]
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def adapter_for(ir, workspace, provider, **env):
    return providers.for_system(ir, workspace, env={"OPENAI_BASE_URL": provider.base_url,
                                                    "OPENAI_API_KEY": "test-key", **env})


# ---------------------------------------------------------------------------
# selection (A.11)
# ---------------------------------------------------------------------------

def test_a_model_ai_system_resolves_to_a_provider_adapter(tmp_path, provider):
    ir = compile_benchmark(TEXT)
    assert adapter_for(ir, tmp_path, provider) is not None


def test_an_external_ai_system_has_no_provider_adapter(tmp_path):
    ir = compile_benchmark(edit(data={"path": "./exam.jsonl"}))
    with pytest.raises(BenchyError) as exc:
        providers.for_system(ir, tmp_path, env={})
    assert exc.value.code == "adapter_not_bound"


def test_an_unknown_provider_is_a_setup_error(tmp_path):
    ir = compile_benchmark(edit(
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "wizard-ai", "model": "m"},
    ))
    with pytest.raises(BenchyError) as exc:
        providers.for_system(ir, tmp_path, env={})
    assert exc.value.code == "adapter_not_bound"
    assert "wizard-ai" in exc.value.message


# ---------------------------------------------------------------------------
# the request it builds
# ---------------------------------------------------------------------------

async def test_output_schema_becomes_a_strict_json_schema(workspace, provider):
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    ir = compile_benchmark(TEXT)
    await run(ir, workspace, adapter_for(ir, workspace, provider))

    schema = provider.requests[0]["payload"]["response_format"]["json_schema"]["schema"]
    assert schema == {
        "type": "object",
        "properties": {"supplier": {"type": "string"}, "total": {"type": "number"}},
        "required": ["supplier", "total"],
        "additionalProperties": False,
    }
    assert provider.requests[0]["payload"]["response_format"]["json_schema"]["strict"] is True


async def test_nested_and_enum_output_schemas(tmp_path, provider):
    provider.reply = json.dumps({"supplier": {"name": "ACME"}, "kind": "a"})
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"},
                 "output": {"supplier": {"name": "string"}, "kind": {"enum": ["a", "b"]}}},
        scoring={"weights": {"supplier": {"name": 1}, "kind": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "test-model"},
    ))
    (tmp_path / "exam.jsonl").write_text(
        json.dumps({"input": {"text": "x"}, "expected": {"supplier": {"name": "ACME"}, "kind": "a"}}))
    await run(ir, tmp_path, adapter_for(ir, tmp_path, provider))

    props = provider.requests[0]["payload"]["response_format"]["json_schema"]["schema"]["properties"]
    assert props["supplier"] == {
        "type": "object", "properties": {"name": {"type": "string"}},
        "required": ["name"], "additionalProperties": False,
    }
    assert props["kind"] == {"type": "string", "enum": ["a", "b"]}


async def test_temporal_types_carry_their_canonical_format_to_the_model(tmp_path, provider):
    provider.reply = json.dumps({"when": "2026-09-13"})
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"when": "date"}},
        scoring={"weights": {"when": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "test-model"},
    ))
    (tmp_path / "exam.jsonl").write_text(
        json.dumps({"input": {"text": "x"}, "expected": {"when": "2026-09-13"}}))
    await run(ir, tmp_path, adapter_for(ir, tmp_path, provider))

    when = provider.requests[0]["payload"]["response_format"]["json_schema"]["schema"]["properties"]["when"]
    assert when["type"] == "string"
    assert "YYYY-MM-DD" in when["description"]


async def test_model_and_credentials_come_from_the_definition_and_the_environment(workspace, provider):
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    ir = compile_benchmark(TEXT)
    await run(ir, workspace, adapter_for(ir, workspace, provider))

    request = provider.requests[0]
    assert request["payload"]["model"] == "test-model"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["path"] == "/v1/chat/completions"


async def test_input_fields_reach_the_model_as_text(workspace, provider):
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    ir = compile_benchmark(TEXT)
    await run(ir, workspace, adapter_for(ir, workspace, provider))

    messages = provider.requests[0]["payload"]["messages"]
    text = json.dumps(messages)
    assert "factura" in text and "text" in text


async def test_a_prompt_file_becomes_the_system_message(workspace, provider):
    (workspace / "prompt.md").write_text("You extract invoices.")
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
        scoring={"weights": {"supplier": 1, "total": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "m", "prompt": "./prompt.md"},
    ))
    await run(ir, workspace, adapter_for(ir, workspace, provider))

    system = provider.requests[0]["payload"]["messages"][0]
    assert system["role"] == "system"
    assert "You extract invoices." in json.dumps(system)


async def test_parameters_pass_through_verbatim(workspace, provider):
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
        scoring={"weights": {"supplier": 1, "total": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "m",
                   "parameters": {"temperature": 0, "seed": 7}},
    ))
    await run(ir, workspace, adapter_for(ir, workspace, provider))

    payload = provider.requests[0]["payload"]
    assert payload["temperature"] == 0
    assert payload["seed"] == 7


async def test_an_image_input_is_sent_as_a_base64_data_url(tmp_path, provider):
    (tmp_path / "invoice.png").write_bytes(b"\x89PNG-bytes")
    (tmp_path / "exam.jsonl").write_text(
        json.dumps({"input": {"image": "invoice.png"}, "expected": {"total": 1.0}}))
    provider.reply = json.dumps({"total": 1.0})
    ir = compile_benchmark(IMAGE)
    await run(ir, tmp_path, adapter_for(ir, tmp_path, provider))

    parts = provider.requests[0]["payload"]["messages"][-1]["content"]
    image = next(p for p in parts if p["type"] == "image_url")
    expected = base64.b64encode(b"\x89PNG-bytes").decode()
    assert image["image_url"]["url"] == f"data:image/png;base64,{expected}"


# ---------------------------------------------------------------------------
# what it does with the response
# ---------------------------------------------------------------------------

async def test_a_conforming_reply_scores(workspace, provider):
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    ir = compile_benchmark(TEXT)
    result = await run(ir, workspace, adapter_for(ir, workspace, provider))
    assert result["benchmark_score"] == 1.0
    assert result["results"][0]["status"] == "valid"


async def test_the_adapter_does_not_coerce_types(workspace, provider):
    """A model returning "121.00" for a float is a true invalid_output, not a defect
    for the adapter to paper over."""
    provider.reply = json.dumps({"supplier": "ACME", "total": "121.00"})
    ir = compile_benchmark(TEXT)
    result = await run(ir, workspace, adapter_for(ir, workspace, provider))
    assert result["results"][0]["status"] == "invalid_output"
    assert result["results"][0]["error"]["code"] == "wrong_type"


async def test_a_non_json_reply_becomes_an_invalid_output_retaining_the_text(workspace, provider):
    provider.reply = "I'm afraid I can't do that."
    ir = compile_benchmark(TEXT)
    result = await run(ir, workspace, adapter_for(ir, workspace, provider))
    assert result["results"][0]["status"] == "invalid_output"
    assert result["results"][0]["prediction"] == "I'm afraid I can't do that."


async def test_a_provider_failure_becomes_an_execution_error(workspace):
    failing = Provider(status=500)
    try:
        ir = compile_benchmark(TEXT)
        result = await run(ir, workspace, adapter_for(ir, workspace, failing))
        assert result["results"][0]["status"] == "execution_error"
        assert result["results"][0]["error"]["code"] == "adapter_error"
    finally:
        failing.close()


# ---------------------------------------------------------------------------
# honest limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artifact", ["audio", "document"])
def test_unsupported_input_artifacts_fail_at_setup_not_silently(tmp_path, artifact):
    ir = compile_benchmark(edit(
        program={"input": {"thing": artifact}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "m"},
    ))
    with pytest.raises(BenchyError) as exc:
        providers.for_system(ir, tmp_path, env={"OPENAI_API_KEY": "k"})
    assert artifact in exc.value.message


def test_artifact_outputs_fail_at_setup(tmp_path):
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"picture": "image"}},
        scoring={"weights": {"picture": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "m"},
    ))
    with pytest.raises(BenchyError) as exc:
        providers.for_system(ir, tmp_path, env={"OPENAI_API_KEY": "k"})
    assert "image" in exc.value.message


def test_a_missing_api_key_is_a_setup_error(tmp_path):
    ir = compile_benchmark(TEXT)
    with pytest.raises(BenchyError) as exc:
        providers.for_system(ir, tmp_path, env={})
    assert exc.value.code == "adapter_not_bound"
    assert "OPENAI_API_KEY" in exc.value.message


# ---------------------------------------------------------------------------
# the boundary itself
# ---------------------------------------------------------------------------

def test_the_engine_core_does_not_import_providers():
    """A.11: provider integrations live outside the core. `cli` may select one."""
    from pathlib import Path

    core = ["errors", "types", "ontology", "compiler", "data", "score", "adapter", "run"]
    root = Path(__file__).resolve().parents[1] / "benchy"
    for name in core:
        assert "providers" not in (root / f"{name}.py").read_text(), name


# ---------------------------------------------------------------------------
# CLI selection
# ---------------------------------------------------------------------------

def test_cli_selects_a_provider_when_no_adapter_is_named(workspace, provider, monkeypatch, capsys):
    from benchy.cli import main

    monkeypatch.setenv("OPENAI_BASE_URL", provider.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider.reply = json.dumps({"supplier": "ACME", "total": 121.0})
    (workspace / "benchmark.yaml").write_text(TEXT)

    assert main(["run", str(workspace / "benchmark.yaml")]) == 0
    assert json.loads(capsys.readouterr().out)["benchmark_score"] == 1.0


def test_cli_requires_an_adapter_for_an_external_ai_system(tmp_path, capsys):
    from benchy.cli import main

    (tmp_path / "benchmark.yaml").write_text(edit(data={"path": "./exam.jsonl"}))
    (tmp_path / "exam.jsonl").write_text("{}")
    assert main(["run", str(tmp_path / "benchmark.yaml")]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "adapter_not_bound"


def test_an_explicit_adapter_still_wins(workspace, provider, monkeypatch, capsys):
    from benchy.cli import main

    monkeypatch.setenv("OPENAI_BASE_URL", provider.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (workspace / "benchmark.yaml").write_text(TEXT)
    (workspace / "mine.py").write_text(
        'def system(_):\n    return {"supplier": "MINE", "total": 0.0}\n')

    assert main(["run", str(workspace / "benchmark.yaml"),
                 "--adapter", f"{workspace / 'mine.py'}:system"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["prediction"]["supplier"] == "MINE"
    assert provider.requests == []   # the provider was never contacted
