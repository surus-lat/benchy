"""The built-in provider adapter, with `llm_client` as its transport.

Exercised against a local HTTP server speaking the chat-completions shape, so these
need no API key and no network. That proves schema generation, message construction,
parameter handling and response classification; it does not prove any real provider's
quirks — those were checked live and are recorded where they matter.

Needs the `providers` extra; skipped without it.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from benchy import providers
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run
from conftest import edit

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("llm_client") is None,
    reason="needs the providers extra: pip install 'benchy[providers]'",
)


def _has_converse() -> bool:
    """Bedrock Converse support lives in llm_client; it is unmerged as of writing.

    See surus-lat/llm-client#4. These tests activate the moment it lands.
    """
    try:
        from llm_client import profiles

        return hasattr(profiles, "BedrockConverseProfile")
    except ImportError:
        return False


needs_converse = pytest.mark.skipif(
    not _has_converse(), reason="needs llm_client with BedrockConverseProfile (surus-lat/llm-client#4)"
)

TEXT = edit(
    program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
    scoring={"weights": {"supplier": 1, "total": 1}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
    ai_system={"type": "model", "provider": "openai", "model": "test-model"},
)
GOOD = json.dumps({"supplier": "ACME", "total": 121.0})


def benchmark(provider="openai", *, inp=None, out=None, model_override=None, **system):
    out = out or {"total": "float"}
    weights = {name: 1 for name in out}
    system.setdefault("model", model_override or "test-model")
    return compile_benchmark(edit(
        program={"input": inp or {"text": "string"}, "output": out},
        scoring={"weights": weights, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": provider, **system},
    ))


def reply(content, finish_reason="stop", **message):
    return 200, {"choices": [{"finish_reason": finish_reason, "message": {"content": content, **message}}]}, {}


class Provider:
    """A stand-in chat-completions endpoint that records requests and replays a script.

    Each request consumes the next scripted `(status, body, headers)`; once the script
    runs out, every further request gets `reply(self.content)`.
    """

    def __init__(self, content=GOOD, script=()):
        self.content, self.script, self.requests = content, list(script), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "payload": json.loads(body)})
                status, payload, headers = outer.script.pop(0) if outer.script else reply(outer.content)
                self.send_response(status)
                for name, value in {"Content-Type": "application/json", **headers}.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def payload(self) -> dict:
        return self.requests[0]["payload"]

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


def adapter_for(ir, workspace, provider):
    return providers.for_system(ir, workspace, env={"OPENAI_BASE_URL": provider.base_url,
                                                    "OPENAI_API_KEY": "test-key"})


async def execute(ir, workspace, provider):
    return await run(ir, workspace, adapter_for(ir, workspace, provider))


def setup_error(ir, tmp_path, env) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        providers.for_system(ir, tmp_path, env=env)
    assert exc.value.code == "adapter_not_bound"
    return exc.value


# ---------------------------------------------------------------------------
# selection (A.11)
# ---------------------------------------------------------------------------

def test_a_model_ai_system_resolves_to_a_provider_adapter(tmp_path, provider):
    assert adapter_for(compile_benchmark(TEXT), tmp_path, provider) is not None


def test_an_external_ai_system_has_no_provider_adapter(tmp_path):
    setup_error(compile_benchmark(edit(data={"path": "./exam.jsonl"})), tmp_path, {})


def test_an_unknown_provider_is_a_setup_error(tmp_path):
    assert "wizard-ai" in setup_error(benchmark("wizard-ai"), tmp_path, {}).message


# ---------------------------------------------------------------------------
# endpoints and credentials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider,env,url",
    [
        ("openai", {"OPENAI_API_KEY": "k"}, "https://api.openai.com/v1"),
        ("together", {"TOGETHER_API_KEY": "k"}, "https://api.together.xyz/v1"),
        ("bedrock", {"AWS_BEARER_TOKEN_BEDROCK": "k", "AWS_REGION": "us-east-1"},
         "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"),
    ],
)
def test_each_provider_has_its_own_endpoint_and_credential(tmp_path, provider, env, url):
    adapter = providers.for_system(benchmark(provider), tmp_path, env=env)
    assert (adapter.base_url, adapter.api_key) == (url, "k")


@pytest.mark.parametrize(
    "provider,key_var",
    [("openai", "OPENAI_API_KEY"), ("together", "TOGETHER_API_KEY"), ("bedrock", "AWS_BEARER_TOKEN_BEDROCK")],
)
def test_a_providers_base_url_is_overridable(tmp_path, provider, key_var):
    env = {key_var: "k", f"{provider.upper()}_BASE_URL": "http://localhost:8000/v1/"}
    assert providers.for_system(benchmark(provider), tmp_path, env=env).base_url == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    "provider,key_var",
    [("openai", "OPENAI_API_KEY"), ("together", "TOGETHER_API_KEY"), ("bedrock", "AWS_BEARER_TOKEN_BEDROCK")],
)
def test_a_missing_credential_is_named_in_the_setup_error(tmp_path, provider, key_var):
    env = {"AWS_REGION": "us-east-1"}
    assert key_var in setup_error(benchmark(provider), tmp_path, env).message


def test_bedrock_needs_a_region_and_says_so(tmp_path):
    error = setup_error(benchmark("bedrock"), tmp_path, {"AWS_BEARER_TOKEN_BEDROCK": "k"})
    assert "AWS_REGION" in error.message


# ---------------------------------------------------------------------------
# the request it builds
# ---------------------------------------------------------------------------

async def test_output_schema_becomes_a_strict_json_schema(workspace, provider):
    await execute(compile_benchmark(TEXT), workspace, provider)
    response_format = provider.payload["response_format"]
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == {
        "type": "object",
        "properties": {"supplier": {"type": "string"}, "total": {"type": "number"}},
        "required": ["supplier", "total"],
        "additionalProperties": False,
    }


async def test_nested_and_enum_output_schemas(tmp_path, provider):
    provider.content = json.dumps({"supplier": {"name": "ACME"}, "kind": "a"})
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"},
                 "output": {"supplier": {"name": "string"}, "kind": {"enum": ["a", "b"]}}},
        scoring={"weights": {"supplier": {"name": 1}, "kind": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "model", "provider": "openai", "model": "test-model"},
    ))
    (tmp_path / "exam.jsonl").write_text(
        json.dumps({"input": {"text": "x"}, "expected": {"supplier": {"name": "ACME"}, "kind": "a"}}))
    await execute(ir, tmp_path, provider)

    properties = provider.payload["response_format"]["json_schema"]["schema"]["properties"]
    assert properties["supplier"] == {
        "type": "object", "properties": {"name": {"type": "string"}},
        "required": ["name"], "additionalProperties": False,
    }
    assert properties["kind"] == {"type": "string", "enum": ["a", "b"]}


async def test_temporal_types_carry_their_canonical_format_to_the_model(tmp_path, provider):
    provider.content = json.dumps({"when": "2026-09-13"})
    (tmp_path / "exam.jsonl").write_text(json.dumps({"input": {"text": "x"}, "expected": {"when": "2026-09-13"}}))
    await execute(benchmark(out={"when": "date"}), tmp_path, provider)

    when = provider.payload["response_format"]["json_schema"]["schema"]["properties"]["when"]
    assert when["type"] == "string"
    assert "YYYY-MM-DD" in when["description"]


async def test_model_and_credentials_come_from_the_definition_and_the_environment(workspace, provider):
    await execute(compile_benchmark(TEXT), workspace, provider)
    request = provider.requests[0]
    assert request["payload"]["model"] == "test-model"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["path"] == "/v1/chat/completions"


async def test_input_fields_reach_the_model_as_text(workspace, provider):
    await execute(compile_benchmark(TEXT), workspace, provider)
    assert "text: factura" in json.dumps(provider.payload["messages"])


async def test_a_prompt_file_becomes_the_system_message(workspace, provider):
    (workspace / "prompt.md").write_text("You extract invoices.")
    await execute(benchmark(out={"supplier": "string", "total": "float"}, prompt="./prompt.md"), workspace, provider)
    assert provider.payload["messages"][0] == {"role": "system", "content": "You extract invoices."}


async def test_an_image_input_is_sent_as_a_base64_data_url(tmp_path, provider):
    (tmp_path / "invoice.png").write_bytes(b"\x89PNG-bytes")
    (tmp_path / "exam.jsonl").write_text(json.dumps({"input": {"image": "invoice.png"}, "expected": {"total": 1.0}}))
    provider.content = json.dumps({"total": 1.0})
    await execute(benchmark(inp={"image": "image"}), tmp_path, provider)

    parts = provider.payload["messages"][-1]["content"]
    image = next(p for p in parts if p["type"] == "image_url")
    assert image["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(b'\x89PNG-bytes').decode()}"


async def test_requests_do_not_go_out_as_python_urllib(workspace, provider):
    """Together's WAF returns 403 (Cloudflare 1010) for urllib's default User-Agent.

    Found live. The transport is now `llm_client` over httpx, whose default agent was
    checked against Together and accepted; this pins that the blocked one never returns.
    """
    await execute(compile_benchmark(TEXT), workspace, provider)
    assert "urllib" not in provider.requests[0]["headers"]["User-Agent"].lower()


# ---------------------------------------------------------------------------
# parameters — only what will actually reach the model
# ---------------------------------------------------------------------------

async def test_temperature_and_max_tokens_reach_the_model(workspace, provider):
    ir = benchmark(out={"supplier": "string", "total": "float"}, parameters={"temperature": 0, "max_tokens": 512})
    await execute(ir, workspace, provider)
    assert provider.payload["temperature"] == 0
    assert provider.payload["max_tokens"] == 512


async def test_llm_client_defaults_are_never_injected(workspace, provider):
    """`llm_client.call` defaults to temperature 0.1 and max_tokens 2000.

    A benchmark that sets neither must not be run with either: that would be a hidden
    default changing what is measured. Temperature is omitted; max_tokens is sent as
    null, which leaves the server's own default in force (checked live on Together).
    """
    await execute(compile_benchmark(TEXT), workspace, provider)
    assert "temperature" not in provider.payload
    assert provider.payload.get("max_tokens") is None


@pytest.mark.parametrize("parameter", ["seed", "top_p", "stop"])
def test_parameters_that_would_be_silently_dropped_are_rejected_at_setup(tmp_path, parameter):
    """`llm_client` forwards anything but its named arguments as `extra_body`, nested.

    Checked live against Together: a nested `stop` is ignored without an error. A
    benchmark declaring such a parameter would run without it while claiming otherwise,
    so it is refused before the first example rather than silently not applied.
    """
    ir = benchmark(parameters={"temperature": 0, parameter: 1})
    error = setup_error(ir, tmp_path, {"OPENAI_API_KEY": "k"})
    assert parameter in error.message
    assert "silently" in error.message


# ---------------------------------------------------------------------------
# measurement guarantees
# ---------------------------------------------------------------------------

async def test_a_rejected_request_is_not_retried_without_the_schema(workspace):
    """`llm_client` can fall back json_schema -> json_object -> no format.

    That would score a system on an easier task than the benchmark declares, so it is
    never enabled: one rejected request is one execution_error.
    """
    failing = Provider(script=[(400, {"error": {"message": "schema not supported"}}, {})])
    try:
        result = await execute(compile_benchmark(TEXT), workspace, failing)
        assert result["results"][0]["status"] == "execution_error"
        assert len(failing.requests) == 1
        assert failing.payload["response_format"]["type"] == "json_schema"
    finally:
        failing.close()


async def test_a_transient_rate_limit_is_retried_rather_than_scored(workspace):
    """A 429 says nothing about the system under test, so it must not count against it."""
    limited = Provider(script=[(429, {"error": {"message": "slow down"}}, {"Retry-After": "0"})])
    try:
        result = await execute(compile_benchmark(TEXT), workspace, limited)
        assert result["results"][0]["status"] == "valid"
        assert len(limited.requests) == 2
    finally:
        limited.close()


async def test_the_providers_error_body_reaches_the_result(workspace):
    failing = Provider(script=[(401, {"error": {"message": "Invalid API Key format"}}, {})])
    try:
        result = await execute(compile_benchmark(TEXT), workspace, failing)
        (only,) = result["results"]
        assert only["status"] == "execution_error"
        assert "401" in only["error"]["message"]
        assert "Invalid API Key format" in only["error"]["message"]
    finally:
        failing.close()


# ---------------------------------------------------------------------------
# what it does with the response
# ---------------------------------------------------------------------------

async def test_a_conforming_reply_scores(workspace, provider):
    result = await execute(compile_benchmark(TEXT), workspace, provider)
    assert result["benchmark_score"] == 1.0
    assert result["results"][0]["status"] == "valid"


async def test_the_adapter_does_not_coerce_types(workspace, provider):
    """A model returning "121.00" for a float is a true invalid_output, not a defect
    for the adapter to paper over."""
    provider.content = json.dumps({"supplier": "ACME", "total": "121.00"})
    result = await execute(compile_benchmark(TEXT), workspace, provider)
    assert result["results"][0]["status"] == "invalid_output"
    assert result["results"][0]["error"]["code"] == "wrong_type"


async def test_a_non_json_reply_becomes_an_invalid_output_retaining_the_text(workspace, provider):
    provider.content = "I'm afraid I can't do that."
    result = await execute(compile_benchmark(TEXT), workspace, provider)
    assert result["results"][0]["status"] == "invalid_output"
    assert result["results"][0]["prediction"] == "I'm afraid I can't do that."


async def test_a_server_failure_becomes_an_execution_error(workspace):
    failing = Provider(script=[(500, {"error": {"message": "boom"}}, {})])
    try:
        result = await execute(compile_benchmark(TEXT), workspace, failing)
        assert result["results"][0]["status"] == "execution_error"
        assert result["results"][0]["error"]["code"] == "adapter_error"
    finally:
        failing.close()


@pytest.mark.parametrize(
    "response",
    [
        reply("", finish_reason="length", reasoning_content="thinking..."),
        reply(None, reasoning_content="thought"),
    ],
    ids=["truncated", "null-content"],
)
async def test_an_empty_reply_explains_itself(workspace, response):
    """A reply with no content at all must say why, not look like a bad answer.

    A stand-in only proves what is scripted into it: live, a small budget more often
    returns *partial* content — see the truncation tests below."""
    empty = Provider(script=[response])
    try:
        result = await execute(compile_benchmark(TEXT), workspace, empty)
        (only,) = result["results"]
        assert only["status"] == "execution_error"
        # Which message depends on whether the client surfaces finish_reason; both say
        # what to do about it, which is the part that matters.
        assert "max_tokens" in only["error"]["message"]
    finally:
        empty.close()


# ---------------------------------------------------------------------------
# honest limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artifact", ["audio", "document"])
def test_unsupported_input_artifacts_fail_at_setup_not_silently(tmp_path, artifact):
    error = setup_error(benchmark(inp={"thing": artifact}), tmp_path, {"OPENAI_API_KEY": "k"})
    assert artifact in error.message


def test_artifact_outputs_fail_at_setup(tmp_path):
    assert "image" in setup_error(benchmark(out={"picture": "image"}), tmp_path, {"OPENAI_API_KEY": "k"}).message


# ---------------------------------------------------------------------------
# CLI selection
# ---------------------------------------------------------------------------

def test_cli_selects_a_provider_when_no_adapter_is_named(workspace, provider, monkeypatch, capsys):
    from benchy.cli import main

    monkeypatch.setenv("OPENAI_BASE_URL", provider.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (workspace / "benchmark.yaml").write_text(TEXT)

    assert main(["run", str(workspace / "benchmark.yaml")]) == 0
    assert json.loads(capsys.readouterr().out)["benchmark_score"] == 1.0


def test_an_explicit_adapter_still_wins(workspace, provider, monkeypatch, capsys):
    from benchy.cli import main

    monkeypatch.setenv("OPENAI_BASE_URL", provider.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (workspace / "benchmark.yaml").write_text(TEXT)
    (workspace / "mine.py").write_text('def system(_):\n    return {"supplier": "MINE", "total": 0.0}\n')

    assert main(["run", str(workspace / "benchmark.yaml"), "--adapter", f"{workspace / 'mine.py'}:system"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["prediction"]["supplier"] == "MINE"
    assert provider.requests == []  # the provider was never contacted


async def test_a_truncated_answer_is_explained_not_scored_as_the_systems_own(workspace):
    """Checked live on Together with max_tokens 16: finish_reason "length" and partial JSON.

    Unexplained, that partial JSON lands as invalid_output — "expected an object, got
    str" — as though the system wrote a malformed answer. `llm_client` only reports
    finish_reason from the version that exposes it, so this injects a client that does.
    """
    async def truncating_call(**_):
        return {"content": '{\n  "supplier": "ACME",', "finish_reason": "length",
                "usage": {}, "model": "m", "logprobs": None}

    ir = compile_benchmark(TEXT)
    adapter = providers.OpenAIChat(ir, workspace, {"OPENAI_API_KEY": "k"}, truncating_call)
    (only,) = (await run(ir, workspace, adapter))["results"]
    assert only["status"] == "execution_error"
    assert "token limit" in only["error"]["message"]
    assert "max_tokens" in only["error"]["message"]


async def test_without_finish_reason_a_truncation_still_scores_nothing(workspace):
    """The `llm_client` in use today drops finish_reason. The diagnosis is weaker, but the
    measurement is not: the example is still null-scored and contributes zero."""
    async def call_without_finish_reason(**_):
        return {"content": '{\n  "supplier": "ACME",', "usage": {}, "model": "m", "logprobs": None}

    ir = compile_benchmark(TEXT)
    adapter = providers.OpenAIChat(ir, workspace, {"OPENAI_API_KEY": "k"}, call_without_finish_reason)
    result = await run(ir, workspace, adapter)
    (only,) = result["results"]
    assert (only["status"], only["score"], only["contribution"]) == ("invalid_output", None, 0.0)
    assert only["prediction"] == '{\n  "supplier": "ACME",'


# ---------------------------------------------------------------------------
# Bedrock + Claude: Converse rather than chat completions
# ---------------------------------------------------------------------------

class ConverseProvider(Provider):
    """A stand-in Bedrock Converse endpoint: forced tool in, tool input out."""

    def __init__(self, stop_reason="tool_use", output=None):
        self.stop_reason = stop_reason
        self.output = {"invoice": "A-001"} if output is None else output
        super().__init__()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "payload": json.loads(body)})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "stopReason": outer.stop_reason,
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "output": {"message": {"content": [{"toolUse": {"input": outer.output}}]}},
                }).encode())

        self.server.shutdown()
        self.server.server_close()
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def bedrock_adapter(ir, workspace, provider):
    return providers.for_system(ir, workspace, env={
        "BEDROCK_BASE_URL": provider.base_url.replace("/v1", ""),
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-key",
    })


@needs_converse
async def test_a_claude_model_on_bedrock_goes_to_converse_with_a_forced_tool(workspace):
    """Claude does not serve Bedrock's chat-completions endpoint, so llm_client routes
    it to Converse. benchy needs no Anthropic-specific code for that to work."""
    converse = ConverseProvider(output={"supplier": "ACME", "total": 121.0})
    try:
        ir = benchmark("bedrock", out={"supplier": "string", "total": "float"},
                       model_override="us.anthropic.claude-haiku-4-5-20251001-v1:0")
        result = await run(ir, workspace, bedrock_adapter(ir, workspace, converse))
        assert result["benchmark_score"] == 1.0

        payload = converse.payload
        assert "/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse" in converse.requests[0]["path"]
        assert payload["system"] == [{"text": "Answer with a JSON object matching the required schema."}]
        tool = payload["toolConfig"]["tools"][0]["toolSpec"]
        assert tool["inputSchema"]["json"]["required"] == ["supplier", "total"]
        assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": tool["name"]}}
        assert "response_format" not in payload
    finally:
        converse.close()


@needs_converse
async def test_a_converse_truncation_is_explained_like_any_other(workspace):
    """Converse reports stopReason max_tokens; llm_client maps it to finish_reason length."""
    converse = ConverseProvider(stop_reason="max_tokens", output={"supplier": "ACME", "total": 121.0})
    try:
        ir = benchmark("bedrock", out={"supplier": "string", "total": "float"},
                       model_override="us.anthropic.claude-sonnet-4-6")
        (only,) = (await run(ir, workspace, bedrock_adapter(ir, workspace, converse)))["results"]
        assert only["status"] == "execution_error"
        assert "max_tokens" in only["error"]["message"]
    finally:
        converse.close()


async def test_a_non_claude_bedrock_model_still_uses_chat_completions(workspace, provider):
    ir = benchmark("bedrock", out={"supplier": "string", "total": "float"}, model_override="openai.gpt-oss-120b")
    await run(ir, workspace, providers.for_system(ir, workspace, env={
        "BEDROCK_BASE_URL": provider.base_url, "AWS_BEARER_TOKEN_BEDROCK": "k"}))
    assert provider.requests[0]["path"].endswith("/chat/completions")
    assert provider.payload["response_format"]["type"] == "json_schema"


def test_claude_on_bedrock_without_converse_support_fails_at_setup(tmp_path):
    """Better a named setup error than one mystery execution_error per example."""

    class WithoutConverse:  # an llm_client whose ProviderProfile has no Converse member
        pass

    ir = benchmark("bedrock", model_override="us.anthropic.claude-sonnet-4-6")
    with pytest.raises(BenchyError) as exc:
        providers.OpenAIChat(
            ir, tmp_path,
            {"AWS_BEARER_TOKEN_BEDROCK": "k", "AWS_REGION": "us-east-1"},
            call=None, profiles=WithoutConverse(),
        )
    assert exc.value.code == "adapter_not_bound"
    assert "Converse" in exc.value.message
    assert "llm-client#4" in exc.value.message
