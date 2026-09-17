"""EchoSystem — the dependency-free test double every other module builds on.

Built first, per the build brief: siblings import `benchy.system.EchoSystem`
directly and also reach it through `load("echo:")` / `load("mock:")`.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from benchy.core import (
    AudioPart,
    Capabilities,
    LoadError,
    Message,
    Request,
    Response,
    SystemFailure,
    TextPart,
)
from benchy.system import EchoSystem, load


def _req(text: str = "hola") -> Request:
    return Request(messages=(Message.text("user", text),))


class TestConstructionAndIsSystem:
    def test_load_echo_returns_a_system(self):
        system = load("echo:")
        assert isinstance(system, EchoSystem)
        assert system.url == "echo:"

    def test_load_mock_alias_returns_echo_system(self):
        system = load("mock:")
        assert isinstance(system, EchoSystem)

    def test_default_capabilities_are_plain(self):
        system = load("echo:")
        assert system.capabilities == Capabilities()

    def test_capabilities_are_configurable(self):
        caps = Capabilities(structured_output=True, audio_in=True)
        system = load("echo:", capabilities=caps)
        assert system.capabilities is caps


class TestDefaultEchoBehaviour:
    @pytest.mark.asyncio
    async def test_default_echoes_back_the_request_text(self):
        system = EchoSystem()
        response = await system.invoke(_req("hello world"))
        assert response.text == "hello world"
        assert response.ok

    @pytest.mark.asyncio
    async def test_default_joins_multiple_text_parts(self):
        system = EchoSystem()
        request = Request(
            messages=(
                Message(role="user", parts=(TextPart("a"), TextPart("b"))),
            )
        )
        response = await system.invoke(request)
        assert response.text == "a\nb"

    @pytest.mark.asyncio
    async def test_default_with_no_text_parts_is_empty_string(self):
        system = EchoSystem(capabilities=Capabilities(audio_in=True))
        request = Request(messages=(Message(role="user", parts=(AudioPart(path="/tmp/a.wav"),)),))
        response = await system.invoke(request)
        assert response.text == ""


class TestCannedText:
    @pytest.mark.asyncio
    async def test_canned_text_ignores_request_content(self):
        system = EchoSystem(text="always this")
        r1 = await system.invoke(_req("one"))
        r2 = await system.invoke(_req("two"))
        assert r1.text == "always this"
        assert r2.text == "always this"


class TestCannedData:
    @pytest.mark.asyncio
    async def test_canned_structured_data(self):
        system = EchoSystem(data={"total": 42})
        response = await system.invoke(_req())
        assert response.data == {"total": 42}

    @pytest.mark.asyncio
    async def test_text_and_data_can_be_combined(self):
        system = EchoSystem(text="ok", data={"total": 42})
        response = await system.invoke(_req())
        assert response.text == "ok"
        assert response.data == {"total": 42}


class TestScriptedSequence:
    @pytest.mark.asyncio
    async def test_sequence_of_plain_strings(self):
        system = EchoSystem(responses=["first", "second"])
        r1 = await system.invoke(_req())
        r2 = await system.invoke(_req())
        assert (r1.text, r2.text) == ("first", "second")

    @pytest.mark.asyncio
    async def test_sequence_of_response_objects(self):
        system = EchoSystem(responses=[Response(text="x", data={"a": 1})])
        r1 = await system.invoke(_req())
        assert r1.text == "x"
        assert r1.data == {"a": 1}

    @pytest.mark.asyncio
    async def test_sequence_of_dicts_becomes_structured_data(self):
        system = EchoSystem(responses=[{"a": 1}])
        r1 = await system.invoke(_req())
        assert r1.data == {"a": 1}

    @pytest.mark.asyncio
    async def test_sequence_cycles_once_exhausted(self):
        # A scripted run may be re-run (compare() grades several systems,
        # as_loss() re-runs the same benchmark) so it cycles, not repeats-last.
        system = EchoSystem(responses=["first", "second"])
        texts = [(await system.invoke(_req())).text for _ in range(5)]
        assert texts == ["first", "second", "first", "second", "first"]

    @pytest.mark.asyncio
    async def test_sequence_can_mix_exceptions_to_raise_mid_script(self):
        system = EchoSystem(responses=["ok", SystemFailure("boom")])
        r1 = await system.invoke(_req())
        assert r1.text == "ok"
        with pytest.raises(SystemFailure, match="boom"):
            await system.invoke(_req())


class TestInducedError:
    @pytest.mark.asyncio
    async def test_error_sets_response_error_and_ok_is_false(self):
        system = EchoSystem(error="rate limited")
        response = await system.invoke(_req())
        assert response.error == "rate limited"
        assert not response.ok

    @pytest.mark.asyncio
    async def test_error_on_matches_sample_id_in_request_meta(self):
        system = EchoSystem(error_on=["2"], data={"name": "Ana"})
        ok_request = Request(messages=(Message.text("user", "hi"),), meta={"sample_id": "1"})
        bad_request = Request(messages=(Message.text("user", "hi"),), meta={"sample_id": "2"})

        ok_response = await system.invoke(ok_request)
        bad_response = await system.invoke(bad_request)

        assert ok_response.ok
        assert ok_response.data == {"name": "Ana"}
        assert not bad_response.ok
        assert bad_response.error

    @pytest.mark.asyncio
    async def test_error_on_falls_back_to_scanning_meta_values(self):
        system = EchoSystem(error_on=["7"])
        request = Request(messages=(Message.text("user", "hi"),), meta={"other_key": "7"})
        response = await system.invoke(request)
        assert not response.ok

    @pytest.mark.asyncio
    async def test_error_on_no_match_falls_through_to_normal_behaviour(self):
        system = EchoSystem(error_on=["999"], text="fine")
        response = await system.invoke(_req())
        assert response.ok
        assert response.text == "fine"

    @pytest.mark.asyncio
    async def test_raises_option_raises_instead_of_returning(self):
        system = EchoSystem(raises=SystemFailure("dead"))
        with pytest.raises(SystemFailure, match="dead"):
            await system.invoke(_req())

    @pytest.mark.asyncio
    async def test_raises_accepts_an_exception_type(self):
        system = EchoSystem(raises=SystemFailure)
        with pytest.raises(SystemFailure):
            await system.invoke(_req())


class TestInducedLatency:
    @pytest.mark.asyncio
    async def test_latency_ms_delays_and_is_reported(self):
        system = EchoSystem(text="slow", latency_ms=20)
        start = time.perf_counter()
        response = await system.invoke(_req())
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms >= 15  # allow scheduler slack
        assert response.latency_ms is not None
        assert response.latency_ms >= 15


class TestRequestRecording:
    @pytest.mark.asyncio
    async def test_requests_records_every_request_in_order(self):
        system = EchoSystem()
        await system.invoke(_req("one"))
        await system.invoke(_req("two"))
        assert len(system.requests) == 2
        assert system.requests[0].messages[0].parts[0].text == "one"
        assert system.requests[1].messages[0].parts[0].text == "two"

    @pytest.mark.asyncio
    async def test_concurrent_invokes_all_get_recorded(self):
        system = EchoSystem(text="x")
        await asyncio.gather(*(system.invoke(_req(str(i))) for i in range(5)))
        assert len(system.requests) == 5

    @pytest.mark.asyncio
    async def test_received_is_an_alias_for_requests(self):
        system = EchoSystem()
        await system.invoke(_req())
        assert system.received is system.requests


class TestResponder:
    @pytest.mark.asyncio
    async def test_responder_callable_computes_from_the_request(self):
        def responder(request: Request) -> Response:
            text = request.messages[0].parts[0].text
            return Response(text=text.upper())

        system = EchoSystem(responder=responder)
        response = await system.invoke(_req("shout"))
        assert response.text == "SHOUT"

    @pytest.mark.asyncio
    async def test_responder_can_be_async(self):
        async def responder(request: Request) -> Response:
            return Response(text="async-ok")

        system = EchoSystem(responder=responder)
        response = await system.invoke(_req())
        assert response.text == "async-ok"


class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_is_a_noop_and_marks_closed(self):
        system = EchoSystem()
        await system.aclose()
        assert system.closed is True


class TestLoadErrors:
    def test_echo_url_with_a_name_suffix_is_accepted(self):
        system = load("echo:my-fixture")
        assert system.url == "echo:my-fixture"
