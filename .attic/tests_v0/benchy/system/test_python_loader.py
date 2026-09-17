"""`python:` — wrap an arbitrary Python callable/class as a System.

This is how a node, a workflow, or a LangGraph/DSPy agent enters benchy:
`python:<path-or-dotted-module>[:<attr>]`. Accepts an async function, a sync
function, or a class with an `invoke` method; a bare `str` return is wrapped
into a `Response`.
"""

from __future__ import annotations

import sys

import pytest

from benchy.core import Capabilities, LoadError, Message, Request, Response, System
from benchy.system import load


def _req(text: str = "hi") -> Request:
    return Request(messages=(Message.text("user", text),))


class TestPathBasedLoading:
    def test_async_function_returning_response(self, tmp_path):
        path = tmp_path / "agent_async.py"
        path.write_text(
            "from benchy.core import Response\n"
            "async def run(request):\n"
            "    return Response(text='async-ran')\n"
        )
        system = load(f"python:{path}:run")
        assert isinstance(system, System)

    @pytest.mark.asyncio
    async def test_async_function_actually_invokes(self, tmp_path):
        path = tmp_path / "agent_async2.py"
        path.write_text(
            "from benchy.core import Response\n"
            "async def run(request):\n"
            "    return Response(text='async-ran')\n"
        )
        system = load(f"python:{path}:run")
        response = await system.invoke(_req())
        assert response.text == "async-ran"

    @pytest.mark.asyncio
    async def test_sync_function_returning_response(self, tmp_path):
        path = tmp_path / "agent_sync.py"
        path.write_text(
            "from benchy.core import Response\n"
            "def run(request):\n"
            "    return Response(text='sync-ran')\n"
        )
        system = load(f"python:{path}:run")
        response = await system.invoke(_req())
        assert response.text == "sync-ran"

    @pytest.mark.asyncio
    async def test_bare_str_return_is_wrapped_into_a_response(self, tmp_path):
        path = tmp_path / "agent_str.py"
        path.write_text("def run(request):\n    return 'plain string'\n")
        system = load(f"python:{path}:run")
        response = await system.invoke(_req())
        assert isinstance(response, Response)
        assert response.text == "plain string"

    @pytest.mark.asyncio
    async def test_async_function_with_bare_str_return(self, tmp_path):
        path = tmp_path / "agent_str_async.py"
        path.write_text("async def run(request):\n    return 'plain'\n")
        system = load(f"python:{path}:run")
        response = await system.invoke(_req())
        assert response.text == "plain"

    @pytest.mark.asyncio
    async def test_class_with_invoke_method_sync(self, tmp_path):
        path = tmp_path / "agent_class.py"
        path.write_text(
            "from benchy.core import Response\n"
            "class Agent:\n"
            "    def invoke(self, request):\n"
            "        return Response(text='class-ran')\n"
        )
        system = load(f"python:{path}:Agent")
        response = await system.invoke(_req())
        assert response.text == "class-ran"

    @pytest.mark.asyncio
    async def test_class_with_invoke_method_async(self, tmp_path):
        path = tmp_path / "agent_class_async.py"
        path.write_text(
            "from benchy.core import Response\n"
            "class Agent:\n"
            "    async def invoke(self, request):\n"
            "        return Response(text='async-class-ran')\n"
        )
        system = load(f"python:{path}:Agent")
        response = await system.invoke(_req())
        assert response.text == "async-class-ran"

    @pytest.mark.asyncio
    async def test_class_constructor_receives_load_kwargs(self, tmp_path):
        path = tmp_path / "agent_kwargs.py"
        path.write_text(
            "from benchy.core import Response\n"
            "class Agent:\n"
            "    def __init__(self, greeting='hi'):\n"
            "        self.greeting = greeting\n"
            "    def invoke(self, request):\n"
            "        return Response(text=self.greeting)\n"
        )
        system = load(f"python:{path}:Agent", greeting="configured")
        response = await system.invoke(_req())
        assert response.text == "configured"

    @pytest.mark.asyncio
    async def test_fully_conforming_system_class_is_used_unchanged(self, tmp_path):
        path = tmp_path / "agent_full_system.py"
        path.write_text(
            "from benchy.core import Capabilities, Response\n"
            "class Agent:\n"
            "    def __init__(self):\n"
            "        self.url = 'python:custom-url'\n"
            "        self.capabilities = Capabilities(audio_in=True)\n"
            "    async def invoke(self, request):\n"
            "        return Response(text='ok')\n"
            "    async def aclose(self):\n"
            "        self.closed = True\n"
        )
        system = load(f"python:{path}:Agent")
        # its own url/capabilities are preserved, not overwritten
        assert system.url == "python:custom-url"
        assert system.capabilities.audio_in is True
         

    def test_default_attribute_lookup_finds_system(self, tmp_path):
        path = tmp_path / "agent_default_system.py"
        path.write_text(
            "from benchy.core import Response\n"
            "class _Impl:\n"
            "    def invoke(self, request):\n"
            "        return Response(text='found-system')\n"
            "system = _Impl()\n"
        )
        system = load(f"python:{path}")
        assert isinstance(system, System)

    def test_default_attribute_lookup_finds_agent_class(self, tmp_path):
        path = tmp_path / "agent_default_agent.py"
        path.write_text(
            "from benchy.core import Response\n"
            "class Agent:\n"
            "    def invoke(self, request):\n"
            "        return Response(text='found-agent')\n"
        )
        system = load(f"python:{path}")
        assert isinstance(system, System)

    def test_no_discoverable_attribute_raises_load_error(self, tmp_path):
        path = tmp_path / "agent_nothing.py"
        path.write_text("x = 1\n")
        with pytest.raises(LoadError):
            load(f"python:{path}")

    def test_missing_file_raises_load_error(self, tmp_path):
        with pytest.raises(LoadError):
            load(f"python:{tmp_path / 'does_not_exist.py'}:run")

    def test_target_that_is_not_callable_raises_load_error(self, tmp_path):
        path = tmp_path / "agent_noncallable.py"
        path.write_text("thing = 42\n")
        with pytest.raises(LoadError):
            load(f"python:{path}:thing")

    def test_capabilities_kwarg_is_honoured_for_a_plain_function(self, tmp_path):
        path = tmp_path / "agent_caps.py"
        path.write_text(
            "from benchy.core import Response\n"
            "def run(request):\n"
            "    return Response(text='x')\n"
        )
        caps = Capabilities(audio_in=True)
        system = load(f"python:{path}:run", capabilities=caps)
        assert system.capabilities is caps


class TestDottedModuleLoading:
    def test_dotted_module_path_on_sys_path(self, tmp_path, monkeypatch):
        (tmp_path / "my_bench_agent.py").write_text(
            "from benchy.core import Response\n"
            "def run(request):\n"
            "    return Response(text='dotted-ok')\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("my_bench_agent", None)
        try:
            system = load("python:my_bench_agent:run")
            assert isinstance(system, System)
        finally:
            sys.modules.pop("my_bench_agent", None)

    def test_unimportable_dotted_module_raises_load_error(self):
        with pytest.raises(LoadError):
            load("python:this.module.does.not.exist:run")


class TestUrlAndEmptyRest:
    def test_empty_rest_raises_load_error(self):
        with pytest.raises(LoadError):
            load("python:")

    def test_url_records_the_original_rest(self, tmp_path):
        path = tmp_path / "agent_url.py"
        path.write_text(
            "from benchy.core import Response\n"
            "def run(request):\n"
            "    return Response(text='x')\n"
        )
        system = load(f"python:{path}:run")
        assert system.url == f"python:{path}:run"
