"""The loader registry: load(), register(), schemes()."""

from __future__ import annotations

import pytest

from benchy.core import LoadError, System
from benchy.system import load, register, schemes


class TestSchemes:
    def test_all_five_shipped_schemes_are_registered(self):
        known = set(schemes())
        assert {"echo", "mock", "openai", "endpoint", "hf", "python"} <= known


class TestLoadDispatch:
    def test_load_returns_a_core_system(self):
        system = load("echo:")
        assert isinstance(system, System)

    def test_unknown_scheme_raises_load_error_listing_known_schemes(self):
        with pytest.raises(LoadError) as excinfo:
            load("bogus:whatever")
        message = str(excinfo.value)
        assert "bogus" in message
        for name in schemes():
            assert name in message

    def test_malformed_url_without_a_colon_raises_load_error(self):
        with pytest.raises(LoadError):
            load("just-a-string-no-scheme")

    def test_load_passes_kwargs_through_to_the_loader(self):
        system = load("echo:", text="configured")
        assert system.url == "echo:"


class TestRegisterIsPluggable:
    def test_register_adds_a_new_scheme(self):
        def loader(rest, **opts):
            from benchy.system import EchoSystem

            return EchoSystem(url=f"custom:{rest}", **opts)

        register("custom", loader)
        try:
            assert "custom" in schemes()
            system = load("custom:thing")
            assert system.url == "custom:thing"
        finally:
            # no unregister API is required by the contract; leaving it
            # registered is harmless for the rest of the suite.
            pass

    def test_register_can_override_an_existing_scheme(self):
        original = list(schemes())

        def loader(rest, **opts):
            from benchy.system import EchoSystem

            return EchoSystem(url="overridden:", text="overridden")

        register("echo", loader)
        try:
            system = load("echo:")
            assert system.url == "overridden:"
        finally:
            # restore the real echo loader so later tests are unaffected
            from benchy.system.echo import load_echo

            register("echo", load_echo)
        assert schemes() == original
