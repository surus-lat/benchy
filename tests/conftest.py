"""Shared pytest configuration for Benchy tests."""

import asyncio
import inspect
import pytest

# Import all fixtures so they're available to all tests
from tests.fixtures import *  # noqa: F401, F403


def pytest_configure(config):
    """Register markers used across test suites."""
    config.addinivalue_line("markers", "asyncio: mark test as asyncio-compatible")
    config.addinivalue_line("markers", "anyio: mark test to run with anyio backend")


def pytest_collection_modifyitems(config, items):
    """Support asyncio-marked tests even when pytest-asyncio isn't installed."""
    if config.pluginmanager.hasplugin("asyncio"):
        return

    for item in items:
        if item.get_closest_marker("asyncio") and not item.get_closest_marker("anyio"):
            item.add_marker(pytest.mark.anyio)


def pytest_pyfunc_call(pyfuncitem):
    """Run async tests when no async plugin is available.

    This keeps the suite runnable in environments that have neither
    `pytest-asyncio` nor anyio-based async test auto-handling enabled.
    """
    if pyfuncitem.config.pluginmanager.hasplugin("asyncio"):
        return None

    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    funcargs = {
        arg: pyfuncitem.funcargs[arg]
        for arg in pyfuncitem._fixtureinfo.argnames
        if arg in pyfuncitem.funcargs
    }
    asyncio.run(test_func(**funcargs))
    return True
