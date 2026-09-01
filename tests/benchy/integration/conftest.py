"""Integration suite — the merge gate for the five parallel worktrees.

Each worktree tests its own module against fakes. That proves decoupling but
proves nothing about the seams. These tests only exercise seams: real
`benchy.scoring` against real `benchy.task` against real `benchy.system`
against real `benchy.data`, composed by the real `benchy.benchmark`.

Until all five modules are merged, the suite skips with a message naming what
is missing. After the merge it must be green — that is the definition of
Round 2 done.
"""

from __future__ import annotations

import importlib

import pytest

REQUIRED = ("benchy.scoring", "benchy.system", "benchy.data", "benchy.task", "benchy.benchmark")


def _missing() -> list[str]:
    out = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception:
            out.append(name)
    return out


def pytest_collection_modifyitems(config, items):
    missing = _missing()
    if not missing:
        return
    mark = pytest.mark.skip(reason=f"integration gate: not merged yet, missing {', '.join(missing)}")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(mark)


@pytest.fixture(scope="session")
def mods():
    missing = _missing()
    if missing:
        pytest.skip(f"missing {missing}")
    return {n.split(".")[-1]: importlib.import_module(n) for n in REQUIRED}
