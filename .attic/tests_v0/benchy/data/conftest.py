"""Local test config for benchy.data — keeps the default suite hermetic.

Registered here (not in the shared ``pyproject.toml``) so this worktree does
not need to touch a file other parallel worktrees may also be editing.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires live network access; skipped unless RUN_INTEGRATION=1",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip_integration = pytest.mark.skip(
        reason="integration test requires network; set RUN_INTEGRATION=1 to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
