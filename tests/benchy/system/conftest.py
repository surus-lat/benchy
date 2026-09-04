"""Shared fixtures/markers for benchy.system tests."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: hits the network or loads real model weights; skipped by default runs",
    )
