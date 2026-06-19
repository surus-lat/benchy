import json
import pytest
from pathlib import Path
from src.leaderboard.merge_and_publish import merge_summaries


def test_merge_summaries_new_overwrites_old():
    old = {
        "old-model": {"model_name": "old-model", "publisher": "acme", "categories": {}},
        "shared-model": {"model_name": "shared-model", "publisher": "acme", "categories": {"spanish": {"overall_score": 0.5}}},
    }
    new = {
        "new-model": {"model_name": "new-model", "publisher": "beta", "categories": {}},
        "shared-model": {"model_name": "shared-model", "publisher": "acme", "categories": {"spanish": {"overall_score": 0.9}}},
    }
    merged = merge_summaries(old, new)

    assert "old-model" in merged
    assert "new-model" in merged
    assert merged["shared-model"]["categories"]["spanish"]["overall_score"] == 0.9


def test_merge_summaries_handles_empty_old():
    new = {"model-a": {"model_name": "model-a"}}
    merged = merge_summaries({}, new)
    assert merged == new


def test_merge_summaries_handles_empty_new():
    old = {"model-a": {"model_name": "model-a"}}
    merged = merge_summaries(old, {})
    assert merged == old
