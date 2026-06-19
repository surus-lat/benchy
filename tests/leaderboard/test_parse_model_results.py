import json
import pytest
from pathlib import Path


def test_reads_model_info_from_run_outcome(tmp_path):
    """When run_config.yaml is absent, extract_model_info_from_config should fall back to run_outcome.json."""
    from src.leaderboard.functions.parse_model_results import extract_model_info_from_config
    outcome = {"model": "deepseek-ai/DeepSeek-V4-Pro", "run_id": "test-run"}
    (tmp_path / "run_outcome.json").write_text(json.dumps(outcome))

    info = extract_model_info_from_config(tmp_path)

    assert info["full_model_name"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert info["publisher"] == "deepseek-ai"
    assert info["model_name"] == "DeepSeek-V4-Pro"


def test_falls_back_gracefully_when_no_files(tmp_path):
    """When neither config file exists, model name defaults to directory name."""
    from src.leaderboard.functions.parse_model_results import extract_model_info_from_config
    info = extract_model_info_from_config(tmp_path)
    assert info["publisher"] == "unknown"
    assert info["model_name"] == tmp_path.name


def test_load_latest_summary_ignores_performance_summary(tmp_path):
    """_load_latest_summary must not pick up *_performance_summary.json files."""
    from src.leaderboard.functions.parse_model_results import _load_latest_summary
    perf = {"model": "x", "task": "y", "timestamp": "z", "performance_summary": {}}
    (tmp_path / "model_20260101_performance_summary.json").write_text(json.dumps(perf))

    result = _load_latest_summary(tmp_path)

    assert result is None, "Should return None when only _performance_summary.json exists"


def test_load_latest_summary_picks_correct_summary(tmp_path):
    """_load_latest_summary returns a proper *_summary.json (not *_performance_summary.json)."""
    from src.leaderboard.functions.parse_model_results import _load_latest_summary
    perf = {"model": "x", "task": "y", "timestamp": "z", "performance_summary": {}}
    proper = {"model": "x", "task": "y", "timestamp": "z", "per_subtask_metrics": {"t1": {"acc": 0.8}}}
    (tmp_path / "model_20260101_performance_summary.json").write_text(json.dumps(perf))
    (tmp_path / "model_20260101_summary.json").write_text(json.dumps(proper))

    result = _load_latest_summary(tmp_path)

    assert result is not None
    assert "per_subtask_metrics" in result
