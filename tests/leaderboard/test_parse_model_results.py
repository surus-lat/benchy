import json
import pytest
from pathlib import Path
from src.leaderboard.functions.parse_model_results import (
    extract_model_info_from_config,
    _load_latest_summary,
    structured_extraction_results_processor,
)


def test_reads_model_info_from_run_outcome(tmp_path):
    """When run_config.yaml is absent, should fall back to run_outcome.json."""
    outcome = {"model": "deepseek-ai/DeepSeek-V4-Pro", "run_id": "test-run"}
    (tmp_path / "run_outcome.json").write_text(json.dumps(outcome))

    info = extract_model_info_from_config(tmp_path)

    assert info["full_model_name"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert info["publisher"] == "deepseek-ai"
    assert info["model_name"] == "DeepSeek-V4-Pro"


def test_falls_back_gracefully_when_no_files(tmp_path):
    """When neither config file exists, model name defaults to directory name."""
    info = extract_model_info_from_config(tmp_path)
    assert info["publisher"] == "unknown"
    assert info["model_name"] == tmp_path.name


def test_load_latest_summary_ignores_performance_summary(tmp_path):
    """_load_latest_summary must not pick up *_performance_summary.json files."""
    perf = {"model": "x", "task": "y", "timestamp": "z", "performance_summary": {}}
    (tmp_path / "model_20260101_performance_summary.json").write_text(json.dumps(perf))

    result = _load_latest_summary(tmp_path)

    assert result is None, "Should return None when only _performance_summary.json exists"


def test_load_latest_summary_picks_correct_summary(tmp_path):
    """_load_latest_summary returns a proper *_summary.json (not *_performance_summary.json)."""
    perf = {"model": "x", "task": "y", "timestamp": "z", "performance_summary": {}}
    proper = {"model": "x", "task": "y", "timestamp": "z", "per_subtask_metrics": {"t1": {"acc": 0.8}}}
    (tmp_path / "model_20260101_performance_summary.json").write_text(json.dumps(perf))
    (tmp_path / "model_20260101_summary.json").write_text(json.dumps(proper))

    result = _load_latest_summary(tmp_path)

    assert result is not None
    assert "per_subtask_metrics" in result


# ---------------------------------------------------------------------------
# Known issue — post-restructure
# ---------------------------------------------------------------------------
# Modern benchmark runs (April 2026+) write structured extraction output to
#   model_dir/_adhoc_structured_<hash>/main/*_metrics.json
# instead of the old
#   model_dir/structured_extraction/*_summary.json
#
# The processor currently derives task_dir from load_task_config(), which
# resolves to model_dir/structured_extraction/ and returns None when that
# path is absent.  The fix (glob for _adhoc_structured_*/main as fallback)
# is deferred until the repo restructure settles.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "structured_extraction_results_processor does not yet fall back to "
        "_adhoc_structured_<hash>/main/ — fix deferred until repo restructure"
    ),
)
def test_structured_extraction_processor_finds_adhoc_dirs(tmp_path):
    """Processor must find modern _adhoc_structured_<hash>/main/ output dirs.

    Real example: outputs/benchmark_outputs/20260414_164304_LIMITED/Kimi-K2.5/
      _adhoc_structured_c3167986/
        main/
          moonshotai_Kimi-K2.5_20260414_164356_metrics.json   ← flat metrics dict
          moonshotai_Kimi-K2.5_20260414_164356_performance_summary.json
          moonshotai_Kimi-K2.5_20260414_164356_report.txt
        task_status.json

    The processor already knows how to read *_metrics.json (lines 771-881 of
    parse_model_results.py); the only missing piece is finding the directory.
    """
    model_dir = tmp_path / "Kimi-K2.5"
    model_dir.mkdir()

    adhoc_main = model_dir / "_adhoc_structured_c3167986" / "main"
    adhoc_main.mkdir(parents=True)

    metrics_payload = {
        "model": "moonshotai/Kimi-K2.5",
        "task": "structured",
        "timestamp": "20260414_164356",
        "metrics": {
            "extraction_quality_score": 0.6815,
            "overall_extraction_quality_score": 0.6815,
            "schema_validity_rate": 1.0,
            "hallucination_rate": 0.0,
            "field_f1_partial": 0.545,
            "composite_score_stats": {"mean": 0.7185, "stdev": 0.303},
        },
    }
    (adhoc_main / "moonshotai_Kimi-K2.5_20260414_164356_metrics.json").write_text(
        json.dumps(metrics_payload)
    )

    task_config = {
        "name": "structured_extraction",
        "category_score_key": "structured_extraction",
        "output_prefix": "structured_extraction",
    }

    result = structured_extraction_results_processor(model_dir, "Kimi-K2.5", task_config)

    assert result is not None, (
        "Processor returned None — it could not find _adhoc_structured_*/main/. "
        "Fix: add glob fallback after the task_dir.exists() check."
    )
    eqs = result.get("overall_score", 0)
    assert eqs > 0, f"EQS should be ~0.68, got {eqs}"
    assert "structured_extraction" in result.get("category_scores", {})
