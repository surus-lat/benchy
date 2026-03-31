from src.pipeline import _summarize_single_task_metrics


def test_summarize_single_task_metrics_preserves_performance_summary_dict():
    summarized = _summarize_single_task_metrics(
        {
            "score": 0.9,
            "performance_summary": {
                "status": "ok",
                "primary_metric": "document_extraction_score",
                "quartiles": {"q1": 0.2, "median": 0.5, "q3": 0.8},
            },
        }
    )

    assert summarized["score"] == 0.9
    assert summarized["performance_summary"]["primary_metric"] == "document_extraction_score"
