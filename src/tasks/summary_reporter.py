"""Shared summary writer for task group results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

MetricSpec = Tuple[str, str, str]
logger = logging.getLogger(__name__)


def _format_metric(value: Any, fmt: str) -> str:
    if value is None:
        value = 0
    try:
        if fmt == "d":
            return f"{int(value)}"
        return format(value, fmt)
    except (ValueError, TypeError):
        return str(value)


def write_group_summary(
    *,
    output_dir: Path,
    model_name: str,
    subtasks: Iterable[str],
    aggregated_metrics: Dict[str, Any],
    subtask_metrics: Dict[str, Dict[str, Any]],
    title: str,
    aggregated_fields: List[MetricSpec],
    per_subtask_fields: List[MetricSpec],
) -> None:
    """Write JSON and text summaries for a grouped task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_")

    summary_file = output_dir / f"{safe_name}_{timestamp}_summary.json"
    with summary_file.open("w") as f:
        json.dump(
            {
                "model": model_name,
                "timestamp": timestamp,
                "subtasks": list(subtasks),
                "aggregated_metrics": aggregated_metrics,
                "per_subtask_metrics": subtask_metrics,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved summary to {summary_file}")

    text_file = output_dir / f"{safe_name}_{timestamp}_summary.txt"
    with text_file.open("w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Subtasks: {', '.join(subtasks)}\n\n")

        f.write("AGGREGATED METRICS\n")
        f.write("-" * 40 + "\n")
        for key, label, fmt in aggregated_fields:
            value = aggregated_metrics.get(key, 0)
            f.write(f"{label}: {_format_metric(value, fmt)}\n")
        f.write("\n")

        f.write("PER-SUBTASK BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for name, metrics in subtask_metrics.items():
            f.write(f"\n{name.upper()}:\n")
            for key, label, fmt in per_subtask_fields:
                value = metrics.get(key, 0)
                f.write(f"  {label}: {_format_metric(value, fmt)}\n")
        f.write("=" * 60 + "\n")
    logger.info(f"Saved text summary to {text_file}")
