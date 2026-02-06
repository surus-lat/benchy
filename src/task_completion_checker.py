"""Task completion/status helpers for resumable runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .outcome import COMPLETED_TASK_STATUSES, TASK_STATUS_PENDING

logger = logging.getLogger(__name__)

TASK_STATUS_FILENAME = "task_status.json"


def _task_dir_for_ref(model_output_path: str, task_ref: str) -> Path:
    """Map task refs (including subtasks) to task output directory."""
    root_task = (task_ref or "").split(".", 1)[0]
    return Path(model_output_path) / root_task


def write_task_status(
    task_output_path: str | Path,
    *,
    task_name: str,
    status: str,
    reason: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    subtasks: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write machine-readable status for a task group execution."""
    output_dir = Path(task_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "task": task_name,
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "reason": reason,
        "summary": summary or {},
        "subtasks": subtasks or {},
        "details": details or {},
    }
    status_path = output_dir / TASK_STATUS_FILENAME
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Task status written: {status_path} ({status})")
    return payload


def read_task_status(task_output_path: str | Path) -> Optional[Dict[str, Any]]:
    """Read status payload from a task output directory."""
    status_path = Path(task_output_path) / TASK_STATUS_FILENAME
    if not status_path.exists():
        return None
    try:
        with status_path.open("r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning(f"Failed to read task status from {status_path}: {exc}")
    return None


class TaskCompletionChecker:
    """Checks whether requested tasks are completed based on task status files."""

    def __init__(self, output_path: str, run_id: str, model_name: str):
        self.output_path = output_path
        self.run_id = run_id
        self.model_name = model_name
        self.model_output_path = f"{output_path}/{run_id}/{model_name.split('/')[-1]}"

    def get_task_records(self, requested_tasks: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return completion/status records keyed by requested task refs."""
        records: Dict[str, Dict[str, Any]] = {}
        for task_ref in requested_tasks:
            task_output_dir = _task_dir_for_ref(self.model_output_path, task_ref)
            payload = read_task_status(task_output_dir) or {}
            status = payload.get("status") or TASK_STATUS_PENDING
            records[task_ref] = {
                "task": task_ref,
                "status": status,
                "completed": status in COMPLETED_TASK_STATUSES,
                "task_output_dir": str(task_output_dir),
                "reason": payload.get("reason"),
                "summary": payload.get("summary") or {},
                "subtasks": payload.get("subtasks") or {},
                "updated_at": payload.get("updated_at"),
            }
        return records

    def log_completion_summary(self, records: Dict[str, Dict[str, Any]]) -> None:
        """Log a summary of completion status."""
        completed = [task for task, rec in records.items() if rec.get("completed")]
        pending = [task for task, rec in records.items() if not rec.get("completed")]
        status_counts: Dict[str, int] = {}
        for rec in records.values():
            status = str(rec.get("status") or TASK_STATUS_PENDING)
            status_counts[status] = status_counts.get(status, 0) + 1

        logger.info("=" * 60)
        logger.info("TASK COMPLETION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output Path: {self.model_output_path}")
        logger.info("")

        if completed:
            logger.info(f"✅ COMPLETED TASKS ({len(completed)}):")
            for task in completed:
                logger.info(f"  - {task}")
        else:
            logger.info("✅ COMPLETED TASKS: None")

        if pending:
            logger.info(f"⏳ PENDING TASKS ({len(pending)}):")
            for task in pending:
                logger.info(f"  - {task}")
        else:
            logger.info("⏳ PENDING TASKS: None")

        logger.info("")
        logger.info("Status counts:")
        for status_name in sorted(status_counts):
            logger.info(f"  - {status_name}: {status_counts[status_name]}")

        logger.info("=" * 60)
