"""Structured data extraction benchmark module.

This module contains everything for the structured extraction task:
- Prefect task entry point (run.py)
- Subtask implementations (tasks/)
- Metrics calculation (metrics/)
- Dataset utilities (utils/)

Import the main entry point:
    from src.tasks.structured import run_structured_extraction
"""

from .run import run_structured_extraction

__all__ = ["run_structured_extraction"]

