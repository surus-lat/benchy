"""Translation benchmark module.

This module contains everything for the translation task:
- Prefect task entry point (run.py)
- Dataset implementations (datasets/)
- Metrics calculation (metrics.py)

Import the main entry point:
    from src.tasks.translation import run_translation
"""

from .run import run_translation

__all__ = ["run_translation"]

