"""FLORES dataset utilities.

This module contains preprocessing logic for FLORES+ translation data.
The task implementation is in src/tasks/translation/flores.py (handler-based).
"""

from .download import download_and_preprocess_flores

__all__ = ["download_and_preprocess_flores"]
