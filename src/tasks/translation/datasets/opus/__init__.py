"""OPUS dataset utilities.

This module contains preprocessing logic for OPUS-100 translation data.
The task implementation is in src/tasks/translation/opus.py (handler-based).
"""

from .download import download_and_preprocess_opus

__all__ = ["download_and_preprocess_opus"]
