"""Structured extraction benchmark module.

This module uses the new handler-based task system.
Tasks are discovered automatically from the .py files in this directory.
"""

# Export handlers for direct imports (optional, mainly for testing)
from .chat_extract import ChatExtract
from .paraloq import Paraloq

__all__ = ["ChatExtract", "Paraloq"]

