"""Structured extraction benchmark module.

This module uses the new handler-based task system.
Tasks are discovered automatically from the .py files in this directory.
"""

from .chat_extract import ChatExtract
from .paraloq import Paraloq
from .email_extract import EmailExtract

__all__ = ["ChatExtract", "Paraloq", "EmailExtract"]
