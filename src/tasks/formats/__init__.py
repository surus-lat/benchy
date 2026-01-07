"""Format handlers for common task patterns.

This module provides reusable format handlers that encapsulate common evaluation
patterns (multiple choice, structured output, freeform text) with sensible defaults
and minimal boilerplate for task implementations.

Usage:
    from src.tasks.formats import MultipleChoiceHandler
    
    class MyTask(MultipleChoiceHandler):
        dataset = "org/my-dataset"
        labels = {0: "No", 1: "Yes"}
"""

from .base import BaseHandler, FormatHandler
from .multiple_choice import MultipleChoiceHandler
from .structured import StructuredHandler
from .freeform import FreeformHandler
from .multimodal_structured import MultimodalStructuredHandler

__all__ = [
    "BaseHandler",
    "FormatHandler",
    "MultipleChoiceHandler",
    "StructuredHandler",
    "FreeformHandler",
    "MultimodalStructuredHandler",
]

