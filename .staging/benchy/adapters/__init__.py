"""System adapters for the new benchy engine.

Bare metal: functions that build Systems.
"""

from .http import http_system
from .openai import openai_system

__all__ = ["http_system", "openai_system"]
