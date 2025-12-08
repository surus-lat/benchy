"""Interfaces for AI system evaluation."""

from .llm_interface import LLMInterface
from .http_interface import HTTPInterface
from .surus_interface import SurusInterface

__all__ = ["LLMInterface", "HTTPInterface", "SurusInterface"]

