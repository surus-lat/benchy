"""Interfaces for AI system evaluation.

Interfaces handle communication with different AI systems:
- ChatCompletionsInterface: OpenAI-compatible chat APIs (vLLM, OpenAI, etc.)
- HTTPInterface: Custom HTTP endpoints
- SurusInterface: SURUS AI /extract endpoint

Interfaces are provider-agnostic - they work with connection_info dicts
that contain base_url, api_key, etc. The pipeline resolves provider
configuration to connection_info before passing to interfaces.
"""

from .chat_completions import ChatCompletionsInterface
from .http_interface import HTTPInterface
from .surus_interface import SurusInterface

# Backward compatibility alias
from .llm_interface import LLMInterface

__all__ = [
    "ChatCompletionsInterface",
    "HTTPInterface",
    "SurusInterface",
    "LLMInterface",  # Deprecated, use ChatCompletionsInterface
]

