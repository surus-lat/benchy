"""Interfaces for AI system evaluation.

Interfaces handle communication with different AI systems:
- OpenAIInterface: OpenAI-compatible chat/completions APIs (vLLM, OpenAI, etc.)
- HTTPInterface: Custom HTTP endpoints
- SurusInterface: SURUS AI /extract endpoint
- SurusOCRInterface: SURUS AI /ocr endpoint (image extraction)

Interfaces are provider-agnostic - they work with connection_info dicts
that contain base_url, api_key, etc. The pipeline resolves provider
configuration to connection_info before passing to interfaces.
"""

from .http_interface import HTTPInterface
from .openai_interface import OpenAIInterface
from .surus.surus_interface import SurusInterface
from .surus.surus_ocr_interface import SurusOCRInterface
from .surus.surus_factura_interface import SurusFacturaInterface

__all__ = [
    "OpenAIInterface",
    "HTTPInterface",
    "SurusInterface",
    "SurusOCRInterface",
    "SurusFacturaInterface",
]
