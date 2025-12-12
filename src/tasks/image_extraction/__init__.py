"""Image extraction benchmark module.

This module evaluates model capabilities to extract structured data
from images following a JSON schema.

Import the main entry point:
    from src.tasks.image_extraction import run_image_extraction
"""

from .run import run_image_extraction

__all__ = ["run_image_extraction"]


