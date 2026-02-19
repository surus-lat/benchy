"""Google interfaces for benchy."""

from .google_interface import GoogleInterface
from .google_remove_background_interface import GoogleRemoveBackgroundInterface

__all__ = [
    "GoogleInterface",
    "GoogleRemoveBackgroundInterface",  # Kept for backward compatibility
]
