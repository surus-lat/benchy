"""Image extraction benchmark module.

This module uses the new handler-based task system.
Tasks are discovered automatically from the .py files in this directory.
"""

from .facturas import Facturas

__all__ = ["Facturas"]
