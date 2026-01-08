"""Translation benchmark module.

This module uses the handler-based task system with TaskGroupSpec
for shared COMET model lifecycle.

Tasks are discovered automatically from the .py files in this directory.
The COMET model is loaded once via setup and shared across all subtasks.
"""

from .opus import Opus
from .flores import Flores

__all__ = ["Opus", "Flores"]
