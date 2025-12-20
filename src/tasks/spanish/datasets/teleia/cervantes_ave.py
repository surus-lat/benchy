"""Teleia Cervantes AVE task implementation."""

from typing import Dict

from .base import TeleiaTaskBase
from .preprocessing import process_cervantes


class TeleiaCervantesAveTask(TeleiaTaskBase):
    """Teleia Cervantes AVE task."""
    
    def __init__(self, config: Dict):
        """Initialize the Teleia Cervantes AVE task."""
        super().__init__(config, "cervantes_ave", process_cervantes)
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "teleia_cervantes_ave"

