"""Teleia PCE task implementation."""

from typing import Dict

from .base import TeleiaTaskBase
from .preprocessing import process_pce_siele


class TeleiaPceTask(TeleiaTaskBase):
    """Teleia PCE task."""
    
    def __init__(self, config: Dict):
        """Initialize the Teleia PCE task."""
        super().__init__(config, "pce", process_pce_siele)
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "teleia_pce"

