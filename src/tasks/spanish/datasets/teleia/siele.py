"""Teleia SIELE task implementation."""

from typing import Dict

from .base import TeleiaTaskBase
from .preprocessing import process_pce_siele


class TeleiaSieleTask(TeleiaTaskBase):
    """Teleia SIELE task."""
    
    def __init__(self, config: Dict):
        """Initialize the Teleia SIELE task."""
        super().__init__(config, "siele", process_pce_siele)
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "teleia_siele"

