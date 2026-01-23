"""Signal handling utilities for safely cleaning up running vLLM servers."""

from __future__ import annotations

import logging
import signal
import sys
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_active_server_info: Optional[Dict[str, Any]] = None


def set_active_server_info(server_info: Optional[Dict[str, Any]]) -> None:
    global _active_server_info
    _active_server_info = server_info


def clear_active_server_info() -> None:
    set_active_server_info(None)


def _signal_handler(signum, frame) -> None:  # pragma: no cover
    logger.info("Received signal %s. Cleaning up...", signum)

    if _active_server_info is not None:
        try:
            from .inference.vllm_server import stop_vllm_server

            logger.info("Stopping vLLM server...")
            stop_vllm_server(_active_server_info, {})
        except Exception as exc:
            logger.warning("Error stopping vLLM server: %s", exc)

    logger.info("Cleanup complete. Exiting.")
    sys.exit(130)


def register_signal_handlers() -> None:
    """Register signal handlers for clean termination."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

