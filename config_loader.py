"""Compatibility shim for load_config (moved to misc/ in 2026-06-19 reorg)."""

from src.config_loader import load_config

__all__ = ["load_config"]
