"""Shared configuration loader for Benchy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def _find_project_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start


def _resolve_config_path(config_path: Union[str, Path]) -> Path:
    path = Path(config_path)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    project_root = _find_project_root(Path(__file__).resolve())
    candidate = project_root / path
    return candidate


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get("BENCHY_CONFIG", "configs/config.yaml")

    config_path = _resolve_config_path(config_path)

    if not config_path.exists():
        available_configs = []
        configs_dir = _resolve_config_path("configs")
        if configs_dir.exists():
            available_configs = [
                str(path) for path in configs_dir.glob("*.yaml")
            ]

        error_msg = f"Configuration file '{config_path}' not found."
        if available_configs:
            error_msg += "\n\nAvailable configurations:\n" + "\n".join(
                f"  - {cfg}" for cfg in available_configs[:5]
            )
            if len(available_configs) > 5:
                error_msg += f"\n  ... and {len(available_configs) - 5} more in configs/"
        error_msg += "\n\nTry: python eval.py --config <path-to-config.yaml>"

        raise FileNotFoundError(error_msg)

    with config_path.open("r") as f:
        return yaml.safe_load(f)
