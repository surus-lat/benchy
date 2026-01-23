"""Shared configuration loader for Benchy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Iterable, List

import yaml


def _find_project_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start


def _resolve_config_path(config_path: Union[str, Path]) -> Path:
    # Backward-compatible wrapper.
    return resolve_config_path(config_path)


def _iter_config_search_dirs(project_root: Path) -> Iterable[Path]:
    # Precedence order for short names.
    configs_root = project_root / "configs"
    return (
        configs_root / "models",
        configs_root / "systems",
        configs_root / "tests",
        configs_root / "templates",
        configs_root,
    )


def resolve_config_path(config_path: Union[str, Path]) -> Path:
    """Resolve a config reference to an on-disk YAML path.

    Supports:
    - Absolute or relative file paths
    - Short names that are searched under `configs/models`, `configs/systems`, etc.
      e.g. `openai_gpt-4o-mini.yaml` or `openai_gpt-4o-mini`
    """
    project_root = _find_project_root(Path(__file__).resolve())

    raw = str(config_path)
    path = Path(raw)

    # Explicit paths (absolute or with separators) keep existing behavior.
    if path.is_absolute():
        return path
    if any(sep in raw for sep in ("/", "\\")) or raw.startswith("."):
        candidate = project_root / path
        return candidate if candidate.exists() else path

    # Short name lookup under configs/*.
    variants: List[str] = [raw]
    if not raw.endswith((".yaml", ".yml")):
        variants.extend([f"{raw}.yaml", f"{raw}.yml"])

    matches: List[Path] = []
    for directory in _iter_config_search_dirs(project_root):
        for variant in variants:
            candidate = directory / variant
            if candidate.exists():
                matches.append(candidate)

    if not matches:
        # Fall back to configs/<name> for improved error messages.
        return project_root / "configs" / (variants[1] if len(variants) > 1 else variants[0])

    # If there are multiple matches, require disambiguation via path.
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) > 1:
        rendered = "\n".join(f"  - {m}" for m in unique_matches[:10])
        suffix = "" if len(unique_matches) <= 10 else f"\n  ... and {len(unique_matches) - 10} more"
        raise FileNotFoundError(
            "Config name is ambiguous. Specify a more explicit path, e.g. "
            "`--config configs/models/<name>.yaml`.\n\nMatches:\n"
            f"{rendered}{suffix}"
        )

    return unique_matches[0]


def _resolve_paths_section(config: Dict[str, Any], project_root: Path) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        return

    resolved: Dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, str):
            expanded = os.path.expanduser(os.path.expandvars(value))
            path_value = Path(expanded)
            if not path_value.is_absolute():
                path_value = project_root / path_value
            resolved[key] = str(path_value)
        else:
            resolved[key] = value

    config["paths"] = resolved


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get("BENCHY_CONFIG", "configs/config.yaml")

    config_path = _resolve_config_path(config_path)

    if not config_path.exists():
        available_configs = []
        configs_dir = _resolve_config_path("configs")
        project_root = _find_project_root(Path(__file__).resolve())
        if configs_dir.exists():
            # Show a sample of configs across common subdirectories.
            available = []
            for directory in _iter_config_search_dirs(project_root):
                if directory.exists():
                    available.extend([str(path) for path in directory.glob("*.yaml")])
                    available.extend([str(path) for path in directory.glob("*.yml")])
            available_configs = sorted(set(available))

        error_msg = f"Configuration file '{config_path}' not found."
        if available_configs:
            error_msg += "\n\nAvailable configurations:\n" + "\n".join(
                f"  - {cfg}" for cfg in available_configs[:5]
            )
            if len(available_configs) > 5:
                error_msg += f"\n  ... and {len(available_configs) - 5} more in configs/"
        error_msg += "\n\nTip: you can pass a config name, e.g. `--config openai_gpt-4o-mini.yaml`."

        raise FileNotFoundError(error_msg)

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    project_root = _find_project_root(config_path.resolve().parent)
    _resolve_paths_section(config, project_root)
    return config
