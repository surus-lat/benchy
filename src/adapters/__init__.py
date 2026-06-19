"""Adapter discovery and instantiation.

The filesystem is the registry: each adapter lives in
`src/adapters/<name>.py` and exports a class literally named
`Adapter`. No central registry file to keep in sync.

Names starting with `_` and the literal name `base` are reserved
for internal use and never listed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from src.adapters.base import BaseAdapter


def list_adapters() -> list[str]:
    """Return the sorted list of adapter names found on disk."""
    return sorted(
        m.name
        for m in pkgutil.iter_modules(__path__)
        if not m.name.startswith("_") and m.name != "base"
    )


def get_adapter(
    name: str, model_name: str, config: dict[str, Any]
) -> BaseAdapter:
    """Import the adapter module by name and instantiate its
    `Adapter` class.

    Raises ValueError if no module by that name exists, with the
    available adapter list in the message.
    """
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown adapter {name!r}. Available: {list_adapters()}"
        ) from exc
    return module.Adapter(model_name, config)


__all__ = ["BaseAdapter", "list_adapters", "get_adapter"]
