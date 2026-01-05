"""Task registry and simple runner helpers.

This module provides two capabilities:
- Discover task.json entrypoints for SimpleTask-based tasks.
- Build a generic TaskGroupSpec to run SimpleTask tasks through the engine.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .group_runner import SubtaskContext, TaskGroupSpec, run_task_group

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredTask:
    """Metadata for a registered SimpleTask entrypoint."""

    name: str
    entrypoint: str
    display_name: str
    provider_types: Optional[list]
    output_subdir: str


def is_simple_task_config(task_config: Dict[str, Any]) -> bool:
    # SimpleTask configs are identified by a class entrypoint.
    return bool(task_config.get("entrypoint"))


def _tasks_root() -> Path:
    # Task configs live under src/tasks/*/task.json.
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, RegisteredTask]:
    # Scan task.json files and collect entrypoint metadata.
    tasks_root = _tasks_root()
    registry: Dict[str, RegisteredTask] = {}
    for task_path in tasks_root.rglob("task.json"):
        if task_path.parent.name == "_template":
            continue
        try:
            import json
            with open(task_path, "r") as handle:
                task_config = json.load(handle)
        except Exception:
            continue
        entrypoint = task_config.get("entrypoint")
        if not entrypoint:
            continue
        name = task_config.get("name")
        if not name:
            continue
        display_name = task_config.get("display_name", name)
        provider_types = task_config.get("provider_types")
        output_subdir = task_config.get("output", {}).get("subdirectory", name)
        registry[name] = RegisteredTask(
            name=name,
            entrypoint=entrypoint,
            display_name=display_name,
            provider_types=provider_types,
            output_subdir=output_subdir,
        )
    return registry


def get_registered_task(task_name: str) -> Optional[RegisteredTask]:
    # Cached lookup for registered entrypoints.
    return load_registry().get(task_name)


def resolve_entrypoint(entrypoint: str) -> Any:
    """Resolve a module:attribute entrypoint.

    Returns a class or function, depending on the entrypoint target.
    """
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint '{entrypoint}'. Expected 'module:attribute'.")
    module_path, attr = entrypoint.split(":", 1)
    module = importlib.import_module(module_path)
    resolved = getattr(module, attr, None)
    if resolved is None:
        raise ValueError(f"Entrypoint '{entrypoint}' did not resolve to a symbol.")
    return resolved


def build_simple_task_spec(task_config: Dict[str, Any]) -> TaskGroupSpec:
    # Build a TaskGroupSpec that instantiates the SimpleTask class.
    entrypoint = task_config.get("entrypoint")
    if not entrypoint:
        raise ValueError("Task config missing entrypoint for SimpleTask.")
    task_class = resolve_entrypoint(entrypoint)

    def _prepare_task(context: SubtaskContext):
        # Provide task.json config as-is to the SimpleTask constructor.
        merged_config = dict(context.task_config)
        merged_config["dataset"] = context.task_config.get("dataset", {})
        merged_config["prompts"] = context.task_config.get("prompts", {})
        return task_class(merged_config)

    display_name = task_config.get("display_name") or task_config.get("name", "Simple Task")
    output_subdir = task_config.get("output", {}).get("subdirectory", task_config.get("name", "task"))

    return TaskGroupSpec(
        name=task_config.get("name", "simple_task"),
        display_name=display_name,
        output_subdir=output_subdir,
        supports_subtasks=False,
        prepare_task=_prepare_task,
    )


def run_simple_task(
    *,
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    api_test_result: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    cuda_devices: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a SimpleTask using the generic task group runner.

    This mirrors the signature of run_* wrappers so the pipeline can call it.
    """
    spec = build_simple_task_spec(task_config)
    return run_task_group(
        spec=spec,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )
