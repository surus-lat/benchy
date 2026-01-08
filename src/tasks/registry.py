"""Task registry and simple runner helpers.

This module provides task discovery capabilities:
- Convention-based discovery for format handler tasks (new system)
- JSON config entrypoint discovery for SimpleTask-based tasks (legacy)
- Build generic TaskGroupSpec to run tasks through the engine
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

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
    """
    Run a SimpleTask via the generic task group runner.
    
    Returns:
        results (Dict[str, Any]): Execution results and metrics produced by the task group run.
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


# ============================================================================
# NEW CONVENTION-BASED DISCOVERY SYSTEM
# ============================================================================


@dataclass(frozen=True)
class SubtaskInfo:
    """Information about a discovered subtask."""

    name: str
    group_name: str
    module_path: str
    class_name: str
    file_path: Path


@dataclass(frozen=True)
class TaskGroupInfo:
    """Information about a discovered task group."""

    name: str
    display_name: str
    description: str
    subtasks: List[SubtaskInfo]
    metadata: Dict[str, Any]
    metadata_file: Optional[Path]


def _snake_to_pascal(snake_str: str) -> str:
    """
    Convert a snake_case string to PascalCase.
    
    Parameters:
        snake_str (str): Input string in snake_case (e.g., "diag_test").
    
    Returns:
        pascal (str): PascalCase version of the input (e.g., "DiagTest").
    """
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _is_subtask_file(file_path: Path) -> bool:
    """
    Determine whether a given filesystem path refers to a Python module that should be treated as a subtask definition.
    
    Parameters:
        file_path (Path): Path to the candidate file.
    
    Returns:
        bool: `True` if the path is a `.py` file whose stem is not excluded and does not start with an underscore, `False` otherwise.
    """
    if not file_path.suffix == ".py":
        return False
    excluded_stems = [
        "__init__", "__pycache__", "run", "base", "task",
        "metrics", "preprocessing", "spanish_handlers"  # Helper modules
    ]
    if file_path.stem in excluded_stems:
        return False
    if file_path.stem.startswith("_"):
        return False
    return True


def discover_task_group(group_name: str) -> Optional[TaskGroupInfo]:
    """
    Discover a convention-based task group by scanning its directory for subtask modules and optional metadata.
    
    Scans src/tasks/{group_name} for Python files considered subtasks and for an optional metadata.yaml. If the directory does not exist or contains no valid subtask files, returns None. When found, returns a TaskGroupInfo containing discovered SubtaskInfo entries, parsed metadata, and the metadata file path (if present).
    
    Parameters:
        group_name (str): Task group directory name (e.g., "classify").
    
    Returns:
        Optional[TaskGroupInfo]: A TaskGroupInfo for the discovered group, or `None` if the group is not present or contains no convention-based subtasks.
    """
    tasks_root = _tasks_root()
    group_dir = tasks_root / group_name

    if not group_dir.exists() or not group_dir.is_dir():
        return None

    # Check if this is a convention-based task group (has subtask .py files)
    subtask_files = [f for f in group_dir.glob("*.py") if _is_subtask_file(f)]
    if not subtask_files:
        return None

    # Load metadata if available
    metadata_file = group_dir / "metadata.yaml"
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load metadata for {group_name}: {e}")

    # Discover subtasks
    subtasks = []
    for subtask_file in subtask_files:
        subtask_name = subtask_file.stem
        class_name = _snake_to_pascal(subtask_name)
        module_path = f"src.tasks.{group_name}.{subtask_name}"

        subtasks.append(
            SubtaskInfo(
                name=subtask_name,
                group_name=group_name,
                module_path=module_path,
                class_name=class_name,
                file_path=subtask_file,
            )
        )

    display_name = metadata.get("display_name", group_name.replace("_", " ").title())
    description = metadata.get("description", "")

    return TaskGroupInfo(
        name=group_name,
        display_name=display_name,
        description=description,
        subtasks=subtasks,
        metadata=metadata,
        metadata_file=metadata_file if metadata_file.exists() else None,
    )


def load_subtask_handler(subtask_info: SubtaskInfo, config: Optional[Dict[str, Any]] = None):
    """
    Import and instantiate the handler class for a discovered subtask.
    
    Parameters:
        subtask_info (SubtaskInfo): Metadata identifying the subtask's module path, class name, and name.
        config (Optional[Dict[str, Any]]): Configuration passed to the handler's constructor.
    
    Returns:
        Any: An instance of the handler class.
    
    Raises:
        Exception: If the module cannot be imported, the class cannot be resolved, or instantiation fails.
    """
    try:
        module = importlib.import_module(subtask_info.module_path)
        handler_class = getattr(module, subtask_info.class_name)
        return handler_class(config)
    except Exception as e:
        logger.error(
            f"Failed to load subtask {subtask_info.name} from {subtask_info.module_path}: {e}"
        )
        raise


def build_handler_task_spec(
    group_info: TaskGroupInfo, subtask_name: Optional[str] = None
) -> TaskGroupSpec:
    """
    Create a TaskGroupSpec for a convention-based handler task group, optionally limited to a single subtask.
    
    Parameters:
        group_info (TaskGroupInfo): Discovered task group and its subtasks.
        subtask_name (Optional[str]): If provided, restrict the spec to this subtask.
    
    Returns:
        TaskGroupSpec: A spec configured to prepare and run the group's handler-based subtasks. The spec will have supports_subtasks=True and default_subtasks set to the selected subtasks.
    
    Raises:
        ValueError: If `subtask_name` is provided but not found in `group_info.subtasks`.
    """
    # Filter subtasks if specific one requested
    if subtask_name:
        subtasks = [s for s in group_info.subtasks if s.name == subtask_name]
        if not subtasks:
            raise ValueError(
                f"Subtask '{subtask_name}' not found in group '{group_info.name}'"
            )
    else:
        subtasks = group_info.subtasks

    def _prepare_task(context: SubtaskContext):
        """
        Prepare and instantiate a handler for the subtask identified in the given context.
        
        Looks up the SubtaskInfo matching context.subtask_name, builds a handler configuration
        including subtask_name, dataset, prompts, and metrics (merging context and subtask-specific overrides),
        and returns a new handler instance created by load_subtask_handler. If the subtask is not found,
        returns None.
        
        Parameters:
            context (SubtaskContext): Context containing `subtask_name`, `subtask_config`, and `prompts`.
        
        Returns:
            handler (Any | None): An instantiated subtask handler, or `None` if the subtask was not found.
        """
        # Find the subtask info
        subtask_info = next(
            (s for s in subtasks if s.name == context.subtask_name), None
        )
        if not subtask_info:
            return None

        # Build config for handler
        handler_config = {
            "subtask_name": context.subtask_name,
            "dataset": context.subtask_config.get("dataset", {}),
            "prompts": context.subtask_config.get("prompts", context.prompts),
            "metrics": context.subtask_config.get("metrics", {}),
        }

        # Load and instantiate the handler
        return load_subtask_handler(subtask_info, handler_config)

    def _aggregate_metrics(all_metrics: Dict[str, Dict[str, Any]], subtasks_run: List[str]) -> Dict[str, Any]:
        """
        Aggregate metrics returned by one or more subtasks into the final metrics structure.
        
        For a single subtask, returns that subtask's metrics directly (flat dict) for backward compatibility.
        For multiple subtasks, returns a dict with a top-level "subtasks" key mapping each subtask name to its metrics.
        
        Parameters:
            all_metrics (Dict[str, Dict[str, Any]]): Mapping from subtask name to its metrics dictionary.
            subtasks_run (List[str]): Ordered list of subtask names that were executed.
        
        Returns:
            Dict[str, Any]: If one subtask ran, the metrics dict for that subtask; if multiple ran, a dict of the form
            {"subtasks": {subtask_name: metrics_dict, ...}}.
        """
        if len(subtasks_run) == 1:
            # Single subtask: keep metrics flat for backward compatibility
            return all_metrics.get(subtasks_run[0], {})
        else:
            # Multiple subtasks: structure under 'subtasks' key
            return {"subtasks": {name: all_metrics.get(name, {}) for name in subtasks_run}}

    return TaskGroupSpec(
        name=group_info.name,
        display_name=group_info.display_name,
        output_subdir=group_info.name,
        supports_subtasks=True,  # Always True for handler-based tasks
        default_subtasks=[s.name for s in subtasks],
        prepare_task=_prepare_task,
        aggregate_metrics=_aggregate_metrics,
    )


def discover_and_run_handler_task(
    task_ref: str,
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Discover and run a convention-based handler task identified by a task reference.
    
    Parses `task_ref` into a group name and optional subtask name, discovers the corresponding task group, builds a TaskGroupSpec for the group (or specific subtask), updates `task_config` to restrict to the requested subtask when provided, and runs the task group via the generic runner.
    
    Parameters:
    	task_ref (str): Task reference like "group" or "group.subtask".
    	model_name (str): Name of the model to evaluate.
    	output_path (str): Directory to write outputs to.
    	server_info (Optional[Dict[str, Any]]): Optional server configuration (e.g., for vLLM).
    	task_config (Dict[str, Any]): Task configuration; may be modified to include a `tasks` list when a specific subtask is requested.
    	limit (Optional[int]): Optional sample limit.
    	provider_config (Optional[Dict[str, Any]]): Optional provider configuration.
    
    Returns:
    	results (Dict[str, Any]): Results dictionary produced by running the task group.
    
    Raises:
    	ValueError: If the referenced task group is not found or is not using the convention-based handler system.
    """
    # Parse task reference
    parts = task_ref.split(".")
    group_name = parts[0]
    subtask_name = parts[1] if len(parts) > 1 else None

    # Discover task group
    group_info = discover_task_group(group_name)
    if not group_info:
        raise ValueError(f"Task group '{group_name}' not found or not using convention system")

    # Build spec
    spec = build_handler_task_spec(group_info, subtask_name)

    # If specific subtask requested, update task_config
    if subtask_name:
        # Ensure the subtask is in the tasks list
        if "tasks" not in task_config:
            task_config["tasks"] = [subtask_name]
        elif subtask_name not in task_config["tasks"]:
            task_config["tasks"] = [subtask_name]

    # Run using group runner
    return run_task_group(
        spec=spec,
        model_name=model_name,
        output_path=output_path,
        server_info=server_info,
        task_config=task_config,
        limit=limit,
        provider_config=provider_config,
    )


def is_handler_based_task(task_name: str) -> bool:
    """
    Determine whether a task name corresponds to a convention-based handler group.
    
    Parameters:
        task_name (str): Task reference or name. If the name contains a dot (e.g., "group.subtask"),
            the portion before the first dot is used as the group name to discover a handler group.
    
    Returns:
        `true` if the group is discovered (uses the convention-based handler system), `false` otherwise.
    """
    group_name = task_name.split(".")[0]
    group_info = discover_task_group(group_name)
    return group_info is not None