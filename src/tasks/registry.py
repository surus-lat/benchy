"""Task registry and simple runner helpers.

This module provides task discovery capabilities:
- Convention-based discovery for format handler tasks (new system)
- JSON config entrypoint discovery for SimpleTask-based tasks (legacy)
- Build generic TaskGroupSpec to run tasks through the engine
"""

from __future__ import annotations

import importlib
import logging
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


_PRIMARY_METRIC_PRIORITY = [
    "accuracy",
    "acc",
    "exact_match",
    "f1_macro",
    "pearson",
    "comet",
    "chrf",
    "bleu",
    "overall_extraction_quality_score",
    "extraction_quality_score",
    "field_f1_partial",
    "mse",
]

_AGGREGATION_SKIP_KEYS = {
    "total_samples",
    "valid_samples",
    "error_count",
    "error_rate",
    "total_duration",
    "throughput",
    "run_samples",
    "sample_count",
}


def _metric_weight(metrics: Dict[str, Any]) -> float:
    for key in ("valid_samples", "total_samples"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 0.0


def _select_primary_metric(
    metrics: Dict[str, Any],
    preferred: Optional[str] = None,
) -> tuple[Optional[str], Optional[float]]:
    if preferred and preferred != "auto":
        value = metrics.get(preferred)
        if isinstance(value, (int, float)):
            return preferred, float(value)

    for key in _PRIMARY_METRIC_PRIORITY:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    return None, None


def _sample_weighted_aggregate(
    per_subtask_metrics: Dict[str, Dict[str, Any]],
    primary_metric: Optional[str],
) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}

    total_samples = sum(
        m.get("total_samples", 0) for m in per_subtask_metrics.values() if isinstance(m.get("total_samples"), (int, float))
    )
    valid_samples = sum(
        m.get("valid_samples", 0) for m in per_subtask_metrics.values() if isinstance(m.get("valid_samples"), (int, float))
    )
    error_count = 0.0
    for metrics in per_subtask_metrics.values():
        if isinstance(metrics.get("error_count"), (int, float)):
            error_count += float(metrics.get("error_count"))
        else:
            total = metrics.get("total_samples")
            valid = metrics.get("valid_samples")
            if isinstance(total, (int, float)) and isinstance(valid, (int, float)):
                error_count += float(total - valid)

    aggregated["total_samples"] = int(total_samples)
    aggregated["valid_samples"] = int(valid_samples)
    aggregated["error_count"] = int(error_count)
    aggregated["error_rate"] = (error_count / total_samples) if total_samples > 0 else 0.0

    numeric_keys: set[str] = set()
    for metrics in per_subtask_metrics.values():
        for key, value in metrics.items():
            if key in _AGGREGATION_SKIP_KEYS:
                continue
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    for key in sorted(numeric_keys):
        weighted_sum = 0.0
        total_weight = 0.0
        for metrics in per_subtask_metrics.values():
            value = metrics.get(key)
            if not isinstance(value, (int, float)):
                continue
            weight = _metric_weight(metrics)
            if weight <= 0:
                continue
            weighted_sum += float(value) * weight
            total_weight += weight
        if total_weight > 0:
            aggregated[key] = weighted_sum / total_weight

    if primary_metric:
        if primary_metric == "auto":
            weighted_sum = 0.0
            total_weight = 0.0
            for metrics in per_subtask_metrics.values():
                _, value = _select_primary_metric(metrics)
                if value is None:
                    continue
                weight = _metric_weight(metrics)
                if weight <= 0:
                    continue
                weighted_sum += value * weight
                total_weight += weight
            if total_weight > 0:
                aggregated["overall_score"] = weighted_sum / total_weight
        else:
            weighted_sum = 0.0
            total_weight = 0.0
            for metrics in per_subtask_metrics.values():
                value = metrics.get(primary_metric)
                if not isinstance(value, (int, float)):
                    continue
                weight = _metric_weight(metrics)
                if weight <= 0:
                    continue
                weighted_sum += float(value) * weight
                total_weight += weight
            if total_weight > 0:
                aggregated[primary_metric] = weighted_sum / total_weight

    return aggregated


def _snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case to PascalCase.
    
    Args:
        snake_str: String in snake_case (e.g., "diag_test")
        
    Returns:
        String in PascalCase (e.g., "DiagTest")
    """
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _is_subtask_file(file_path: Path) -> bool:
    """Check if a file is a subtask definition.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if this is a subtask file
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
    """Discover a task group using convention-based file scanning.
    
    Scans the task group directory for:
    - metadata.yaml (optional) for group metadata
    - *.py files (excluding __init__, run, etc.) as subtasks
    
    Args:
        group_name: Name of the task group (e.g., "classify")
        
    Returns:
        TaskGroupInfo if found and using new convention, None otherwise
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
    """Load and instantiate a subtask handler class.
    
    Args:
        subtask_info: Information about the subtask
        config: Optional configuration dict to pass to handler
        
    Returns:
        Instantiated handler object
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
    """Build a TaskGroupSpec for convention-based handler tasks.
    
    Args:
        group_info: Information about the task group
        subtask_name: Specific subtask to run, or None for all
        
    Returns:
        TaskGroupSpec configured for the handler task(s)
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
        """Prepare a handler task for the given subtask context."""
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

    aggregation_config = group_info.metadata.get("aggregation", {}) if group_info.metadata else {}
    aggregation_method = aggregation_config.get("method")
    primary_metric = aggregation_config.get("primary_metric")

    def _aggregate_metrics(all_metrics: Dict[str, Dict[str, Any]], subtasks_run: List[str]) -> Dict[str, Any]:
        """Aggregate metrics from multiple subtasks.
        
        For single subtask: return metrics directly (flat).
        For multiple subtasks: return under 'subtasks' key for clarity.
        """
        if len(subtasks_run) == 1:
            # Single subtask: keep metrics flat for backward compatibility
            return all_metrics.get(subtasks_run[0], {})
        else:
            # Multiple subtasks: structure under 'subtasks' key
            per_subtask = {name: all_metrics.get(name, {}) for name in subtasks_run}
            if aggregation_method == "sample_weighted":
                aggregated = _sample_weighted_aggregate(per_subtask, primary_metric)
                return {
                    "subtasks": per_subtask,
                    "aggregated_metrics": aggregated,
                }
            return {"subtasks": per_subtask}

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
    """Discover and run a convention-based handler task.
    
    Args:
        task_ref: Task reference like "classify" or "classify.diag_test"
        model_name: Model to evaluate
        output_path: Output directory
        server_info: Server info for vLLM
        task_config: Task configuration
        limit: Sample limit
        provider_config: Provider configuration
        
    Returns:
        Task results dict
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
    """Check if a task uses the convention-based handler system.
    
    Args:
        task_name: Name of the task (e.g., "classify")
        
    Returns:
        True if task uses handler system, False otherwise
    """
    group_name = task_name.split(".")[0]
    group_info = discover_task_group(group_name)
    return group_info is not None
