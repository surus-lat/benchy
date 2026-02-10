"""Task registry and runner helpers for handler-based tasks."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .group_runner import SubtaskContext, TaskGroupSpec, run_task_group

logger = logging.getLogger(__name__)

_INTERNAL_TASK_GROUPS = {"common"}


def _tasks_root() -> Path:
    # Task modules live under src/tasks/*.
    return Path(__file__).resolve().parent


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


def _is_internal_task_group(group_name: str) -> bool:
    return group_name.startswith("_") or group_name in _INTERNAL_TASK_GROUPS


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
    if _is_internal_task_group(group_name):
        return None

    tasks_root = _tasks_root()
    group_dir = tasks_root / group_name

    if not group_dir.exists() or not group_dir.is_dir():
        return None

    # Check if this is a convention-based task group (has subtask .py files)
    subtask_files = sorted(
        (f for f in group_dir.glob("*.py") if _is_subtask_file(f)),
        key=lambda p: p.name,
    )
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


def list_handler_task_groups() -> List[str]:
    """List discoverable handler-based task groups."""
    tasks_root = _tasks_root()
    if not tasks_root.exists():
        return []

    groups: List[str] = []
    for group_dir in sorted(tasks_root.iterdir(), key=lambda p: p.name):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        if _is_internal_task_group(group_name):
            continue
        if discover_task_group(group_name):
            groups.append(group_name)
    return groups


def build_handler_task_config(
    task_ref: str,
    task_defaults_overrides: Optional[Dict[str, Any]] = None,
    adhoc_task_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a task config payload for a handler-based task reference.
    
    Args:
        task_ref: Task reference (e.g., "classify.environmental_claims" or "_adhoc_classification_abc123")
        task_defaults_overrides: Optional defaults to override
        adhoc_task_configs: Optional ad-hoc task configurations from config["task_configs"]
        
    Returns:
        Task configuration dict
    """
    # Check if this is an ad-hoc task
    if task_ref.startswith("_adhoc_") and adhoc_task_configs and task_ref in adhoc_task_configs:
        return build_adhoc_task_config(
            task_ref,
            adhoc_task_configs[task_ref],
            task_defaults_overrides,
        )
    
    # Regular handler-based task
    parts = task_ref.split(".", 1)
    group_name = parts[0]
    specific_subtask = parts[1].replace("-", "_") if len(parts) > 1 else None

    group_info = discover_task_group(group_name)
    if not group_info:
        raise ValueError(f"Unknown task '{task_ref}'. Handler-based task group not found.")

    available_subtasks = [s.name for s in group_info.subtasks]
    if specific_subtask:
        if specific_subtask not in available_subtasks:
            raise ValueError(f"Subtask '{specific_subtask}' not found in group '{group_name}'.")
        tasks_to_run = [specific_subtask]
    else:
        tasks_to_run = available_subtasks

    metadata = group_info.metadata or {}
    defaults = dict(metadata.get("defaults") or {})
    if task_defaults_overrides:
        defaults.update(task_defaults_overrides)

    output_cfg = metadata.get("output")
    if not isinstance(output_cfg, dict):
        output_cfg = {"subdirectory": group_name}

    return {
        "name": group_info.name,
        "display_name": group_info.display_name,
        "description": group_info.description,
        "tasks": tasks_to_run,
        "defaults": defaults,
        "prompts": dict(metadata.get("prompts") or {}),
        "task_configs": dict(metadata.get("task_configs") or {}),
        "output": output_cfg,
        "capability_requirements": dict(metadata.get("capability_requirements") or {}),
        "metrics_manifest": metadata.get("metrics_manifest", []),
    }


def build_adhoc_task_config(
    task_name: str,
    task_config: Dict[str, Any],
    task_defaults_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build configuration for an ad-hoc task.
    
    Args:
        task_name: Name of the ad-hoc task (e.g., "_adhoc_classification_abc123")
        task_config: Task configuration from config["task_configs"]
        task_defaults_overrides: Optional defaults to override
        
    Returns:
        Task configuration dict compatible with registry
    """
    from src.tasks.common import build_task_metadata, apply_defaults
    
    # Extract task type from name
    # Format: _adhoc_{task_type}_{hash}
    parts = task_name.split("_")
    if len(parts) >= 3 and parts[0] == "" and parts[1] == "adhoc":
        task_type = parts[2]
    else:
        raise ValueError(f"Invalid ad-hoc task name format: {task_name}")
    
    # Apply defaults to task_config (adds default field mappings)
    task_config_with_defaults = apply_defaults(task_type, dict(task_config))
    
    # Build metadata using task_config_schema
    metadata = build_task_metadata(task_type, task_config_with_defaults)
    
    # Apply defaults overrides
    defaults = dict(task_config_with_defaults.get("defaults", {}))
    if task_defaults_overrides:
        defaults.update(task_defaults_overrides)
    
    # Merge dataset config from task_config
    if "dataset" in task_config_with_defaults:
        defaults["dataset"] = task_config_with_defaults["dataset"]
    
    # Build subtask name (ad-hoc tasks have a single subtask)
    subtask_name = "main"
    
    return {
        "name": task_name,
        "display_name": metadata["display_name"],
        "description": metadata["description"],
        "tasks": [subtask_name],
        "defaults": defaults,
        "prompts": task_config.get("prompts", {}),
        "task_configs": {
            subtask_name: task_config
        },
        "output": {"subdirectory": task_name},
        "capability_requirements": metadata.get("capability_requirements", {}),
        "metrics_manifest": metadata.get("metrics_manifest", []),
        "_adhoc": True,  # Mark as ad-hoc for special handling
        "_task_type": task_type,  # Store task type for handler loading
    }


def build_adhoc_task_spec(
    task_name: str,
    task_config: Dict[str, Any],
    task_defaults_overrides: Optional[Dict[str, Any]] = None,
) -> TaskGroupSpec:
    """Build a TaskGroupSpec for an ad-hoc task.
    
    Args:
        task_name: Name of the ad-hoc task
        task_config: Task configuration
        task_defaults_overrides: Optional defaults to override
        
    Returns:
        TaskGroupSpec for the ad-hoc task
    """
    from src.tasks.common import get_handler_class
    
    # Extract task type
    parts = task_name.split("_")
    if len(parts) >= 3 and parts[0] == "" and parts[1] == "adhoc":
        task_type = parts[2]
    else:
        raise ValueError(f"Invalid ad-hoc task name format: {task_name}")
    
    # Get the handler class for this task type
    handler_class = get_handler_class(task_type)
    
    # Build config
    config_dict = build_adhoc_task_config(task_name, task_config, task_defaults_overrides)
    
    def _prepare_task(context: SubtaskContext):
        """Prepare an ad-hoc task handler."""
        # Merge defaults with subtask config
        def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = dict(default)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        dataset_config = _merge_dict(
            context.defaults.get("dataset", {}),
            context.subtask_config.get("dataset", {})
        )
        
        handler_config = {
            "subtask_name": context.subtask_name,
            "dataset": dataset_config,
            "prompts": context.subtask_config.get("prompts", context.prompts),
            "metrics": context.subtask_config.get("metrics", {}),
            "capability_requirements": context.task_config.get("capability_requirements", {}),
        }
        
        # Add prompts to handler config
        if "system_prompt" in task_config:
            handler_config["system_prompt"] = task_config["system_prompt"]
        if "user_prompt_template" in task_config:
            handler_config["user_prompt_template"] = task_config["user_prompt_template"]
        
        # Instantiate handler
        return handler_class(handler_config)
    
    return TaskGroupSpec(
        name=config_dict["name"],
        display_name=config_dict["display_name"],
        default_subtasks=config_dict["tasks"],
        prepare_task=_prepare_task,
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

        # Build config for handler.
        # Note: group-level metadata capability_requirements are applied here so new tasks
        # can declare requirements in metadata.yaml without needing boilerplate in each class.
        capability_requirements: Dict[str, Any] = {}
        group_reqs = (group_info.metadata or {}).get("capability_requirements")
        if isinstance(group_reqs, dict):
            capability_requirements.update(group_reqs)
        task_reqs = context.task_config.get("capability_requirements")
        if isinstance(task_reqs, dict):
            capability_requirements.update(task_reqs)
        subtask_reqs = context.subtask_config.get("capability_requirements")
        if isinstance(subtask_reqs, dict):
            capability_requirements.update(subtask_reqs)

        # Merge defaults with subtask_config for dataset, prompts, and metrics
        # Priority: subtask_config > defaults
        def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Deep merge two dicts, with override taking precedence."""
            result = dict(default)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        dataset_config = _merge_dict(
            context.defaults.get("dataset", {}),
            context.subtask_config.get("dataset", {})
        )
        metrics_config = _merge_dict(
            context.defaults.get("metrics", {}),
            context.subtask_config.get("metrics", {})
        )

        handler_config = {
            "subtask_name": context.subtask_name,
            "dataset": dataset_config,
            "prompts": context.subtask_config.get("prompts", context.prompts),
            "metrics": metrics_config,
            "capability_requirements": capability_requirements,
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
    compatibility_mode: str = "skip",
) -> Dict[str, Any]:
    """Discover and run a convention-based handler task.
    
    Args:
        task_ref: Task reference like "classify", "classify.diag_test", or "_adhoc_classification_abc123"
        model_name: Model to evaluate
        output_path: Output directory
        server_info: Server info for vLLM
        task_config: Task configuration
        limit: Sample limit
        provider_config: Provider configuration
        
    Returns:
        Task results dict
    """
    def _normalize_subtask_ref(value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        return value.replace("-", "_")

    # Check if this is an ad-hoc task
    if task_ref.startswith("_adhoc_"):
        # Build spec for ad-hoc task
        adhoc_config = task_config.get("task_configs", {}).get(task_ref, {})
        if not adhoc_config:
            # Try to get from defaults (for CLI-created tasks)
            adhoc_config = task_config.get("defaults", {})
        
        spec = build_adhoc_task_spec(
            task_ref,
            adhoc_config,
            task_config.get("defaults"),
        )
    else:
        # Parse task reference
        parts = task_ref.split(".")
        group_name = parts[0]
        subtask_name_raw = parts[1] if len(parts) > 1 else None
        subtask_name = _normalize_subtask_ref(subtask_name_raw)

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
        compatibility_mode=compatibility_mode,
    )


def is_handler_based_task(task_name: str) -> bool:
    """Check if a task uses the convention-based handler system.
    
    Args:
        task_name: Name of the task (e.g., "classify" or "_adhoc_classification_abc123")
        
    Returns:
        True if task uses handler system, False otherwise
    """
    # Ad-hoc tasks are handler-based
    if task_name.startswith("_adhoc_"):
        return True
    
    group_name = task_name.split(".")[0]
    group_info = discover_task_group(group_name)
    return group_info is not None
