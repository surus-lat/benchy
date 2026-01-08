"""Simple configuration manager for merging model and provider configs."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Centralized task config schema for light validation and discoverability.
TASK_CONFIG_SCHEMA = {
    "field_descriptions": {
        "name": "Unique task identifier used in configs and output paths.",
        "display_name": "Human-readable task name used in logs and summaries.",
        "description": "Short description of the task's purpose.",
        "entrypoint": "SimpleTask class entrypoint (module:ClassName).",
        "runner_entrypoint": "Runner entrypoint (module:run_fn) for grouped/custom tasks.",
        "provider_types": "List of supported provider types for this task.",
        "pipeline_overrides": "Optional pipeline flags (set_api_endpoint, set_generation_config).",
        "task_format": "Task format: multiple_choice, freeform, structured, grouped.",
        "defaults": "Task-level defaults (batch_size, timeout, max_tokens, etc.).",
        "output": "Output directory settings (subdirectory).",
        "metrics_manifest": "Aggregate metric keys to surface in run summaries.",
        "capability_requirements": "Required/preferred interface capabilities.",
        "prompts": "Prompt templates for LLM-style interfaces.",
        "dataset": "Dataset config (data_file or dataset_path + split).",
        "metrics": "Metric configuration (task-specific).",
        "tasks": "Subtask name list for grouped tasks.",
        "task_configs": "Per-subtask configuration overrides.",
        "group": "Group identifier for leaderboards/collections.",
        "group_metadata": "Group descriptions and metadata.",
        "task_metadata": "Per-subtask metadata used in reporting.",
        "source_dir": "Source data path for local datasets (e.g., images).",
    },
    "format_fields": {
        # Additional allowed fields per task_format (currently none).
        "multiple_choice": set(),
        "freeform": set(),
        "structured": set(),
        "grouped": set(),
    },
    "required_fields": {
        # Minimal requirements per format; the validator only warns.
        "multiple_choice": {"dataset", "prompts"},
        "freeform": {"dataset", "prompts"},
        "structured": {"prompts"},
        "grouped": {"tasks"},
    },
    "deprecated_fields": {
        "task_name": "Deprecated top-level field. Use 'name' instead.",
        "dataset_file": "Deprecated top-level field. Move under dataset.data_file.",
        "dataset_name": "Deprecated top-level field. Move under dataset.dataset_name.",
    },
}


class ConfigManager:
    """Simple config manager that merges model configs with provider configs."""
    
    def __init__(self, configs_dir: str = "configs"):
        self.configs_dir = Path(configs_dir)
    
    def load_model_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load model config and merge with provider config if needed.
        
        Args:
            config_path: Path to the model configuration file
            
        Returns:
            Merged configuration dictionary
        """
        # Load model config
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Handle OpenAI config merging
        if 'openai' in model_config:
            self._merge_cloud_provider_config(model_config, 'openai')
            model_config['provider_type'] = 'openai'
        
        # Handle Anthropic config merging
        elif 'anthropic' in model_config:
            self._merge_cloud_provider_config(model_config, 'anthropic')
            model_config['provider_type'] = 'anthropic'
        
        # Handle Together AI config merging
        elif 'together' in model_config:
            self._merge_cloud_provider_config(model_config, 'together')
            model_config['provider_type'] = 'together'
        
        # Handle vLLM config merging
        elif 'vllm' in model_config:
            vllm_config = model_config['vllm']
            
            # Check if this is a new format config with provider_config
            if 'provider_config' in vllm_config:
                provider_name = vllm_config['provider_config']
                logger.info(f"Loading provider config: {provider_name}")
                
                # Load base provider config
                provider_path = self.configs_dir / "providers" / f"{provider_name}.yaml"
                
                if not provider_path.exists():
                    raise FileNotFoundError(f"Provider config not found: {provider_path}")
                
                with open(provider_path, 'r') as f:
                    base_vllm_config = yaml.safe_load(f)
                
                # Merge with overrides
                overrides = vllm_config.get('overrides', {})
                merged_vllm_config = {**base_vllm_config, **overrides}
                
                # Handle vLLM version-specific virtual environment
                if 'vllm_version' in merged_vllm_config:
                    from .inference.venv_manager import get_vllm_venv_path
                    vllm_version = merged_vllm_config['vllm_version']
                    transformers_version = merged_vllm_config.get('transformers_version', None)
                    
                    if transformers_version:
                        print(f"🔍 Configuring vLLM {vllm_version} + transformers {transformers_version} environment...")
                    else:
                        print(f"🔍 Configuring vLLM {vllm_version} environment...")
                    
                    venv_path = get_vllm_venv_path(vllm_version, transformers_version=transformers_version)
                    merged_vllm_config['vllm_venv_path'] = venv_path
                    
                    if transformers_version:
                        logger.info(f"Using vLLM {vllm_version} + transformers {transformers_version} from virtual environment: {venv_path}")
                        print(f"✅ Using vLLM {vllm_version} + transformers {transformers_version} from: {venv_path}")
                    else:
                        logger.info(f"Using vLLM {vllm_version} from virtual environment: {venv_path}")
                        print(f"✅ Using vLLM {vllm_version} from: {venv_path}")
                else:
                    # Default to main project environment (latest vLLM version)
                    # Find the project root by looking for pyproject.toml
                    current_dir = Path(__file__).parent
                    while current_dir != current_dir.parent:
                        if (current_dir / "pyproject.toml").exists():
                            break
                        current_dir = current_dir.parent
                    default_venv_path = str(current_dir / ".venv")
                    merged_vllm_config['vllm_venv_path'] = default_venv_path
                    logger.info("Using default vLLM version from main project environment")
                    print(f"✅ Using default vLLM version from main project environment")
                
                # Log what was overridden
                if overrides:
                    logger.info(f"Applied overrides: {list(overrides.keys())}")
                
                # Replace vllm section with merged config
                model_config['vllm'] = merged_vllm_config
                
            # If no provider_config, assume it's an old format config (backward compatibility)
            else:
                logger.info("Using legacy vLLM config format")
            
            # Mark as vLLM provider
            model_config['provider_type'] = 'vllm'
        
        self._apply_metadata_capabilities(model_config)

        return model_config

    def _apply_metadata_capabilities(self, model_config: Dict[str, Any]) -> None:
        """Apply metadata capability tags into provider config."""
        metadata = model_config.get("metadata") or {}
        if not isinstance(metadata, dict):
            return

        provider_type = model_config.get("provider_type")
        if not provider_type:
            return

        provider_section = model_config.get(provider_type)
        if not isinstance(provider_section, dict):
            return

        model_capabilities = dict(provider_section.get("model_capabilities") or {})

        mapping = {
            "supports_multimodal": "supports_multimodal",
            "supports_schema": "supports_schema",
            "supports_files": "supports_files",
            "supports_logprobs": "supports_logprobs",
            "supports_streaming": "supports_streaming",
        }
        if "is_multimodal" in metadata:
            logger.warning("metadata.is_multimodal is deprecated; use metadata.supports_multimodal")
            model_capabilities["supports_multimodal"] = metadata["is_multimodal"]
        for meta_key, cap_key in mapping.items():
            if meta_key in metadata:
                model_capabilities[cap_key] = metadata[meta_key]

        if "request_modes" in metadata:
            model_capabilities["request_modes"] = metadata["request_modes"]

        if model_capabilities:
            provider_section["model_capabilities"] = model_capabilities
    
    def _merge_cloud_provider_config(self, model_config: Dict[str, Any], provider: str):
        """
        Merge cloud provider configuration with base provider config.
        
        Args:
            model_config: Model configuration dictionary to modify in-place
            provider: Provider name ('openai' or 'anthropic')
        """
        provider_config = model_config[provider]
        
        # Check if this is a new format config with provider_config
        if 'provider_config' in provider_config:
            provider_name = provider_config['provider_config']
            logger.info(f"Loading {provider} provider config: {provider_name}")
            
            # Load base provider config
            provider_path = self.configs_dir / "providers" / f"{provider_name}.yaml"
            
            if not provider_path.exists():
                raise FileNotFoundError(f"Provider config not found: {provider_path}")
            
            with open(provider_path, 'r') as f:
                base_provider_config = yaml.safe_load(f)
            
            # Merge with overrides
            overrides = provider_config.get('overrides', {})
            merged_config = {**base_provider_config, **overrides}
            
            # Log what was overridden
            if overrides:
                logger.info(f"Applied overrides: {list(overrides.keys())}")
            
            # Replace provider section with merged config
            model_config[provider] = merged_config
        else:
            # If no provider_config, assume it's a complete config (backward compatibility)
            logger.info(f"Using inline {provider} config format")
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Load a provider configuration file.
        
        Args:
            provider_name: Name of the provider config (without .yaml)
            
        Returns:
            Provider configuration dictionary
        """
        provider_path = self.configs_dir / "providers" / f"{provider_name}.yaml"
        
        if not provider_path.exists():
            raise FileNotFoundError(f"Provider config not found: {provider_path}")
        
        with open(provider_path, 'r') as f:
            return yaml.safe_load(f)
    
    def list_available_providers(self) -> list:
        """List all available provider configurations."""
        providers_dir = self.configs_dir / "providers"
        
        if not providers_dir.exists():
            return []
        
        return [f.stem for f in providers_dir.glob("*.yaml")]
    
    def list_available_models(self) -> list:
        """List all available model configurations."""
        models_dir = self.configs_dir / "models"
        
        if not models_dir.exists():
            return []
        
        return [f.stem for f in models_dir.glob("*.yaml")]

    def _get_tasks_root(self) -> Path:
        """Return the root directory that contains task.json configs."""
        return Path(__file__).resolve().parent / "tasks"

    def _load_task_config_from_tasks_root(self, task_name: str) -> Dict[str, Any]:
        tasks_root = self._get_tasks_root()

        if not tasks_root.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_root}")

        for task_path in tasks_root.rglob("task.json"):
            if task_path.parent.name == "_template":
                continue
            try:
                with open(task_path, "r") as f:
                    task_config = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in task config: {task_path}") from exc

            if task_config.get("name") == task_name:
                return task_config

        raise FileNotFoundError(f"Task config not found for: {task_name}")

    def get_task_config(self, task_name: str, task_defaults_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a task configuration by name, perform lightweight validation, and apply any provided default overrides.
        
        This searches for the task's task.json under the repository tasks tree, validates the loaded config for schema drift (may emit warnings), and then applies values from task_defaults_overrides into the task's "defaults" section (merging into an existing "defaults" map or creating one if missing).
        
        Parameters:
            task_name (str): Name of the task to load.
            task_defaults_overrides (Optional[Dict[str, Any]]): Mapping of default values to merge into the task's "defaults" section.
        
        Returns:
            Dict[str, Any]: The loaded task configuration with validation applied and defaults merged.
        """
        task_config = self._load_task_config_from_tasks_root(task_name)
        # Run lightweight validation to surface schema drift early.
        self._validate_task_config(task_config, task_name)
        
        # Apply task defaults overrides if provided
        if task_defaults_overrides:
            if 'defaults' in task_config:
                logger.info(f"Applying task defaults overrides for {task_name}: {task_defaults_overrides}")
                task_config['defaults'].update(task_defaults_overrides)
            else:
                logger.info(f"Creating defaults section for {task_name} with overrides: {task_defaults_overrides}")
                task_config['defaults'] = task_defaults_overrides
        
        return task_config

    def _validate_task_config(self, task_config: Dict[str, Any], task_name: str) -> None:
        """
        Warns about unknown, deprecated, or missing schema fields in a task configuration.
        
        Only validates configs that define an `entrypoint` or `runner_entrypoint`. Uses
        the module-level `TASK_CONFIG_SCHEMA` to determine allowed fields, format-specific
        allowed fields, deprecated fields, and format-specific required fields; logs
        warnings for unknown fields, deprecated fields, and missing required fields.
        
        Parameters:
            task_config (Dict[str, Any]): The task configuration to validate.
            task_name (str): The task name used in warning messages.
        """
        format_name = task_config.get("task_format")
        # Keep warnings limited to configs that define entrypoints (i.e., runnable tasks).
        if not task_config.get("entrypoint") and not task_config.get("runner_entrypoint"):
            return
        if not format_name:
            logger.warning("Task config '%s' missing task_format.", task_name)

        # Build the allowed field set from the centralized schema.
        allowed_fields = set(TASK_CONFIG_SCHEMA["field_descriptions"].keys())
        # Extend allowed fields for the declared task_format (if any).
        format_fields = TASK_CONFIG_SCHEMA["format_fields"].get(format_name)
        if format_fields:
            allowed_fields |= set(format_fields)

        # Report unknown keys once at load time.
        unknown_fields = sorted(set(task_config.keys()) - allowed_fields)
        if unknown_fields:
            logger.warning(
                "Task config '%s' has unknown fields: %s",
                task_name,
                ", ".join(unknown_fields),
            )

        # Report deprecated keys so they can be cleaned up.
        for key, message in TASK_CONFIG_SCHEMA["deprecated_fields"].items():
            if key in task_config:
                logger.warning("Task config '%s' uses '%s': %s", task_name, key, message)

        # Report missing format-required keys.
        required_fields = TASK_CONFIG_SCHEMA["required_fields"].get(format_name, set())
        missing_fields = [field for field in required_fields if field not in task_config]
        if missing_fields:
            logger.warning(
                "Task config '%s' missing required fields for format '%s': %s",
                task_name,
                format_name,
                ", ".join(missing_fields),
            )
    
    def list_available_tasks(self) -> list:
        """
        Collects names of task configurations found under the project's tasks directory.
        
        Searches recursively for files named `task.json` under the tasks root, skips any files located in `_template` directories, ignores files with invalid JSON, and omits entries that lack a `"name"` field or whose name equals `"template_task"`. Duplicate names are removed and the result is sorted.
        
        Returns:
            list: A sorted list of unique task names discovered.
        """
        tasks_root = self._get_tasks_root()

        if not tasks_root.exists():
            return []

        task_names = []
        for task_path in tasks_root.rglob("task.json"):
            if task_path.parent.name == "_template":
                continue
            try:
                with open(task_path, "r") as f:
                    task_config = json.load(f)
            except json.JSONDecodeError:
                continue
            name = task_config.get("name")
            if name and name != "template_task":
                task_names.append(name)

        return sorted(set(task_names))
    
    def expand_task_groups(self, tasks: list, central_config: Dict[str, Any]) -> list:
        """
        Expand task groups into individual task names.
        
        Args:
            tasks: List of task names and/or task group names
            central_config: Central configuration containing task_groups
            
        Returns:
            Expanded list of individual task names
        """
        task_groups = central_config.get('task_groups', {})
        expanded_tasks = []
        
        for task in tasks:
            if task in task_groups:
                # This is a task group, expand it
                group_tasks = task_groups[task].get('tasks', [])
                group_description = task_groups[task].get('description', task)
                logger.info(f"Expanding task group '{task}': {group_description}")
                logger.info(f"  Group contains tasks: {group_tasks}")
                expanded_tasks.extend(group_tasks)
            else:
                # This is an individual task, keep it as is
                expanded_tasks.append(task)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tasks = []
        for task in expanded_tasks:
            if task not in seen:
                seen.add(task)
                unique_tasks.append(task)
        
        return unique_tasks