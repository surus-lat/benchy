"""Simple configuration manager for merging model and provider configs."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


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
        """Return the root directory that contains task definitions."""
        return Path(__file__).resolve().parent / "tasks"

    def _load_task_config_from_tasks_root(self, task_name: str) -> Dict[str, Any]:
        tasks_root = self._get_tasks_root()

        if not tasks_root.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_root}")

        metadata_path = tasks_root / task_name / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f) or {}
            if isinstance(metadata, dict):
                subtasks = metadata.get("subtasks") or {}
                if isinstance(subtasks, dict):
                    task_list = list(subtasks.keys())
                elif isinstance(subtasks, list):
                    task_list = [task for task in subtasks if isinstance(task, str)]
                else:
                    task_list = []

                output_cfg = metadata.get("output")
                if not isinstance(output_cfg, dict):
                    output_cfg = {"subdirectory": task_name}

                return {
                    "name": metadata.get("name", task_name),
                    "display_name": metadata.get("display_name", task_name.replace("_", " ").title()),
                    "description": metadata.get("description", ""),
                    "tasks": task_list,
                    "defaults": {},
                    "prompts": {},
                    "task_configs": {},
                    "output": output_cfg,
                    "capability_requirements": metadata.get("capability_requirements", {}),
                    "metrics_manifest": metadata.get("metrics_manifest", []),
                }

        raise FileNotFoundError(f"Task config not found for: {task_name}")

    def get_task_config(self, task_name: str, task_defaults_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a task configuration file and apply any overrides.
        
        Args:
            task_name: Name of the task config
            task_defaults_overrides: Optional dictionary to override task defaults
            
        Returns:
            Task configuration dictionary with overrides applied
        """
        task_config = self._load_task_config_from_tasks_root(task_name)
        
        # Apply task defaults overrides if provided
        if task_defaults_overrides:
            if 'defaults' in task_config:
                logger.info(f"Applying task defaults overrides for {task_name}: {task_defaults_overrides}")
                task_config['defaults'].update(task_defaults_overrides)
            else:
                logger.info(f"Creating defaults section for {task_name} with overrides: {task_defaults_overrides}")
                task_config['defaults'] = task_defaults_overrides
        
        return task_config
    
    def list_available_tasks(self) -> list:
        """List all available handler-based task groups."""
        from .tasks.registry import list_handler_task_groups

        return list_handler_task_groups()
    
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
