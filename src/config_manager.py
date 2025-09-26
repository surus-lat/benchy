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
        
        # Handle vLLM config merging
        if 'vllm' in model_config:
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
        
        return model_config
    
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
    
    def get_task_config(self, task_name: str, task_defaults_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a task configuration file and apply any overrides.
        
        Args:
            task_name: Name of the task config (without .yaml)
            task_defaults_overrides: Optional dictionary to override task defaults
            
        Returns:
            Task configuration dictionary with overrides applied
        """
        task_path = self.configs_dir / "tasks" / f"{task_name}.yaml"
        
        if not task_path.exists():
            raise FileNotFoundError(f"Task config not found: {task_path}")
        
        with open(task_path, 'r') as f:
            task_config = yaml.safe_load(f)
        
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
        """List all available task configurations."""
        tasks_dir = self.configs_dir / "tasks"
        
        if not tasks_dir.exists():
            return []
        
        return [f.stem for f in tasks_dir.glob("*.yaml")]
