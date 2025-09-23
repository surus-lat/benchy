"""Simple configuration manager for merging model and provider configs."""

import yaml
from pathlib import Path
from typing import Dict, Any
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
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """
        Load a task configuration file.
        
        Args:
            task_name: Name of the task config (without .yaml)
            
        Returns:
            Task configuration dictionary
        """
        task_path = self.configs_dir / "tasks" / f"{task_name}.yaml"
        
        if not task_path.exists():
            raise FileNotFoundError(f"Task config not found: {task_path}")
        
        with open(task_path, 'r') as f:
            return yaml.safe_load(f)
    
    def list_available_tasks(self) -> list:
        """List all available task configurations."""
        tasks_dir = self.configs_dir / "tasks"
        
        if not tasks_dir.exists():
            return []
        
        return [f.stem for f in tasks_dir.glob("*.yaml")]
