"""GPU configuration management for benchy system."""

import os
import logging
from typing import Dict, Any, Optional, List
import subprocess

logger = logging.getLogger(__name__)


class GPUConfigManager:
    """Manages GPU configuration for vLLM server and evaluation tasks."""
    
    def __init__(self, gpu_config: Dict[str, Any]):
        """
        Initialize GPU configuration manager.
        
        Args:
            gpu_config: GPU configuration dictionary from config.yaml
        """
        self.gpu_config = gpu_config
        self.vllm_devices = gpu_config.get('vllm', {}).get('devices', '3')
        self.task_devices = gpu_config.get('tasks', {}).get('devices', '2')
        self.validation = gpu_config.get('validation', {})
        
        # Validate configuration
        if self.validation.get('check_gpu_availability', True):
            self._validate_gpu_configuration()
    
    def _validate_gpu_configuration(self) -> None:
        """Validate GPU configuration and check for conflicts."""
        logger.info("Validating GPU configuration...")
        
        # Check for GPU overlap if not allowed
        if not self.validation.get('allow_overlap', False):
            vllm_gpus = self._parse_gpu_list(self.vllm_devices)
            task_gpus = self._parse_gpu_list(self.task_devices)
            
            overlap = set(vllm_gpus) & set(task_gpus)
            if overlap:
                raise ValueError(
                    f"GPU overlap detected between vLLM ({self.vllm_devices}) and tasks ({self.task_devices}). "
                    f"Overlapping GPUs: {list(overlap)}. Set 'allow_overlap: true' to override."
                )
        
        # Check GPU availability
        available_gpus = self._get_available_gpus()
        if available_gpus is not None:
            all_used_gpus = set(self._parse_gpu_list(self.vllm_devices)) | set(self._parse_gpu_list(self.task_devices))
            unavailable_gpus = all_used_gpus - set(available_gpus)
            if unavailable_gpus:
                logger.warning(f"Some specified GPUs may not be available: {list(unavailable_gpus)}")
                logger.info(f"Available GPUs: {available_gpus}")
        
        logger.info(f"GPU configuration validated successfully")
        logger.info(f"vLLM will use GPUs: {self.vllm_devices}")
        logger.info(f"Tasks will use GPUs: {self.task_devices if self.task_devices else 'CPU only'}")
    
    def _parse_gpu_list(self, gpu_string: str) -> List[int]:
        """Parse GPU string into list of integers."""
        if not gpu_string or gpu_string.strip() == "":
            return []
        
        gpus = []
        for part in gpu_string.split(','):
            part = part.strip()
            if part:
                try:
                    gpus.append(int(part))
                except ValueError:
                    logger.warning(f"Invalid GPU ID: {part}")
        return gpus
    
    def _get_available_gpus(self) -> Optional[List[int]]:
        """Get list of available GPU IDs using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if 'GPU' in line:
                        # Extract GPU ID from line like "GPU 0: NVIDIA GeForce RTX 4090"
                        parts = line.split(':')
                        if len(parts) > 0:
                            gpu_part = parts[0].strip()
                            if 'GPU' in gpu_part:
                                gpu_id = gpu_part.replace('GPU', '').strip()
                                try:
                                    gpus.append(int(gpu_id))
                                except ValueError:
                                    pass
                return gpus
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Could not detect available GPUs: {e}")
        
        return None
    
    def get_vllm_cuda_devices(self) -> str:
        """Get CUDA devices string for vLLM server."""
        return self.vllm_devices
    
    def get_task_cuda_devices(self) -> str:
        """Get CUDA devices string for evaluation tasks."""
        return self.task_devices
    
    def get_vllm_env_vars(self) -> Dict[str, str]:
        """Get environment variables for vLLM server."""
        return {
            'CUDA_VISIBLE_DEVICES': self.vllm_devices
        }
    
    def get_task_env_vars(self) -> Dict[str, str]:
        """Get environment variables for evaluation tasks."""
        env_vars = {}
        
        if self.task_devices:
            # Use specified GPU for tasks
            env_vars['CUDA_VISIBLE_DEVICES'] = self.task_devices
            # Clear CPU-only settings
            env_vars['PYTORCH_CUDA_ALLOC_CONF'] = ''
        else:
            # Use CPU only for tasks (current default behavior)
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
            env_vars['PYTORCH_CUDA_ALLOC_CONF'] = ''
        
        return env_vars
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current GPU configuration."""
        return {
            'vllm_devices': self.vllm_devices,
            'task_devices': self.task_devices,
            'vllm_gpu_count': len(self._parse_gpu_list(self.vllm_devices)),
            'task_gpu_count': len(self._parse_gpu_list(self.task_devices)),
            'validation': self.validation
        }


def load_gpu_config(central_config: Dict[str, Any]) -> GPUConfigManager:
    """
    Load GPU configuration from central config.
    
    Args:
        central_config: Central configuration dictionary
        
    Returns:
        GPUConfigManager instance
    """
    gpu_config = central_config.get('gpu_config', {})
    
    # Set defaults if not specified
    if 'vllm' not in gpu_config:
        gpu_config['vllm'] = {'devices': '3'}
    if 'tasks' not in gpu_config:
        gpu_config['tasks'] = {'devices': '2'}
    if 'validation' not in gpu_config:
        gpu_config['validation'] = {
            'check_gpu_availability': True,
            'allow_overlap': False
        }
    
    return GPUConfigManager(gpu_config)
