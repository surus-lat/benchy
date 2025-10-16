#!/usr/bin/env python3
"""
Model Testing Automation Script

This script automates the process of testing models to determine:
1. If they work with single GPU
2. If they work with two GPUs  
3. If they pass the testing suite
4. Generates appropriate config files for working models

Usage:
    python test_model_automation.py --models-file next_models.txt [--start-from N] [--max-models N] [--run-id ID]
"""

import os
import sys
import subprocess
import time
import json
import yaml
import argparse
import logging
import signal
import shutil
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Run ID generation functions (self-contained)
def generate_run_id(custom_run_id: str = None, prefix: str = "") -> str:
    """Generate a run ID for model testing."""
    if custom_run_id:
        return custom_run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}{timestamp}" if prefix else timestamp

def setup_run_directories(run_id: str):
    """Create necessary directories for a run."""
    os.makedirs(f"logs/{run_id}", exist_ok=True)
    os.makedirs(f"configs/testing/{run_id}", exist_ok=True)
    os.makedirs(f"outputs/benchmark_outputs/{run_id}", exist_ok=True)

# Setup logging
def setup_logging(run_id: str = None):
    """Setup logging for the automation script."""
    if run_id:
        # Use run_id-based directory structure
        log_dir = Path("logs") / run_id
    else:
        # Fallback to old structure
        log_dir = Path("logs").joinpath("model_testing")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if run_id:
        log_file = log_dir / "model_testing.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir.joinpath(f"model_testing_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

class ModelTester:
    """Main class for testing models and generating configurations."""
    
    def __init__(self, models_file: str, start_from: int = 0, max_models: Optional[int] = None, skip_downloads: bool = False, run_id: str = None):
        self.models_file = models_file
        self.start_from = start_from
        self.max_models = max_models
        self.skip_downloads = skip_downloads
        self.run_id = run_id
        self.logger = logging.getLogger(__name__)
        self.running_processes = []  # Track running processes for cleanup
        
        # Results tracking
        self.results = {
            'single_gpu_passed': [],
            'two_gpu_passed': [],
            'minimal_gpu_passed': [],
            'failed': [],
            'total_tested': 0,
            'start_time': datetime.now().isoformat(),
            'run_id': run_id
        }
        
        # Setup directories using run_id if available
        if run_id:
            setup_run_directories(run_id)
            self.config_dir = Path("configs/testing") / run_id
        else:
            self.config_dir = Path("configs/testing")
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Template configs
        self.single_gpu_template = Path("configs/templates/test-model_new.yaml")
        self.two_gpu_template = Path("configs/templates/test-model_two_cards.yaml")
        
        # Verify templates exist
        if not self.single_gpu_template.exists():
            raise FileNotFoundError(f"Single GPU template not found: {self.single_gpu_template}")
        if not self.two_gpu_template.exists():
            raise FileNotFoundError(f"Two GPU template not found: {self.two_gpu_template}")
    
    def cleanup_processes(self):
        """Clean up all running processes."""
        for process in self.running_processes:
            try:
                if process.poll() is None:  # Process is still running
                    self.logger.info(f"🛑 Terminating process {process.pid}")
                    process.terminate()
                    # Wait a bit for graceful termination
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"⚠️  Force killing process {process.pid}")
                        process.kill()
            except Exception as e:
                self.logger.warning(f"⚠️  Error cleaning up process: {e}")
        
        self.running_processes.clear()
    
    def estimate_model_size(self, model_name: str) -> Tuple[Optional[float], str]:
        """
        Estimate model size using multiple methods.
        
        Returns:
            Tuple of (estimated_size_gb, message)
        """
        try:
            # Method 1: Try to use huggingface_hub if available
            try:
                from huggingface_hub import model_info
                info = model_info(model_name)
                
                # Get total size from model info
                total_size_bytes = 0
                for sibling in info.siblings:
                    if hasattr(sibling, 'size') and sibling.size:
                        total_size_bytes += sibling.size
                
                if total_size_bytes > 0:
                    # Convert to GB and add 20% overhead for extraction
                    size_gb = (total_size_bytes / (1024**3)) * 1.2
                    return size_gb, f"Estimated size: {size_gb:.1f}GB (from HF Hub API)"
                    
            except ImportError:
                pass  # huggingface_hub not available
            except Exception:
                pass  # API call failed
            
            # Method 2: Estimate based on model name patterns
            estimated_size = self._estimate_size_from_name(model_name)
            if estimated_size:
                return estimated_size, f"Estimated size: {estimated_size:.1f}GB (from model name pattern)"
            
            # Method 3: Fallback - use conservative estimate
            return None, "Model size estimation not available - will use conservative 5GB minimum"
            
        except Exception as e:
            return None, f"Error estimating model size: {e}"
    
    def _estimate_size_from_name(self, model_name: str) -> Optional[float]:
        """Estimate model size based on common naming patterns."""
        name_lower = model_name.lower()
        
        # Common model size patterns
        if any(size in name_lower for size in ['1b', '1.5b', '1.3b']):
            return 2.5  # ~2GB + overhead
        elif any(size in name_lower for size in ['3b', '2.5b', '2.7b']):
            return 6.0  # ~5GB + overhead
        elif any(size in name_lower for size in ['7b', '6b', '8b']):
            return 15.0  # ~12GB + overhead
        elif any(size in name_lower for size in ['13b', '12b', '14b']):
            return 30.0  # ~25GB + overhead
        elif any(size in name_lower for size in ['20b', '30b', '40b']):
            return 80.0  # ~65GB + overhead
        elif any(size in name_lower for size in ['70b', '65b', '80b']):
            return 150.0  # ~130GB + overhead
        
        return None
    
    def get_model_config(self, model_name: str) -> Dict[str, any]:
        """
        Get model configuration from Hugging Face Hub.
        
        Returns:
            Dictionary with model configuration
        """
        try:
            # Try to use huggingface_hub if available
            try:
                from huggingface_hub import hf_hub_download
                import json
                
                # Download config.json
                config_path = hf_hub_download(
                    repo_id=model_name,
                    filename="config.json",
                    cache_dir=None,  # Use default cache
                    local_files_only=False
                )
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                return config
                
            except ImportError:
                pass  # huggingface_hub not available
            except Exception as e:
                self.logger.warning(f"Could not download config.json for {model_name}: {e}")
            
            # Fallback: try to read from local cache if model exists
            if self.model_exists(model_name):
                try:
                    hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
                    cache_path = Path(hf_cache) / "hub"
                    model_dir_name = f"models--{model_name.replace('/', '--')}"
                    model_cache_path = cache_path / model_dir_name
                    
                    # Find the latest revision
                    revisions = [d for d in model_cache_path.iterdir() if d.is_dir()]
                    if revisions:
                        latest_revision = max(revisions, key=lambda x: x.name)
                        config_path = latest_revision / "config.json"
                        
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            return config
                            
                except Exception as e:
                    self.logger.warning(f"Could not read local config.json for {model_name}: {e}")
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Error getting model config for {model_name}: {e}")
            return {}
    
    def is_multimodal(self, model_name: str) -> bool:
        """
        Check if a model is multimodal by looking for processor config files.
        
        Returns:
            True if model is multimodal, False otherwise
        """
        try:
            # Try to use huggingface_hub if available
            try:
                from huggingface_hub import hf_hub_download
                
                # Check for preprocessor_config.json or processor_config.json
                for config_file in ["preprocessor_config.json", "processor_config.json"]:
                    try:
                        config_path = hf_hub_download(
                            repo_id=model_name,
                            filename=config_file,
                            cache_dir=None,
                            local_files_only=False
                        )
                        if config_path:
                            self.logger.info(f"🔍 Found {config_file} - model is multimodal")
                            return True
                    except Exception:
                        continue  # File doesn't exist, try next one
                
                return False
                
            except ImportError:
                pass  # huggingface_hub not available
            except Exception as e:
                self.logger.warning(f"Could not check multimodal status for {model_name}: {e}")
            
            # Fallback: try to read from local cache if model exists
            if self.model_exists(model_name):
                try:
                    hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
                    cache_path = Path(hf_cache) / "hub"
                    model_dir_name = f"models--{model_name.replace('/', '--')}"
                    model_cache_path = cache_path / model_dir_name
                    
                    # Find the latest revision
                    revisions = [d for d in model_cache_path.iterdir() if d.is_dir()]
                    if revisions:
                        latest_revision = max(revisions, key=lambda x: x.name)
                        
                        # Check for processor config files
                        for config_file in ["preprocessor_config.json", "processor_config.json"]:
                            config_path = latest_revision / config_file
                            if config_path.exists():
                                self.logger.info(f"🔍 Found {config_file} in cache - model is multimodal")
                                return True
                                
                except Exception as e:
                    self.logger.warning(f"Could not check local multimodal status for {model_name}: {e}")
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking multimodal status for {model_name}: {e}")
            return False
    
    def get_optimal_vllm_params(self, model_name: str) -> Dict[str, any]:
        """
        Get optimal vLLM parameters based on model configuration.
        
        Returns:
            Dictionary with optimized vLLM parameters
        """
        config = self.get_model_config(model_name)
        params = {}
        
        # Get max_position_embeddings and adjust max_model_len
        max_position_embeddings = config.get('max_position_embeddings')
        if max_position_embeddings and max_position_embeddings < 8192:
            params['max_model_len'] = max_position_embeddings
            self.logger.info(f"📏 Using max_model_len={max_position_embeddings} from config (was 8192)")
        
        # Ensure max_num_batched_tokens is at least as large as max_model_len
        max_model_len = params.get('max_model_len', 8192)
        if max_model_len > 8192:
            params['max_num_batched_tokens'] = max_model_len
            self.logger.info(f"📏 Using max_num_batched_tokens={max_model_len} to match max_model_len")
        
        # Add trust_remote_code=True by default
        params['trust_remote_code'] = True
        
        # Check if model is multimodal and disable multimodal features
        is_multimodal_model = self.is_multimodal(model_name)
        if is_multimodal_model:
            params['limit_mm_per_prompt'] = '{"images": 0, "audios": 0}'
            self.logger.info(f"🔧 Disabled multimodal features for {model_name} (images=0, audios=0)")
        
        # Check for Mistral models and add special parameters
        model_type = config.get('model_type', '').lower()
        if 'mistral' in model_type or 'mistral' in model_name.lower():
            params.update({
                'tokenizer_mode': 'mistral',
                'config_format': 'mistral', 
                'load_format': 'mistral',
                'tool_call_parser': 'mistral',
                'enable_auto_tool_choice': True
            })
            self.logger.info(f"🔧 Added Mistral-specific parameters for {model_name}")
        
        return params
    
    def get_model_metadata(self, model_name: str) -> Dict[str, any]:
        """
        Get model metadata including max context length and multimodal status.
        
        Returns:
            Dictionary with model metadata
        """
        config = self.get_model_config(model_name)
        is_multimodal_model = self.is_multimodal(model_name)
        
        # Get max context length
        max_position_embeddings = config.get('max_position_embeddings')
        if max_position_embeddings:
            max_context_length = max_position_embeddings
        else:
            # Fallback for models that use different field names
            max_context_length = config.get('n_positions', config.get('max_sequence_length', 8192))
        
        metadata = {
            'max_context_length': max_context_length,
            'is_multimodal': is_multimodal_model,
            'model_type': config.get('model_type', 'unknown'),
            'detected_from_config': bool(config)
        }
        
        self.logger.info(f"📊 Model metadata: max_context={max_context_length}, multimodal={is_multimodal_model}")
        
        return metadata
    
    def check_disk_space(self, path: str = None) -> Tuple[bool, str]:
        """
        Check available disk space.
        
        Returns:
            Tuple of (has_space, message)
        """
        try:
            if path is None:
                # Check the Hugging Face cache directory (prioritize HF_HOME over HF_CACHE)
                hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
                path = hf_cache
            
            # Get disk usage
            total, used, free = shutil.disk_usage(path)
            
            # Convert to GB
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            
            # Define minimum free space (5GB)
            min_free_gb = 5.0
            
            if free_gb < min_free_gb:
                return False, f"Low disk space: {free_gb:.1f}GB free (minimum: {min_free_gb}GB)"
            else:
                return True, f"Disk space OK: {free_gb:.1f}GB free of {total_gb:.1f}GB total"
                
        except Exception as e:
            return False, f"Could not check disk space: {e}"
    
    def get_available_space_gb(self) -> float:
        """Get available disk space in GB."""
        try:
            hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
            _, _, free = shutil.disk_usage(hf_cache)
            return free / (1024**3)
        except Exception:
            return 0.0
    
    def save_progress(self):
        """Save current progress to a recovery file."""
        try:
            if self.run_id:
                progress_file = Path("logs") / self.run_id / "model_testing_progress.json"
            else:
                progress_file = Path("logs/model_testing_progress.json")
            
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'results': self.results,
                'current_model_index': getattr(self, 'current_model_index', 0),
                'total_models': getattr(self, 'total_models', 0),
                'disk_space_check': self.check_disk_space()[1],
                'run_id': self.run_id
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            self.logger.info(f"💾 Progress saved to: {progress_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save progress: {e}")
    
    def load_models(self) -> List[str]:
        """Load models from the models file."""
        models = []
        
        with open(self.models_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Remove inline comments (everything after #)
                if '#' in line:
                    line = line.split('#')[0].strip()
                
                # Skip if line is empty after removing comments
                if not line:
                    continue
                
                # Extract model name from URL or use as-is
                if line.startswith('https://huggingface.co/'):
                    model_name = line.replace('https://huggingface.co/', '')
                else:
                    model_name = line
                
                models.append(model_name)
        
        # Apply start_from and max_models filters
        if self.start_from > 0:
            models = models[self.start_from:]
            self.logger.info(f"Starting from model {self.start_from + 1}")
        
        if self.max_models:
            models = models[:self.max_models]
            self.logger.info(f"Limiting to {self.max_models} models")
        
        self.logger.info(f"Loaded {len(models)} models to test")
        return models
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model already exists in Hugging Face cache."""
        try:
            # Get HF cache directory
            hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
            cache_path = Path(hf_cache) / "hub"
            
            # Find model directory (models are stored with -- replaced by __)
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            model_cache_path = cache_path / model_dir_name
            
            if model_cache_path.exists():
                # Check if it has the required files (at least one .bin or .safetensors file)
                for file_path in model_cache_path.rglob("*"):
                    if file_path.is_file() and (file_path.suffix in ['.bin', '.safetensors'] or 'model' in file_path.name.lower()):
                        return True
            return False
        except Exception:
            return False
    
    def download_model(self, model_name: str) -> bool:
        """Download a model using huggingface-cli with progress visibility."""
        # Check if model already exists
        if self.model_exists(model_name):
            self.logger.info(f"✅ Model already exists: {model_name}")
            return True
        
        # Estimate model size and check disk space
        estimated_size, size_msg = self.estimate_model_size(model_name)
        self.logger.info(f"📏 {size_msg}")
        
        # Check available disk space
        has_space, space_msg = self.check_disk_space()
        self.logger.info(f"💾 {space_msg}")
        
        # Check if we have enough space for this specific model
        if estimated_size:
            available_gb = self.get_available_space_gb()
            if available_gb < estimated_size:
                self.logger.error(f"❌ Insufficient disk space for {model_name}")
                self.logger.error(f"   Required: {estimated_size:.1f}GB, Available: {available_gb:.1f}GB")
                self.logger.error(f"   Please free up {estimated_size - available_gb:.1f}GB or clean up old models")
                return False
            else:
                self.logger.info(f"✅ Sufficient space: {available_gb:.1f}GB available for {estimated_size:.1f}GB model")
        else:
            # Fallback to simple check if we can't estimate size
            if not has_space:
                self.logger.error(f"❌ Insufficient disk space for {model_name}")
                self.logger.error(f"   Please free up space or clean up old models")
                return False
        
        self.logger.info(f"📥 Downloading model: {model_name}")
        self.logger.info(f"   This may take several minutes for large models...")
        
        try:
            # Use huggingface-cli to download with progress
            cmd = ["huggingface-cli", "download", model_name]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track process for cleanup
            self.running_processes.append(process)
            
            # Stream output with progress indicators
            start_time = time.time()
            last_progress_time = start_time
            last_heartbeat_time = start_time
            
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    break
                
                current_time = time.time()
                
                # Read output line by line
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            # Log progress every 30 seconds or on important messages
                            if (current_time - last_progress_time > 30 or 
                                any(keyword in line.lower() for keyword in ['downloading', 'progress', 'mb', 'gb', 'eta', 'resume', 'checkpoint'])):
                                self.logger.info(f"   📊 {line}")
                                last_progress_time = current_time
                except:
                    pass
                
                # Heartbeat every 2 minutes to show it's still working
                if current_time - last_heartbeat_time > 120:
                    elapsed = int(current_time - start_time)
                    self.logger.info(f"   ⏳ Still downloading... ({elapsed//60}m {elapsed%60}s elapsed)")
                    
                    # Check disk space during download
                    has_space, space_msg = self.check_disk_space()
                    if not has_space:
                        self.logger.error(f"❌ {space_msg}")
                        self.logger.error(f"   Download interrupted due to insufficient disk space")
                        process.terminate()
                        return False
                    else:
                        self.logger.info(f"   💾 {space_msg}")
                    
                    last_heartbeat_time = current_time
                
                # Check timeout
                if current_time - start_time > 5400:  # 90 minutes
                    process.terminate()
                    self.logger.error(f"⏰ Timeout downloading {model_name} (90 minutes)")
                    return False
                
                time.sleep(1)  # Small delay to prevent busy waiting
            
            # Get final result
            return_code = process.returncode
            # Remove from tracking
            if process in self.running_processes:
                self.running_processes.remove(process)
                
            if return_code == 0:
                self.logger.info(f"✅ Successfully downloaded: {model_name}")
                return True
            else:
                self.logger.error(f"❌ Failed to download {model_name} (exit code: {return_code})")
                return False
                
        except Exception as e:
            self.logger.error(f"💥 Error downloading {model_name}: {e}")
            # Clean up process if it exists
            if 'process' in locals() and process in self.running_processes:
                self.running_processes.remove(process)
            return False
    
    def cleanup_model(self, model_name: str):
        """Clean up model files from Hugging Face cache."""
        self.logger.info(f"🧹 Cleaning up model: {model_name}")
        
        try:
            # Get HF cache directory
            hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
            cache_path = Path(hf_cache) / "hub"
            
            # Find model directory (models are stored with -- replaced by __)
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            model_cache_path = cache_path / model_dir_name
            
            if model_cache_path.exists():
                shutil.rmtree(model_cache_path)
                self.logger.info(f"🗑️  Removed cache for: {model_name}")
            else:
                self.logger.info(f"ℹ️  No cache found for: {model_name}")
                
        except Exception as e:
            self.logger.warning(f"⚠️  Could not clean up {model_name}: {e}")
    
    def test_model_with_config(self, model_name: str, config_path: str, test_type: str) -> Tuple[bool, str]:
        """Test a model with a specific configuration."""
        self.logger.info(f"🧪 Testing {model_name} with {test_type} config")
        
        try:
            # Run the test command
            cmd = [
                "python", "eval.py",
                "--config", config_path,
                "--test",
                "--no-log-samples"
            ]
            
            # Add run_id if available
            if self.run_id:
                cmd.extend(["--run-id", self.run_id])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for vLLM startup
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ {test_type} test PASSED for {model_name}")
                return True, ""
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                self.logger.error(f"❌ {test_type} test FAILED for {model_name}: {error_msg}")
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Test timeout (10 minutes)"
            self.logger.error(f"⏰ {test_type} test TIMEOUT for {model_name}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Test error: {e}"
            self.logger.error(f"💥 {test_type} test ERROR for {model_name}: {e}")
            return False, error_msg
    
    def test_model_suite(self, model_name: str, config_path: str) -> Tuple[bool, str]:
        """Test model with the full benchmarking suite (limited to 10 samples)."""
        self.logger.info(f"🎯 Testing {model_name} with full suite (limited)")
        
        try:
            cmd = [
                "python", "eval.py",
                "--config", config_path,
                "--limit", "10",
                "--log-samples"
            ]
            
            # Add run_id if available
            if self.run_id:
                cmd.extend(["--run-id", self.run_id])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout for full suite
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Full suite test PASSED for {model_name}")
                return True, ""
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                self.logger.warning(f"⚠️  Full suite test FAILED for {model_name}: {error_msg}")
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Full suite test timeout (30 minutes)"
            self.logger.warning(f"⏰ Full suite test TIMEOUT for {model_name}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Full suite test error: {e}"
            self.logger.warning(f"💥 Full suite test ERROR for {model_name}: {e}")
            return False, error_msg
    
    def generate_config(self, model_name: str, template_path: str, gpu_type: str) -> str:
        """Generate a configuration file for a working model."""
        # Create safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', model_name.replace('/', '_'))
        config_filename = f"{safe_name}_{gpu_type}.yaml"
        config_path = self.config_dir / config_filename
        
        # Load template
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model name
        config['model']['name'] = model_name
        
        # Get optimal vLLM parameters for this model
        optimal_params = self.get_optimal_vllm_params(model_name)
        
        # Update vLLM configuration with optimal parameters
        if 'vllm' in config and 'overrides' in config['vllm']:
            config['vllm']['overrides'].update(optimal_params)
        else:
            if 'vllm' not in config:
                config['vllm'] = {}
            config['vllm']['overrides'] = optimal_params
        
        # Add model metadata section
        metadata = self.get_model_metadata(model_name)
        config['metadata'] = {
            'max_context_length': metadata['max_context_length'],
            'is_multimodal': metadata['is_multimodal'],
            'model_type': metadata['model_type'],
            'detected_from_config': metadata['detected_from_config'],
            'generated_at': datetime.now().isoformat(),
            'gpu_type': gpu_type
        }
        
        # Write new config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        self.logger.info(f"📝 Generated config: {config_path}")
        return str(config_path)
    
    def test_single_model(self, model_name: str) -> Dict:
        """Test a single model and return results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔄 Testing model: {model_name}")
        self.logger.info(f"{'='*60}")
        
        result = {
            'model_name': model_name,
            'single_gpu_worked': False,
            'two_gpu_worked': False,
            'minimal_gpu_worked': False,
            'suite_worked': False,
            'config_path': None,
            'gpu_type': None,
            'error': None
        }
        
        # Step 1: Download model (unless skipped)
        if not self.skip_downloads:
            if not self.download_model(model_name):
                result['error'] = "Download failed"
                self.cleanup_model(model_name)
                return result
        else:
            # Check if model exists when skipping downloads
            if not self.model_exists(model_name):
                result['error'] = "Model not found (downloads skipped)"
                return result
            else:
                self.logger.info(f"✅ Model exists (skipping download): {model_name}")
        
        # Step 2: Test with single GPU
        single_gpu_config = self.generate_config(model_name, self.single_gpu_template, "single")
        single_worked, single_error = self.test_model_with_config(model_name, single_gpu_config, "single GPU")
        
        if single_worked:
            result['single_gpu_worked'] = True
            result['gpu_type'] = 'single'
            result['config_path'] = single_gpu_config
            
            # Test full suite
            suite_worked, suite_error = self.test_model_suite(model_name, single_gpu_config)
            result['suite_worked'] = suite_worked
            if not suite_worked:
                result['error'] = f"Suite failed: {suite_error}"
            
            # Clean up two GPU config
            two_gpu_config = self.generate_config(model_name, self.two_gpu_template, "two")
            os.remove(two_gpu_config)
            
            return result
        
        # Step 3: Test with two GPUs
        two_gpu_config = self.generate_config(model_name, self.two_gpu_template, "two")
        two_worked, two_error = self.test_model_with_config(model_name, two_gpu_config, "two GPU")
        
        if two_worked:
            result['two_gpu_worked'] = True
            result['gpu_type'] = 'two'
            result['config_path'] = two_gpu_config
            
            # Test full suite
            suite_worked, suite_error = self.test_model_suite(model_name, two_gpu_config)
            result['suite_worked'] = suite_worked
            if not suite_worked:
                result['error'] = f"Suite failed: {suite_error}"
            
            # Clean up single GPU config
            os.remove(single_gpu_config)
            
            return result
        
        # Step 4: Try minimal fallback configuration
        self.logger.info(f"🔄 Trying minimal fallback config for {model_name} (conservative settings)...")
        minimal_gpu_config = self.generate_config(model_name, "configs/templates/test-model_minimal.yaml", "minimal")
        minimal_worked, minimal_error = self.test_model_with_config(model_name, minimal_gpu_config, "minimal GPU")
        
        if minimal_worked:
            result['minimal_gpu_worked'] = True
            result['gpu_type'] = 'minimal'
            result['config_path'] = minimal_gpu_config
            
            # Test full suite
            suite_worked, suite_error = self.test_model_suite(model_name, minimal_gpu_config)
            result['suite_worked'] = suite_worked
            if not suite_worked:
                result['error'] = f"Suite failed: {suite_error}"
            
            # Clean up other config files
            for config_path in [single_gpu_config, two_gpu_config]:
                if os.path.exists(config_path):
                    os.remove(config_path)
            
            return result
        
        # Step 5: Model failed completely
        result['error'] = f"Single GPU: {single_error}; Two GPU: {two_error}; Minimal: {minimal_error}"
        self.cleanup_model(model_name)
        
        # Clean up config files
        for config_path in [single_gpu_config, two_gpu_config, minimal_gpu_config]:
            if os.path.exists(config_path):
                os.remove(config_path)
        
        return result
    
    def run_tests(self):
        """Run tests for all models."""
        models = self.load_models()
        
        if not models:
            self.logger.error("No models to test!")
            return
        
        self.logger.info(f"🚀 Starting automated testing of {len(models)} models")
        self.logger.info(f"📁 Configs will be saved to: {self.config_dir}")
        self.logger.info(f"📊 Results will be tracked and summarized")
        
        # Check which models already exist
        existing_models = []
        for model in models:
            if self.model_exists(model):
                existing_models.append(model)
        
        if existing_models:
            self.logger.info(f"✅ {len(existing_models)} models already downloaded (will skip download)")
            for model in existing_models:
                self.logger.info(f"   - {model}")
        
        new_models = [m for m in models if m not in existing_models]
        if new_models:
            self.logger.info(f"📥 {len(new_models)} models need to be downloaded")
            for model in new_models:
                self.logger.info(f"   - {model}")
        
        self.logger.info("")
        
        # Store model info for progress tracking
        self.total_models = len(models)
        
        for i, model_name in enumerate(models, 1):
            self.logger.info(f"\n📋 Progress: {i}/{len(models)}")
            self.current_model_index = i - 1  # 0-based index
            
            # Save progress before each model
            self.save_progress()
            
            try:
                result = self.test_single_model(model_name)
                self.results['total_tested'] += 1
                
                # Categorize results
                if result['single_gpu_worked']:
                    self.results['single_gpu_passed'].append(result)
                elif result['two_gpu_worked']:
                    self.results['two_gpu_passed'].append(result)
                elif result['minimal_gpu_worked']:
                    self.results['minimal_gpu_passed'].append(result)
                else:
                    self.results['failed'].append(result)
                
                # Log result
                if result['single_gpu_worked'] or result['two_gpu_worked'] or result['minimal_gpu_worked']:
                    suite_status = "✅" if result['suite_worked'] else "⚠️"
                    self.logger.info(f"🎉 {model_name} WORKED with {result['gpu_type']} GPU {suite_status}")
                else:
                    self.logger.info(f"💥 {model_name} FAILED: {result['error']}")
                
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                
                # Check if it's a disk space issue
                if "No space left on device" in str(e) or "ENOSPC" in str(e):
                    error_msg = f"Disk space error: {e}"
                    self.logger.error(f"💾 {error_msg}")
                    self.logger.error(f"   Stopping testing due to insufficient disk space")
                    
                    # Save progress before stopping
                    self.save_progress()
                    
                    # Generate emergency summary
                    self.generate_emergency_summary("Disk space exhausted")
                    sys.exit(1)
                
                self.logger.error(f"💥 Unexpected error testing {model_name}: {e}")
                self.results['failed'].append({
                    'model_name': model_name,
                    'error': error_msg,
                    'single_gpu_worked': False,
                    'two_gpu_worked': False,
                    'suite_worked': False
                })
                self.results['total_tested'] += 1
        
        # Generate final summary
        self.generate_summary()
    
    def generate_emergency_summary(self, reason: str):
        """Generate emergency summary when testing is interrupted."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("🚨 EMERGENCY STOP - TESTING INTERRUPTED")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Reason: {reason}")
        
        total = self.results['total_tested']
        single_passed = len(self.results['single_gpu_passed'])
        two_passed = len(self.results['two_gpu_passed'])
        failed = len(self.results['failed'])
        
        self.logger.info(f"Models tested before interruption: {total}")
        self.logger.info(f"Single GPU passed: {single_passed}")
        self.logger.info(f"Two GPU passed: {two_passed}")
        self.logger.info(f"Failed: {failed}")
        
        # Save emergency results
        if self.run_id:
            emergency_file = Path("logs") / self.run_id / "model_testing_emergency_results.json"
            progress_file_path = f"logs/{self.run_id}/model_testing_progress.json"
        else:
            emergency_file = Path("logs/model_testing_emergency_results.json")
            progress_file_path = "logs/model_testing_progress.json"
        
        emergency_data = {
            'interruption_reason': reason,
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'disk_space_check': self.check_disk_space()[1],
            'run_id': self.run_id
        }
        
        with open(emergency_file, 'w') as f:
            json.dump(emergency_data, f, indent=2)
        
        self.logger.info(f"📄 Emergency results saved to: {emergency_file}")
        self.logger.info(f"💾 Progress file: {progress_file_path}")
        
        if single_passed + two_passed > 0:
            self.logger.info(f"\n✅ {single_passed + two_passed} models were successfully tested before interruption")
            self.logger.info("   Generated configs are available in configs/testing/")
        
        self.logger.info(f"\n💡 To resume testing after freeing space:")
        self.logger.info(f"   python test_model_automation.py --models-file next_models.txt --start-from {self.current_model_index + 1}")
    
    def generate_summary(self):
        """Generate final summary of all tests."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("📊 FINAL TESTING SUMMARY")
        self.logger.info(f"{'='*80}")
        
        total = self.results['total_tested']
        single_passed = len(self.results['single_gpu_passed'])
        two_passed = len(self.results['two_gpu_passed'])
        minimal_passed = len(self.results['minimal_gpu_passed'])
        failed = len(self.results['failed'])
        
        self.logger.info(f"Total models tested: {total}")
        self.logger.info(f"Single GPU passed: {single_passed}")
        self.logger.info(f"Two GPU passed: {two_passed}")
        self.logger.info(f"Minimal fallback passed: {minimal_passed}")
        self.logger.info(f"Failed: {failed}")
        
        # Single GPU results
        if self.results['single_gpu_passed']:
            self.logger.info(f"\n✅ SINGLE GPU MODELS ({single_passed}):")
            for result in self.results['single_gpu_passed']:
                suite_status = "✅" if result['suite_worked'] else "⚠️"
                self.logger.info(f"  ✓ {result['model_name']} {suite_status}")
        
        # Two GPU results
        if self.results['two_gpu_passed']:
            self.logger.info(f"\n✅ TWO GPU MODELS ({two_passed}):")
            for result in self.results['two_gpu_passed']:
                suite_status = "✅" if result['suite_worked'] else "⚠️"
                self.logger.info(f"  ✓ {result['model_name']} {suite_status}")
        
        # Minimal fallback results
        if self.results['minimal_gpu_passed']:
            self.logger.info(f"\n🔧 MINIMAL FALLBACK MODELS ({minimal_passed}):")
            for result in self.results['minimal_gpu_passed']:
                suite_status = "✅" if result['suite_worked'] else "⚠️"
                self.logger.info(f"  ✓ {result['model_name']} {suite_status}")
        
        # Failed results
        if self.results['failed']:
            self.logger.info(f"\n❌ FAILED MODELS ({failed}):")
            for result in self.results['failed']:
                self.logger.info(f"  ✗ {result['model_name']}: {result['error']}")
        
        # Save detailed results to JSON
        if self.run_id:
            results_file = Path("logs") / self.run_id / "model_testing_results.json"
        else:
            results_file = Path("logs") / f"model_testing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.results['end_time'] = datetime.now().isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"\n📄 Detailed results saved to: {results_file}")
        self.logger.info(f"📁 Generated configs saved to: {self.config_dir}")
        
        # Exit code based on results
        if failed == 0:
            self.logger.info("\n🎉 ALL MODELS PASSED!")
            return 0
        else:
            self.logger.info(f"\n⚠️  {failed} models failed, but {single_passed + two_passed} passed!")
            return 1


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n🛑 Received signal {signum}. Cleaning up...")
    
    # Try to clean up any running processes
    if hasattr(main, 'tester') and main.tester:
        main.tester.cleanup_processes()
    
    logger.info("👋 Exiting gracefully")
    sys.exit(1)

def main():
    """Main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Automated model testing for benchy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model_automation.py --models-file next_models.txt
  python test_model_automation.py --models-file next_models.txt --start-from 5
  python test_model_automation.py --models-file next_models.txt --max-models 10
  python test_model_automation.py --models-file next_models.txt --skip-downloads
  python test_model_automation.py --models-file next_models.txt --recover
  python test_model_automation.py --models-file next_models.txt --run-id my_test_run
  nohup python test_model_automation.py --models-file next_models.txt > testing.log 2>&1 &
        """
    )
    
    parser.add_argument(
        '--models-file',
        required=True,
        help='Path to file containing list of models to test'
    )
    
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start testing from this model number (0-based)'
    )
    
    parser.add_argument(
        '--max-models',
        type=int,
        help='Maximum number of models to test'
    )
    
    parser.add_argument(
        '--skip-downloads',
        action='store_true',
        help='Skip downloading models, only test existing ones'
    )
    
    parser.add_argument(
        '--recover',
        action='store_true',
        help='Recover from previous interrupted run using progress file'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID for organizing outputs (default: auto-generated with MODEL_TEST prefix)'
    )
    
    args = parser.parse_args()
    
    # Generate run ID with MODEL_TEST prefix
    run_id = generate_run_id(
        custom_run_id=args.run_id,
        prefix="MODEL_TEST_"
    )
    
    # Setup logging with run_id
    log_file = setup_logging(run_id)
    logger = logging.getLogger(__name__)
    
    # Create PID file for easy process management
    pid_file = Path("logs/model_testing.pid")
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    
    logger.info("🤖 Starting Model Testing Automation")
    logger.info(f"📋 Models file: {args.models_file}")
    logger.info(f"🆔 Run ID: {run_id}")
    logger.info(f"📊 Log file: {log_file}")
    logger.info(f"🆔 PID file: {pid_file} (PID: {os.getpid()})")
    logger.info(f"💡 To stop: kill {os.getpid()} or Ctrl+C")
    
    if args.start_from > 0:
        logger.info(f"🎯 Starting from model: {args.start_from + 1}")
    
    if args.max_models:
        logger.info(f"🔢 Max models: {args.max_models}")
    
    if args.skip_downloads:
        logger.info("⏭️  Downloads will be skipped - only testing existing models")
    
    if args.recover:
        logger.info("🔄 Recovery mode enabled - will load previous progress")
    
    # Verify we're in the right directory
    if not Path("eval.py").exists():
        logger.error("❌ eval.py not found. Please run from benchy root directory.")
        sys.exit(1)
    
    # Activate virtual environment
    venv_path = Path(".venv/bin/activate")
    if venv_path.exists():
        logger.info("✅ Virtual environment found")
    else:
        logger.warning("⚠️  Virtual environment not found at .venv/")
    
    # Run tests
    try:
        tester = ModelTester(
            models_file=args.models_file,
            start_from=args.start_from,
            max_models=args.max_models,
            skip_downloads=args.skip_downloads,
            run_id=run_id
        )
        
        # Handle recovery mode
        if args.recover:
            # Try to find progress file in run_id directory first, then fallback to old location
            progress_file = None
            if run_id:
                progress_file = Path("logs") / run_id / "model_testing_progress.json"
                if not progress_file.exists():
                    progress_file = Path("logs/model_testing_progress.json")
            else:
                progress_file = Path("logs/model_testing_progress.json")
            
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                    
                    logger.info(f"📂 Loaded previous progress from: {progress_file}")
                    logger.info(f"   Previous run: {progress_data.get('timestamp', 'unknown')}")
                    logger.info(f"   Previous run ID: {progress_data.get('run_id', 'unknown')}")
                    logger.info(f"   Models tested: {progress_data.get('current_model_index', 0)}")
                    logger.info(f"   Disk space: {progress_data.get('disk_space_check', 'unknown')}")
                    
                    # Restore results
                    tester.results = progress_data.get('results', tester.results)
                    
                    # Ask user if they want to continue from where they left off
                    logger.info(f"💡 To continue from where you left off:")
                    logger.info(f"   python test_model_automation.py --models-file {args.models_file} --start-from {progress_data.get('current_model_index', 0) + 1}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Could not load progress file: {e}")
            else:
                logger.info("ℹ️  No previous progress file found")
        
        # Store tester reference for signal handler
        main.tester = tester
        
        exit_code = tester.run_tests()
        
        # Clean up PID file on normal completion
        if pid_file.exists():
            pid_file.unlink()
        
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Testing interrupted by user")
        if 'tester' in locals():
            tester.cleanup_processes()
        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        if 'tester' in locals():
            tester.cleanup_processes()
        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
