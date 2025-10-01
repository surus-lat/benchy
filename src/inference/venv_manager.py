"""Virtual environment management for different vLLM versions."""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)


class VLLMVenvManager:
    """Manages virtual environments for different vLLM versions."""
    
    def __init__(self, base_venv_dir: str = None):
        if base_venv_dir is None:
            # Default to venvs directory in the project root
            # Find the project root by looking for pyproject.toml
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                if (current_dir / "pyproject.toml").exists():
                    break
                current_dir = current_dir.parent
            base_venv_dir = str(current_dir / "venvs")
        
        self.base_venv_dir = Path(base_venv_dir)
        self.base_venv_dir.mkdir(exist_ok=True)
    
    def get_venv_path(self, vllm_version: str) -> str:
        """
        Get the path to a virtual environment for a specific vLLM version.
        
        Args:
            vllm_version: vLLM version (e.g., "0.8.0", "0.10.2")
            
        Returns:
            Path to the virtual environment
        """
        # Clean version string for directory name
        clean_version = vllm_version.replace(".", "_")
        venv_path = self.base_venv_dir / f"vllm_{clean_version}"
        return str(venv_path)
    
    def create_venv(self, vllm_version: str, force_recreate: bool = False, transformers_version: str = None) -> str:
        """
        Create a virtual environment for a specific vLLM version.
        
        Args:
            vllm_version: vLLM version to install
            force_recreate: Whether to recreate if it already exists
            transformers_version: Optional transformers version to install
            
        Returns:
            Path to the created virtual environment
        """
        venv_path = self.get_venv_path(vllm_version)
        venv_path_obj = Path(venv_path)
        
        if venv_path_obj.exists() and not force_recreate:
            logger.info(f"Virtual environment already exists: {venv_path}")
            print(f"✅ Virtual environment already exists: {venv_path}")
            return venv_path
        
        if force_recreate and venv_path_obj.exists():
            logger.info(f"Removing existing virtual environment: {venv_path}")
            import shutil
            shutil.rmtree(venv_path)
        
        logger.info(f"Creating virtual environment for vLLM {vllm_version} at {venv_path}")
        print(f"🔧 Creating virtual environment for vLLM {vllm_version}...")
        
        try:
            # Create virtual environment using uv
            print(f"   Running: uv venv {venv_path}")
            result = subprocess.run(
                ["uv", "venv", venv_path],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Created virtual environment: {venv_path}")
            print(f"✅ Virtual environment created: {venv_path}")
            
            # Install vLLM using uv pip install in the virtual environment
            self._install_vllm_with_uv(venv_path, vllm_version, transformers_version)
            
            print(f"🎉 vLLM {vllm_version} environment ready at: {venv_path}")
            return venv_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            print(f"❌ Failed to create virtual environment: {e}")
            if e.stdout:
                print(f"   stdout: {e.stdout}")
            if e.stderr:
                print(f"   stderr: {e.stderr}")
            raise
    
    def _install_vllm_with_uv(self, venv_path: str, vllm_version: str, transformers_version: str = None) -> None:
        """Install vLLM and optionally transformers in the virtual environment using uv pip install."""
        logger.info(f"Installing vLLM {vllm_version} in {venv_path}")
        print(f"📦 Installing vLLM {vllm_version}...")
        
        try:
            # Install transformers first if specific version is requested
            if transformers_version:
                logger.info(f"Installing transformers {transformers_version} first")
                print(f"📦 Installing transformers {transformers_version}...")
                print(f"   Running: uv pip install --python {venv_path}/bin/python transformers=={transformers_version}")
                transformers_result = subprocess.run(
                    ["uv", "pip", "install", "--python", f"{venv_path}/bin/python", f"transformers=={transformers_version}"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✅ Successfully installed transformers {transformers_version}")
            
            # Install vLLM
            print(f"   Running: uv pip install --python {venv_path}/bin/python vllm=={vllm_version}")
            result = subprocess.run(
                ["uv", "pip", "install", "--python", f"{venv_path}/bin/python", f"vllm=={vllm_version}"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Successfully installed vLLM {vllm_version}")
            print(f"✅ Successfully installed vLLM {vllm_version}")
            
            # Verify installation
            print(f"   Verifying installation...")
            venv_python = str(Path(venv_path) / "bin" / "python")
            try:
                verify_result = subprocess.run(
                    [venv_python, "-c", "import vllm; print(f'vLLM version: {vllm.__version__}')"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"   {verify_result.stdout.strip()}")
                
                # Also verify transformers version if specified
                if transformers_version:
                    transformers_verify = subprocess.run(
                        [venv_python, "-c", "import transformers; print(f'Transformers version: {transformers.__version__}')"],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"   {transformers_verify.stdout.strip()}")
                    
            except subprocess.CalledProcessError as verify_error:
                print(f"   ⚠️  Warning: Could not verify installation: {verify_error}")
                print(f"   stderr: {verify_error.stderr}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            print(f"❌ Failed to install packages: {e}")
            if e.stdout:
                print(f"   stdout: {e.stdout}")
            if e.stderr:
                print(f"   stderr: {e.stderr}")
            raise
    
    def ensure_venv_exists(self, vllm_version: str, transformers_version: str = None) -> str:
        """
        Ensure a virtual environment exists for the specified vLLM version.
        Create it if it doesn't exist.
        
        Args:
            vllm_version: vLLM version
            transformers_version: Optional transformers version
            
        Returns:
            Path to the virtual environment
        """
        venv_path = self.get_venv_path(vllm_version)
        
        if not Path(venv_path).exists():
            logger.info(f"Virtual environment not found for vLLM {vllm_version}, creating automatically...")
            print(f"🚀 Virtual environment not found for vLLM {vllm_version}")
            print(f"   Creating automatically...")
            try:
                return self.create_venv(vllm_version, transformers_version=transformers_version)
            except Exception as e:
                logger.error(f"Failed to create virtual environment for vLLM {vllm_version}: {e}")
                logger.info("Falling back to main project environment")
                print(f"⚠️  Failed to create virtual environment for vLLM {vllm_version}")
                print(f"   Falling back to main project environment")
                # Find the project root by looking for pyproject.toml
                current_dir = Path(__file__).parent
                while current_dir != current_dir.parent:
                    if (current_dir / "pyproject.toml").exists():
                        break
                    current_dir = current_dir.parent
                return str(current_dir / ".venv")
        
        return venv_path
    
    def list_available_versions(self) -> list:
        """List all available vLLM versions with virtual environments."""
        versions = []
        for venv_dir in self.base_venv_dir.iterdir():
            if venv_dir.is_dir() and venv_dir.name.startswith("vllm_"):
                version = venv_dir.name.replace("vllm_", "").replace("_", ".")
                versions.append(version)
        return sorted(versions)
    
    def get_venv_info(self, vllm_version: str) -> Dict[str, Any]:
        """
        Get information about a virtual environment.
        
        Args:
            vllm_version: vLLM version
            
        Returns:
            Dictionary with venv information
        """
        venv_path = self.get_venv_path(vllm_version)
        venv_path_obj = Path(venv_path)
        
        info = {
            "version": vllm_version,
            "path": venv_path,
            "exists": venv_path_obj.exists(),
            "size": 0
        }
        
        if venv_path_obj.exists():
            # Calculate directory size
            total_size = sum(f.stat().st_size for f in venv_path_obj.rglob('*') if f.is_file())
            info["size"] = total_size
        
        return info


def create_vllm_venv(vllm_version: str, base_venv_dir: str = None) -> str:
    """
    Convenience function to create a vLLM virtual environment.
    
    Args:
        vllm_version: vLLM version to install
        base_venv_dir: Base directory for virtual environments (defaults to project root/venvs)
        
    Returns:
        Path to the created virtual environment
    """
    manager = VLLMVenvManager(base_venv_dir)
    return manager.create_venv(vllm_version)


def get_vllm_venv_path(vllm_version: str, base_venv_dir: str = None, transformers_version: str = None) -> str:
    """
    Get the path to a vLLM virtual environment, creating it if necessary.
    
    Args:
        vllm_version: vLLM version
        base_venv_dir: Base directory for virtual environments (defaults to project root/venvs)
        transformers_version: Optional transformers version
        
    Returns:
        Path to the virtual environment
    """
    manager = VLLMVenvManager(base_venv_dir)
    return manager.ensure_venv_exists(vllm_version, transformers_version)
