#!/usr/bin/env python3
"""
Optional utility script for managing vLLM virtual environments.
This is only needed for manual management - the system creates environments automatically.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add the src directory to the path so we can import our modules
# Since this script is in scripts/, we need to go up one level to find src/
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.inference.venv_manager import VLLMVenvManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Optional utility for managing vLLM virtual environments",
        epilog="Note: The system automatically creates environments when needed. This script is only for manual management."
    )
    parser.add_argument(
        "action",
        choices=["create", "list", "info"],
        help="Action to perform: create, list, or info"
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="vLLM version (required for create and info actions)"
    )
    parser.add_argument(
        "--base-dir",
        default=str(repo_root / "venvs"),
        help="Base directory for virtual environments"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate the virtual environment if it already exists"
    )
    
    args = parser.parse_args()
    
    manager = VLLMVenvManager(args.base_dir)
    
    if args.action == "list":
        logger.info("Available vLLM versions with virtual environments:")
        versions = manager.list_available_versions()
        if versions:
            for version in versions:
                info = manager.get_venv_info(version)
                size_mb = info["size"] / (1024 * 1024)
                status = "✓" if info["exists"] else "✗"
                print(f"  {status} vLLM {version} - {info['path']} ({size_mb:.1f} MB)")
        else:
            print("  No virtual environments found")
            print("  (Environments are created automatically when needed)")
        return
    
    if not args.version:
        logger.error("Version is required for create and info actions")
        sys.exit(1)
    
    if args.action == "create":
        try:
            logger.info(f"Creating vLLM {args.version} virtual environment...")
            venv_path = manager.create_venv(args.version, force_recreate=args.force)
            logger.info(f"✅ Successfully created virtual environment at: {venv_path}")
            
            # Show some info about the created environment
            info = manager.get_venv_info(args.version)
            size_mb = info["size"] / (1024 * 1024)
            logger.info(f"Environment size: {size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    
    elif args.action == "info":
        info = manager.get_venv_info(args.version)
        print(f"vLLM {args.version} Environment Info:")
        print(f"  Path: {info['path']}")
        print(f"  Exists: {'Yes' if info['exists'] else 'No'}")
        if info['exists']:
            size_mb = info["size"] / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
