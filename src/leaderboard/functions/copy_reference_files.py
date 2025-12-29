#!/usr/bin/env python3
"""
Copy reference files to the publish directory.
"""

import shutil
from pathlib import Path

from config_loader import load_config

def copy_reference_files(reference_dir: str, publish_dir: str) -> bool:
    """Copy reference files to the publish directory."""
    # Get paths
    reference_dir_path = Path(reference_dir)
    publish_dir_path = Path(publish_dir)
    
    # Create publish directory if it doesn't exist
    publish_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying reference files from: {reference_dir_path}")
    print(f"To publish directory: {publish_dir_path}")
    
    if not reference_dir_path.exists():
        print(f"❌ Reference directory not found: {reference_dir_path}")
        return False
    
    # Copy all files from reference directory to publish directory
    copied_files = []
    for file_path in reference_dir_path.iterdir():
        if file_path.is_file():
            target_path = publish_dir_path / file_path.name
            shutil.copy2(file_path, target_path)
            copied_files.append(file_path.name)
            print(f"  ✓ Copied {file_path.name}")
    
    if copied_files:
        print(f"\n✓ Successfully copied {len(copied_files)} files:")
        for file_name in copied_files:
            print(f"  - {file_name}")
    else:
        print("⚠️  No files found in reference directory")
    
    return True

def main():
    """Main function for standalone execution."""
    config = load_config()
    reference_dir = config["paths"]["reference_dir"]
    publish_dir = config["paths"]["publish_dir"]
    return copy_reference_files(reference_dir, publish_dir)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
