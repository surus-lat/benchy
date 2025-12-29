#!/usr/bin/env python3
"""
Script to upload the processed results to Hugging Face datasets.
This uploads the publish directory contents to the configured dataset.
"""

import os
from pathlib import Path
from typing import Dict
from config_loader import load_config

def upload_to_huggingface(publish_dir: Path, dataset_name: str) -> bool:
    """Upload files to Hugging Face dataset."""
    try:
        from huggingface_hub import HfApi, create_repo
        
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id=dataset_name, repo_type="dataset", exist_ok=True)
            print(f"✓ Repository {dataset_name} is ready")
        except Exception as e:
            print(f"ℹ️  Repository {dataset_name} already exists or error: {e}")
        
        # Upload all files in publish directory (including subdirectories)
        uploaded_files = []
        for file_path in publish_dir.rglob("*"):
            if file_path.is_file():
                # Calculate relative path from publish directory
                relative_path = file_path.relative_to(publish_dir)
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=str(relative_path),
                        repo_id=dataset_name,
                        repo_type="dataset"
                    )
                    uploaded_files.append(str(relative_path))
                    print(f"  ✓ Uploaded {relative_path}")
                except Exception as e:
                    print(f"  ✗ Failed to upload {relative_path}: {e}")
                    return False
        
        print(f"\n🎉 Successfully uploaded {len(uploaded_files)} files to {dataset_name}")
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        return False

def upload_to_hf(publish_dir: str, dataset_name: str) -> bool:
    """Upload results to Hugging Face dataset."""
    # Set up paths
    publish_dir_path = Path(publish_dir)
    
    print(f"🚀 Uploading results to Hugging Face dataset: {dataset_name}")
    print(f"📁 Source directory: {publish_dir_path}")
    
    if not publish_dir_path.exists():
        print("❌ Publish directory not found! Run the processing pipeline first.")
        return False
    
    # Check if files exist (including subdirectories)
    files = list(publish_dir_path.rglob("*"))
    file_list = [f for f in files if f.is_file()]
    if not file_list:
        print("❌ No files found in publish directory!")
        return False
    
    print(f"📄 Found {len(file_list)} files to upload:")
    for file in file_list:
        relative_path = file.relative_to(publish_dir_path)
        size = file.stat().st_size
        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
        print(f"  - {relative_path} ({size_str})")
    
    # Upload files
    success = upload_to_huggingface(publish_dir_path, dataset_name)
    
    if success:
        print(f"\n✨ Upload completed! View your dataset at:")
        print(f"   https://huggingface.co/datasets/{dataset_name}")
    
    return success

def main():
    """Main function for standalone execution."""
    import yaml
    config = load_config()
    publish_dir = config["paths"]["publish_dir"]
    dataset_name = config["datasets"]["results"]
    return upload_to_hf(publish_dir, dataset_name)

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
