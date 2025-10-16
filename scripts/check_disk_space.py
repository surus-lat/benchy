#!/usr/bin/env python3
"""
Disk space checker and cleanup utility for model testing automation.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

def get_disk_usage(path: str) -> Tuple[float, float, float]:
    """Get disk usage in GB."""
    total, used, free = shutil.disk_usage(path)
    return total / (1024**3), used / (1024**3), free / (1024**3)

def find_large_models(cache_path: str, min_size_gb: float = 1.0) -> List[Tuple[str, float]]:
    """Find large models in the cache."""
    large_models = []
    cache_dir = Path(cache_path) / "hub"
    
    if not cache_dir.exists():
        return large_models
    
    for model_dir in cache_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("models--"):
            try:
                total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_gb = total_size / (1024**3)
                
                if size_gb >= min_size_gb:
                    model_name = model_dir.name.replace("models--", "").replace("--", "/")
                    large_models.append((model_name, size_gb))
            except Exception:
                continue
    
    return sorted(large_models, key=lambda x: x[1], reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Check disk space and find large models")
    parser.add_argument("--path", help="Path to check (default: HF cache directory)")
    parser.add_argument("--min-size", type=float, default=1.0, help="Minimum model size in GB to show")
    parser.add_argument("--cleanup", action="store_true", help="Interactive cleanup of large models")
    
    args = parser.parse_args()
    
    # Determine path to check
    if args.path:
        check_path = args.path
    else:
        check_path = os.environ.get('HF_HOME') or os.environ.get('HF_CACHE') or os.path.expanduser('~/.cache/huggingface')
    
    print(f"💾 Checking disk space for: {check_path}")
    
    # Check disk space
    total_gb, used_gb, free_gb = get_disk_usage(check_path)
    print(f"📊 Disk Usage:")
    print(f"   Total: {total_gb:.1f} GB")
    print(f"   Used:  {used_gb:.1f} GB ({used_gb/total_gb*100:.1f}%)")
    print(f"   Free:  {free_gb:.1f} GB ({free_gb/total_gb*100:.1f}%)")
    
    # Check if space is low
    if free_gb < 5.0:
        print(f"⚠️  WARNING: Low disk space! Only {free_gb:.1f} GB free")
    else:
        print(f"✅ Disk space OK")
    
    # Find large models
    print(f"\n🔍 Large models (>{args.min_size} GB):")
    large_models = find_large_models(check_path, args.min_size)
    
    if not large_models:
        print("   No large models found")
    else:
        total_model_size = sum(size for _, size in large_models)
        print(f"   Found {len(large_models)} models totaling {total_model_size:.1f} GB")
        print()
        
        for i, (model_name, size_gb) in enumerate(large_models, 1):
            print(f"   {i:2d}. {model_name:<50} {size_gb:6.1f} GB")
    
    # Interactive cleanup
    if args.cleanup and large_models:
        print(f"\n🧹 Interactive cleanup:")
        print("Enter model numbers to delete (comma-separated, or 'all' for all, 'q' to quit):")
        
        while True:
            try:
                choice = input("> ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == 'all':
                    # Delete all large models
                    cache_dir = Path(check_path) / "hub"
                    deleted_size = 0
                    
                    for model_name, size_gb in large_models:
                        model_dir_name = f"models--{model_name.replace('/', '--')}"
                        model_path = cache_dir / model_dir_name
                        
                        if model_path.exists():
                            shutil.rmtree(model_path)
                            deleted_size += size_gb
                            print(f"   🗑️  Deleted {model_name} ({size_gb:.1f} GB)")
                    
                    print(f"✅ Deleted {len(large_models)} models, freed {deleted_size:.1f} GB")
                    break
                    
                else:
                    # Parse model numbers
                    indices = [int(x.strip()) for x in choice.split(',')]
                    cache_dir = Path(check_path) / "hub"
                    deleted_size = 0
                    
                    for idx in indices:
                        if 1 <= idx <= len(large_models):
                            model_name, size_gb = large_models[idx - 1]
                            model_dir_name = f"models--{model_name.replace('/', '--')}"
                            model_path = cache_dir / model_dir_name
                            
                            if model_path.exists():
                                shutil.rmtree(model_path)
                                deleted_size += size_gb
                                print(f"   🗑️  Deleted {model_name} ({size_gb:.1f} GB)")
                    
                    print(f"✅ Freed {deleted_size:.1f} GB")
                    break
                    
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Use numbers, 'all', or 'q'")
                continue

if __name__ == "__main__":
    main()
