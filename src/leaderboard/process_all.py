#!/usr/bin/env python3
"""
Main script to orchestrate the entire processing pipeline.
This runs all steps: copy raw data, parse results, and generate final table.
"""

import subprocess
import sys
from pathlib import Path
import yaml

# Add the src directory to the path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from leaderboard.functions.parse_model_results import parse_model_results
from leaderboard.functions.generate_leaderboard_table import generate_leaderboard_table
from leaderboard.functions.copy_reference_files import copy_reference_files

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Find config.yaml relative to the project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/leaderboard to benchy root
        config_path = project_root / "configs" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_function(function, description: str, *args) -> bool:
    """Run a function and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = function(*args)
        return result
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Main function to run the entire processing pipeline."""
    print("🚀 Starting LM-Evaluation-Harness Results Processing Pipeline")
    print("=" * 70)
    
    # Load configuration
    try:
        config = load_config()
        print(f"✓ Configuration loaded from configs/config.yaml")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    # Check if required directories exist
    benchmark_outputs = Path(config["paths"]["benchmark_outputs"])
    if not benchmark_outputs.exists():
        print(f"❌ Benchmark outputs directory not found: {benchmark_outputs}")
        print("   Please check the path in config.yaml")
        return False
    
    print(f"✓ Benchmark outputs directory found: {benchmark_outputs}")
    
    # Step 1: Parse model results
    success = run_function(
        parse_model_results, 
        "Step 1: Parsing model results and generating summaries",
        config["paths"]["benchmark_outputs"],
        config["paths"]["publish_dir"]
    )
    if not success:
        print("❌ Pipeline failed at Step 1")
        return False
    
    # Step 2: Generate leaderboard table
    success = run_function(
        generate_leaderboard_table,
        "Step 2: Generating leaderboard table",
        config["paths"]["publish_dir"]
    )
    if not success:
        print("❌ Pipeline failed at Step 2")
        return False
    
    # Step 3: Copy reference files to publish directory
    success = run_function(
        copy_reference_files,
        "Step 3: Copying reference files to publish directory",
        config["paths"]["reference_dir"],
        config["paths"]["publish_dir"]
    )
    if not success:
        print("❌ Pipeline failed at Step 3")
        return False
    
    # Pipeline completed successfully
    print(f"\n{'='*70}")
    print("🎉 Pipeline completed successfully!")
    print(f"{'='*70}")
    
    # Show final results
    publish_dir = Path(config["paths"]["publish_dir"])
    
    if publish_dir.exists():
        print(f"\n📁 Results available in: {publish_dir}")
        print("   Generated files:")
        for file in sorted(publish_dir.glob("*")):
            size = file.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"     - {file.name} ({size_str})")
    
    print(f"\n✨ Ready for upload to Hugging Face dataset!")
    print(f"   Dataset: {config['datasets']['results']}")
    print(f"   To upload: python upload_to_hf.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
