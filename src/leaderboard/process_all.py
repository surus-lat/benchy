#!/usr/bin/env python3
"""
Main script to orchestrate the entire processing pipeline.
This runs all steps: copy raw data, parse results, and generate final table.

Usage:
    python -m src.leaderboard.process_all <run_id>
    
Where run_id is the specific run directory under outputs/benchmark_outputs/
"""

import sys
import argparse
from pathlib import Path
import argparse

# Add the src directory to the path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from leaderboard.functions.parse_model_results import parse_model_results
from leaderboard.functions.generate_leaderboard_table import generate_leaderboard_table
from leaderboard.functions.copy_reference_files import copy_reference_files
from leaderboard.functions.generate_reference_files import generate_reference_files
from config_loader import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Benchy results and generate leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_all.py                                    # Process all results in benchmark_outputs/
  python process_all.py batch_20241201_143022             # Process specific run_id results
  python process_all.py my_experiment_001                 # Process custom run_id results
  python process_all.py --input-path custom/input --output-path custom/output  # Use custom paths
  python process_all.py --run-id exp_001 --input-path custom/input --output-path custom/output  # Custom paths with run_id
        """
    )
    
    parser.add_argument(
        'run_id',
        nargs='?',  # Optional positional argument
        help='Run ID to process (looks in benchmark_outputs/run_id/). If not provided, processes all results in benchmark_outputs/'
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        help='Custom input path for benchmark results (overrides config benchmark_outputs path)'
    )
    
    parser.add_argument(
        '--output-path', 
        type=str,
        help='Custom output path for published results (overrides config publish_dir path)'
    )
    
    return parser.parse_args()

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
    # Parse command line arguments
    args = parse_args()
    
    print("🚀 Starting Benchy Results Processing Pipeline")
    print("=" * 70)
    
    # Load configuration
    try:
        config = load_config()
        print(f"✓ Configuration loaded from configs/config.yaml")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    # Determine the benchmark outputs path based on custom input path or config
    if args.input_path:
        base_benchmark_outputs = Path(args.input_path)
        print(f"🎯 Using custom input path: {base_benchmark_outputs}")
    else:
        base_benchmark_outputs = Path(config["paths"]["benchmark_outputs"])
        print(f"📁 Using config input path: {base_benchmark_outputs}")
    
    if args.run_id:
        # Use specific run_id subdirectory
        benchmark_outputs = base_benchmark_outputs / args.run_id
        print(f"🎯 Processing specific run ID: {args.run_id}")
        print(f"   Looking in: {benchmark_outputs}")
    else:
        # Use base directory (backward compatibility)
        benchmark_outputs = base_benchmark_outputs
        print(f"📁 Processing all results in base directory")
        print(f"   Looking in: {benchmark_outputs}")
    
    # Check if required directories exist
    if not benchmark_outputs.exists():
        print(f"❌ Benchmark outputs directory not found: {benchmark_outputs}")
        if args.run_id:
            print(f"   Run ID '{args.run_id}' not found in {base_benchmark_outputs}")
            print(f"   Available run IDs:")
            if base_benchmark_outputs.exists():
                for subdir in sorted(base_benchmark_outputs.iterdir()):
                    if subdir.is_dir():
                        print(f"     - {subdir.name}")
            else:
                print(f"     Base directory {base_benchmark_outputs} does not exist")
        else:
            print("   Please check the path in config.yaml")
        return False
    
    print(f"✓ Benchmark outputs directory found: {benchmark_outputs}")
    
    # Determine the output path based on custom output path or config
    if args.output_path:
        publish_dir = args.output_path
        print(f"🎯 Using custom output path: {publish_dir}")
    else:
        publish_dir = config["paths"]["publish_dir"]
        print(f"📁 Using config output path: {publish_dir}")
    
    # Step 1: Parse model results
    success = run_function(
        parse_model_results, 
        "Step 1: Parsing model results and generating summaries",
        str(benchmark_outputs),  # Use the determined path (with or without run_id)
        publish_dir
    )
    if not success:
        print("❌ Pipeline failed at Step 1")
        return False
    
    # Step 2: Generate leaderboard table
    success = run_function(
        generate_leaderboard_table,
        "Step 2: Generating leaderboard table",
        publish_dir
    )
    if not success:
        print("❌ Pipeline failed at Step 2")
        return False
    
    # Step 3: Generate reference files from task configs
    success = run_function(
        generate_reference_files,
        "Step 3: Generating reference files from task configs",
        config["paths"]["reference_dir"]
    )
    if not success:
        print("❌ Pipeline failed at Step 3")
        return False

    # Step 4: Copy reference files to publish directory
    success = run_function(
        copy_reference_files,
        "Step 4: Copying reference files to publish directory",
        config["paths"]["reference_dir"],
        publish_dir
    )
    if not success:
        print("❌ Pipeline failed at Step 4")
        return False
    
    # Pipeline completed successfully
    print(f"\n{'='*70}")
    print("🎉 Pipeline completed successfully!")
    print(f"{'='*70}")
    
    # Show final results
    publish_dir_path = Path(publish_dir)
    
    if publish_dir_path.exists():
        print(f"\n📁 Results available in: {publish_dir_path}")
        if args.run_id:
            print(f"   Processed run ID: {args.run_id}")
        if args.input_path:
            print(f"   Used custom input path: {args.input_path}")
        if args.output_path:
            print(f"   Used custom output path: {args.output_path}")
        print("   Generated files:")
        for file in sorted(publish_dir_path.glob("*")):
            size = file.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"     - {file.name} ({size_str})")
    
    print(f"\n✨ Ready for upload to Hugging Face dataset!")
    print(f"   Dataset: {config['datasets']['results']}")
    print(f"   To upload: python -m src.leaderboard.publish --path {publish_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
