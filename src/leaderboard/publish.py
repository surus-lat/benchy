#!/usr/bin/env python3
"""
Upload the contents of the publish directory to Hugging Face dataset.
"""

import os
import yaml
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Import the function module
from .functions.upload_to_hf import upload_to_hf

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Find config.yaml relative to the project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/leaderboard to benchy root
        config_path = project_root / "configs" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload the contents of the publish directory to Hugging Face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.leaderboard.publish                    # Upload from config publish_dir
  python -m src.leaderboard.publish --path custom/publish  # Upload from custom directory
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        help='Custom directory path to upload (overrides config publish_dir path)'
    )
    
    return parser.parse_args()

def load_env_file():
    """Load environment variables from .env file."""
    # Look for .env file in project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up from src/leaderboard to benchy root
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file.absolute()}")
        print("   Please create a .env file with your HF_TOKEN")

def main():
    """Main function to upload to Hugging Face."""
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    load_env_file()
    
    # Load configuration
    config = load_config()
    
    # Determine the publish directory path based on custom directory or config
    if args.path:
        publish_dir = args.path
        print(f"🎯 Using custom directory: {publish_dir}")
    else:
        publish_dir = config["paths"]["publish_dir"]
        print(f"📁 Using config directory: {publish_dir}")
    
    dataset_name = config["datasets"]["results"]
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("   Please set HF_TOKEN in your .env file or environment")
        return False
    
    # Call the function from the functions module
    return upload_to_hf(publish_dir, dataset_name)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
