#!/usr/bin/env python3
"""
Download existing HF summaries, merge new results on top, and push the merged dataset.

Usage:
    # Process a new run then merge+push
    python -m src.leaderboard.merge_and_publish --run-id <run_id>

    # Skip processing (outputs/publish/ already exists): just merge+push
    python -m src.leaderboard.merge_and_publish --skip-process
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_src_dir = Path(__file__).resolve().parents[1]  # src/
_project_root = _src_dir.parent

from src.config_loader import load_config
from src.leaderboard.functions.generate_leaderboard_table import generate_leaderboard_table
from src.leaderboard.functions.generate_reference_files import generate_reference_files
from src.leaderboard.functions.copy_reference_files import copy_reference_files
from src.leaderboard.functions.upload_to_hf import upload_to_hf


def merge_summaries(old: dict, new: dict) -> dict:
    """Merge two all_model_summaries dicts. New entries overwrite old for the same key."""
    return {**old, **new}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge new results with existing HF summaries and publish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.leaderboard.merge_and_publish --run-id batch_20241201_143022
  python -m src.leaderboard.merge_and_publish --skip-process
        """,
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to process before merging+pushing",
    )

    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip processing step (outputs/publish/ already exists)",
    )

    return parser.parse_args()


def _load_env():
    """Load .env from project root."""
    env_file = _project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"No .env file found at {env_file.absolute()}")
        print("Please create a .env file with your HF_TOKEN")


def _download_hf_summaries(dataset_name: str) -> dict:
    """Download all_model_summaries.json from HF dataset. Returns empty dict if not found."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        print(f"Downloading summaries from {dataset_name} ...")
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename="summaries/all_model_summaries.json",
            repo_type="dataset",
        )
        with open(local_path, "r") as f:
            data = json.load(f)
        print(f"Downloaded {len(data)} existing model summaries from HF")
        return data
    except Exception as e:
        # EntryNotFoundError (404) or RepositoryNotFoundError — treat as empty
        err_str = str(e).lower()
        if "404" in err_str or "not found" in err_str or "entry" in err_str:
            print("No existing summaries found on HF — starting fresh")
            return {}
        # Re-raise unexpected errors
        raise


def main() -> bool:
    """Main entry point for merge-and-publish pipeline."""
    args = parse_args()

    # Validate args
    if not args.skip_process and not args.run_id:
        print("Error: provide --run-id <run_id> or --skip-process")
        sys.exit(1)

    # Load environment
    _load_env()

    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables")
        print("Please set HF_TOKEN in your .env file or environment")
        return False

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False

    publish_dir = Path(config["paths"]["publish_dir"])
    reference_dir = config["paths"]["reference_dir"]
    dataset_name = config["datasets"]["results"]

    # Step 1: (Optional) Process new run
    if args.run_id and not args.skip_process:
        print(f"\nProcessing run ID: {args.run_id}")
        result = subprocess.run(
            [sys.executable, "-m", "src.leaderboard.process_all", args.run_id],
            cwd=str(_project_root),
        )
        if result.returncode != 0:
            print(f"Error: process_all.py failed with return code {result.returncode}")
            return False

    # Step 2: Download existing HF summaries
    print("\nFetching existing summaries from Hugging Face ...")
    hf_summaries = _download_hf_summaries(dataset_name)

    # Step 3: Load local summaries
    local_summaries_path = publish_dir / "summaries" / "all_model_summaries.json"
    if local_summaries_path.exists():
        with open(local_summaries_path, "r") as f:
            local_summaries = json.load(f)
        print(f"Loaded {len(local_summaries)} local model summaries from {local_summaries_path}")
    else:
        local_summaries = {}
        print(f"No local summaries found at {local_summaries_path} — using empty dict")

    # Step 4: Merge (local wins on conflict)
    merged = merge_summaries(hf_summaries, local_summaries)
    print(f"Merged summaries: {len(merged)} models total")

    # Step 5: Save merged summaries
    summaries_dir = publish_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    with open(summaries_dir / "all_model_summaries.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged all_model_summaries.json ({len(merged)} models)")

    for model_key, model_data in merged.items():
        individual_path = summaries_dir / f"{model_key}_summary.json"
        with open(individual_path, "w") as f:
            json.dump(model_data, f, indent=2)

    print(f"Wrote {len(merged)} individual summary files")

    # Step 6: Regenerate leaderboard table
    print("\nRegenerating leaderboard table ...")
    success = generate_leaderboard_table(str(publish_dir))
    if not success:
        print("Error: generate_leaderboard_table failed")
        return False

    # Step 7: Regenerate reference files
    print("\nRegenerating reference files ...")
    success = generate_reference_files(reference_dir)
    if not success:
        print("Error: generate_reference_files failed")
        return False

    success = copy_reference_files(reference_dir, str(publish_dir))
    if not success:
        print("Error: copy_reference_files failed")
        return False

    # Step 8: Upload to HF
    print(f"\nUploading to Hugging Face dataset: {dataset_name} ...")
    success = upload_to_hf(str(publish_dir), dataset_name)
    if not success:
        print("Error: upload_to_hf failed")
        return False

    print("\nmerge_and_publish completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
