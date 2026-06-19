#!/usr/bin/env python3
"""
Merge new benchmark results with the existing HuggingFace dataset and publish.

Workflow:
  1. (Optional) Process a new benchmark run via process_all.py
  2. Download existing summaries/all_model_summaries.json from HF
  3. Load locally-generated summaries/all_model_summaries.json
  4. Merge: HF data is baseline; local data overwrites same-key entries
  5. Regenerate leaderboard_table.json/.csv and reference files
  6. Upload everything to HF

Usage:
  python -m src.leaderboard.merge_and_publish --run-id <run_id>
  python -m src.leaderboard.merge_and_publish --skip-process
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_src_dir = Path(__file__).resolve().parents[1]

from src.config_loader import load_config
from src.leaderboard.functions.generate_leaderboard_table import generate_leaderboard_table
from src.leaderboard.functions.upload_to_hf import upload_to_hf


def merge_summaries(old: dict, new: dict) -> dict:
    """Merge two all_model_summaries dicts. New entries overwrite old for the same key."""
    return {**old, **new}


def download_hf_summaries(dataset_name: str, publish_dir: Path) -> dict:
    """Download summaries/all_model_summaries.json from the HF dataset.

    Returns an empty dict if the file does not exist on HF yet.
    """
    try:
        from huggingface_hub import hf_hub_download
        try:
            local_path = hf_hub_download(
                repo_id=dataset_name,
                filename="summaries/all_model_summaries.json",
                repo_type="dataset",
                local_dir=str(publish_dir / "_hf_download"),
            )
            with open(local_path, "r") as f:
                data = json.load(f)
            print(f"  ✓ Downloaded {len(data)} existing model entries from HF")
            return data
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            if "404" in err_str or "not found" in err_str.lower() or "EntryNotFoundError" in err_type:
                print("  ℹ  No existing summaries on HF yet — starting fresh")
                return {}
            raise
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)


def load_local_summaries(publish_dir: Path) -> dict:
    """Load locally-generated all_model_summaries.json if it exists."""
    summaries_file = publish_dir / "summaries" / "all_model_summaries.json"
    if not summaries_file.exists():
        return {}
    with open(summaries_file, "r") as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data)} locally-processed model entries")
    return data


def save_summaries(merged: dict, publish_dir: Path) -> None:
    """Persist the merged summaries dict back to outputs/publish/."""
    summaries_dir = publish_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    all_file = summaries_dir / "all_model_summaries.json"
    with open(all_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  ✓ Saved merged summaries ({len(merged)} models) → {all_file}")

    for model_key, model_data in merged.items():
        individual_file = summaries_dir / f"{model_key}_summary.json"
        with open(individual_file, "w") as f:
            json.dump(model_data, f, indent=2)


def run_process_all(run_id: str) -> bool:
    """Run process_all.py for the given run_id."""
    cmd = [sys.executable, "-m", "src.leaderboard.process_all", run_id]
    print(f"\n▶ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_src_dir.parent))
    return result.returncode == 0


def _try_generate_reference_files(reference_dir: str, publish_dir: Path) -> None:
    """Attempt to regenerate reference files; log a warning if unavailable."""
    try:
        from src.leaderboard.functions.generate_reference_files import generate_reference_files
        generate_reference_files(reference_dir)
    except (ImportError, Exception) as e:
        print(f"  ⚠  Could not regenerate reference files: {e}")
        return

    try:
        from src.leaderboard.functions.copy_reference_files import copy_reference_files
        copy_reference_files(reference_dir, str(publish_dir))
    except (ImportError, Exception) as e:
        print(f"  ⚠  Could not copy reference files: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge new benchmark results with HF dataset and publish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.leaderboard.merge_and_publish --run-id my_run_20260521
  python -m src.leaderboard.merge_and_publish --skip-process
        """,
    )
    parser.add_argument("--run-id", help="Run ID to process first")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip process_all.py; use existing outputs/publish/")
    return parser.parse_args()


def main() -> bool:
    args = parse_args()

    project_root = _src_dir.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not set. Add it to .env or export it in your shell.")
        return False

    config = load_config()
    publish_dir = Path(config["paths"]["publish_dir"])
    dataset_name = config["datasets"]["results"]
    reference_dir = config["paths"].get("reference_dir", "reference")

    print("=" * 70)
    print("\U0001f680 Merge-and-Publish Pipeline")
    print(f"   Dataset: {dataset_name}")
    print("=" * 70)

    if not args.skip_process:
        if not args.run_id:
            print("❌ Provide --run-id <id> or --skip-process")
            return False
        if not run_process_all(args.run_id):
            print("❌ process_all.py failed — aborting")
            return False

    print("\n\U0001f4e5 Step 1: Downloading existing summaries from HF...")
    hf_summaries = download_hf_summaries(dataset_name, publish_dir)

    print("\n\U0001f4c2 Step 2: Loading locally-processed summaries...")
    local_summaries = load_local_summaries(publish_dir)

    if not hf_summaries and not local_summaries:
        print("❌ No summaries found locally or on HF.")
        return False

    print("\n\U0001f500 Step 3: Merging summaries...")
    merged = merge_summaries(hf_summaries, local_summaries)
    print(f"  HF models:    {len(hf_summaries)}")
    print(f"  Local models: {len(local_summaries)}")
    print(f"  Merged total: {len(merged)}")
    save_summaries(merged, publish_dir)

    print("\n\U0001f4ca Step 4: Regenerating leaderboard table...")
    if not generate_leaderboard_table(str(publish_dir)):
        print("❌ Failed to generate leaderboard table")
        return False

    print("\n\U0001f4cb Step 5: Regenerating reference files...")
    _try_generate_reference_files(reference_dir, publish_dir)

    print("\n☁️  Step 6: Uploading to HuggingFace...")
    if not upload_to_hf(str(publish_dir), dataset_name):
        print("❌ Upload failed")
        return False

    print("\n" + "=" * 70)
    print("\U0001f389 Done! View the dataset at:")
    print(f"   https://huggingface.co/datasets/{dataset_name}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
