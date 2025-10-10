"""Download and preprocess the paraloq/json_data_extraction dataset."""

import argparse
import logging
from pathlib import Path

from .utils.dataset_download import download_and_preprocess_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def download_and_preprocess(output_dir: str = "./.data", cache_dir: str = "./cache"):
    """Download and preprocess the paraloq dataset.
    
    Args:
        output_dir: Directory to save processed data
        cache_dir: Directory for HuggingFace cache
    """
    output_file = Path(output_dir) / "paraloq_data.jsonl"
    
    stats = download_and_preprocess_dataset(
        dataset_name="paraloq/json_data_extraction",
        output_file=output_file,
        cache_dir=cache_dir,
        split="train",
        max_input_chars=20000,
    )
    
    logging.info("Dataset ready for use!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess paraloq dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for processed data (default: ./data)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for HuggingFace (default: ./cache)",
    )
    
    args = parser.parse_args()
    download_and_preprocess(args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()

