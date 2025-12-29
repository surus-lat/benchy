#!/bin/bash

# Benchy setup script.
# Creates a local venv with uv and optionally prefetches structured extraction data.

set -e

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://github.com/astral-sh/uv"
    exit 1
fi

echo "Setting up Benchy environment..."

uv venv
source .venv/bin/activate
uv sync

echo "Environment ready."

# Prefetch structured extraction dataset (optional)
DATASET_FILE="src/tasks/structured/.data/paraloq_data.jsonl"
if [ -f "$DATASET_FILE" ]; then
    echo "Structured dataset already exists at $DATASET_FILE"
else
    echo "Downloading structured extraction dataset..."
    python -c "from src.tasks.structured.download_dataset import download_and_preprocess; download_and_preprocess('src/tasks/structured/.data', 'src/tasks/structured/cache')"
    echo "Dataset downloaded."
fi

echo "Done. Activate with: source .venv/bin/activate"
