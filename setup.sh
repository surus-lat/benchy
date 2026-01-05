#!/bin/bash

# Benchy setup script.
# Creates a local venv and installs dependencies.
# Supports both uv (recommended) and traditional venv + pip.

set -e

echo "Setting up Benchy environment..."

# Check if uv is available
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for dependency management..."
    uv venv
    source .venv/bin/activate
    uv sync
    echo "Environment ready with uv."
else
    echo "uv not found. Using traditional venv + pip..."
    echo "Note: For faster setup, consider installing uv from https://github.com/astral-sh/uv"
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 12 ]); then
        echo "Error: Python 3.12+ is required. Found Python $python_version"
        exit 1
    fi
    
    # Create venv
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install project dependencies
    # Using pip install -e . to install dependencies from pyproject.toml
    # This makes the project importable and installs all dependencies
    pip install -e .
    
    echo "Environment ready with pip."
fi

# Prefetch structured extraction dataset (optional)
DATASET_FILE="src/tasks/structured/.data/paraloq_data.jsonl"
if [ -f "$DATASET_FILE" ]; then
    echo "Structured dataset already exists at $DATASET_FILE"
else
    echo "Downloading structured extraction dataset..."
    python -c "from src.tasks.structured.download_dataset import download_and_preprocess; download_and_preprocess('src/tasks/structured/.data', 'src/tasks/structured/cache')"
    echo "Dataset downloaded."
fi

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Optional: Install translation metrics dependencies:"
echo "  pip install '.[translation]'"
