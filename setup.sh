#!/bin/bash

# Benchy setup script.
# Creates a local venv and installs dependencies.
# Supports both uv (recommended) and traditional venv + pip.

set -e

EXTRAS="${BENCHY_EXTRAS:-}"
UV_EXTRAS_ARGS=()
PIP_EXTRAS=""

if [ -n "$EXTRAS" ]; then
    IFS=',' read -ra EXTRA_LIST <<< "$EXTRAS"
    for extra in "${EXTRA_LIST[@]}"; do
        extra_trimmed=$(echo "$extra" | xargs)
        if [ -n "$extra_trimmed" ]; then
            UV_EXTRAS_ARGS+=(--extra "$extra_trimmed")
            if [ -z "$PIP_EXTRAS" ]; then
                PIP_EXTRAS="$extra_trimmed"
            else
                PIP_EXTRAS="${PIP_EXTRAS},${extra_trimmed}"
            fi
        fi
    done
fi

echo "Setting up Benchy environment..."

# Check if uv is available
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for dependency management..."
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync "${UV_EXTRAS_ARGS[@]}"
    echo "Environment ready with uv."
else
    echo "uv not found. Using traditional venv + pip..."
    echo "Note: For faster setup, consider installing uv from https://github.com/astral-sh/uv"
    
    # Check Python version
    if command -v python3.12 >/dev/null 2>&1; then
        python_bin="python3.12"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    else
        echo "Error: Python 3.12+ is required, but no python3 interpreter was found."
        exit 1
    fi

    python_version=$($python_bin --version 2>&1 | awk '{print $2}')
    python_major=$(echo "$python_version" | cut -d. -f1)
    python_minor=$(echo "$python_version" | cut -d. -f2)
    
    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 12 ]); then
        echo "Error: Python 3.12+ is required. Found Python $python_version"
        exit 1
    fi
    
    # Create venv
    $python_bin -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install project dependencies
    # Using pip install -e . to install dependencies from pyproject.toml
    # This makes the project importable and installs all dependencies
    if [ -n "$PIP_EXTRAS" ]; then
        pip install -e ".[${PIP_EXTRAS}]"
    else
        pip install -e .
    fi
    
    echo "Environment ready with pip."
fi

# Prefetch structured extraction dataset (optional)
DATASET_FILE="src/tasks/structured/.data/paraloq_data.jsonl"
case "${BENCHY_SKIP_DATASET:-}" in
    1|true|TRUE|yes|YES)
        echo "Skipping structured extraction dataset download (BENCHY_SKIP_DATASET set)."
        ;;
    *)
        if [ -f "$DATASET_FILE" ]; then
            echo "Structured dataset already exists at $DATASET_FILE"
        else
            echo "Downloading structured extraction dataset..."
            python -c "from src.tasks.structured.download_dataset import download_and_preprocess; download_and_preprocess('src/tasks/structured/.data', 'src/tasks/structured/cache')"
            echo "Dataset downloaded."
        fi
        ;;
esac

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Optional: Install extras by re-running with BENCHY_EXTRAS (comma-separated), e.g.:"
echo "  BENCHY_EXTRAS=translation,prefect bash setup.sh"
