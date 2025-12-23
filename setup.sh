#!/bin/bash

# Benchy Setup Script
# Creates virtual environments and installs dependencies using uv

set -e  # Exit on any error

echo "🚀 Setting up Benchy - LATAM Leaderboard Benchmarking Suite"
echo "=========================================================="

# Function to create venv and install dependencies
setup_venv() {
    local dir=$1
    local install_cmd=$2
    local name=$3
    
    echo ""
    echo "📦 Setting up $name..."
    echo "   Directory: $dir"
    
    if [ ! -d "$dir" ]; then
        echo "❌ Directory $dir not found!"
        return 1
    fi
    
    cd "$dir"
    
    # Create virtual environment
    echo "   Creating virtual environment..."
    uv venv
    
    # Activate and install
    echo "   Installing dependencies..."
    source .venv/bin/activate
    eval "$install_cmd"
    
    echo "✅ $name setup complete!"
    cd - > /dev/null
}

# Setup root repository
echo ""
echo "🏠 Setting up root repository..."
uv venv
source .venv/bin/activate
uv pip install -e .
echo "✅ Root repository setup complete!"

# Download structured extraction dataset
echo ""
echo "📊 Setting up structured extraction dataset..."
DATASET_FILE="src/tasks/structured/.data/paraloq_data.jsonl"
if [ -f "$DATASET_FILE" ]; then
    echo "✅ Dataset already exists at $DATASET_FILE"
else
    echo "   Downloading paraloq dataset..."
    source .venv/bin/activate
    python -c "from src.tasks.structured.download_dataset import download_and_preprocess; download_and_preprocess('src/tasks/structured/.data', 'src/tasks/structured/cache')"
    echo "✅ Dataset downloaded successfully!"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the main environment:"
echo "  source .venv/bin/activate"
echo ""
echo "No additional external environments are required."
