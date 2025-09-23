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

# Setup external modules
setup_venv "external/lm-evaluation-harness" "uv pip install -e .[api]" "lm-evaluation-harness"
setup_venv "external/portuguese-bench" "uv pip install -e \".[anthropic,openai,sentencepiece]\"" "portuguese-bench"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the main environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To activate external environments:"
echo "  source external/lm-evaluation-harness/.venv/bin/activate"
echo "  source external/portuguese-bench/.venv/bin/activate"
