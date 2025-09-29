#!/bin/bash
models=(
    "swiss-ai/Apertus-8B-Instruct-2509"
    "arcee-ai/AFM-4.5B"
)

for model in "${models[@]}"; do
    echo "Downloading $model..."
    huggingface-cli download "$model"
done