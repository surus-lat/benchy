#!/bin/bash
models=(
    "tencent/Hunyuan-MT-7B" #works
    "ByteDance-Seed/Seed-X-PPO-7B"
    "ByteDance-Seed/Seed-X-Instruct-7B"
)

for model in "${models[@]}"; do
    echo "Downloading $model..."
    huggingface-cli download "$model"
done