#!/bin/bash
models=(
    "01-ai/Yi-1.5-9B-Chat" # 4096
    "bigscience/bloomz-7b1" # 2048
    "01-ai/Yi-1.5-6B-Chat" # 4096
    "HuggingFaceH4/zephyr-7b-beta" # 8192
    "NousResearch/Hermes-3-Llama-3.1-8B" # 8192
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "CohereLabs/c4ai-command-r7b-12-2024"
)

for model in "${models[@]}"; do
    echo "Downloading $model..."
    huggingface-cli download "$model"
done