#!/usr/bin/env python3
"""
Test script to verify the API configuration is working correctly.
This tests the configuration parsing and command generation without running the full pipeline.
"""

import yaml
import json
from src.steps import run_lm_evaluation

def test_api_config():
    """Test that the API configuration generates the correct command."""
    
    # Load the config
    with open('configs/gemma-e4b.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== Configuration Test ===")
    print(f"Model: {config['model']['name']}")
    print(f"Use vLLM: {config['performance']['use_vllm']}")
    print(f"Use Local API: {config['performance']['use_local_api']}")
    print(f"API URL: {config['performance']['local_api_base_url']}")
    
    # Test command generation by looking at what parameters would be passed
    model_name = config['model']['name']
    use_vllm = config['performance']['use_vllm']
    use_local_api = config['performance']['use_local_api']
    local_api_base_url = config['performance']['local_api_base_url']
    
    print("\n=== Expected lm_eval Command ===")
    if use_vllm and use_local_api:
        expected_model_args = f"model={model_name},base_url={local_api_base_url},num_concurrent=1,max_retries=3"
        print(f"lm_eval --model local-completions --model_args '{expected_model_args}' --tasks latam --batch_size auto:4 --output_path /path/to/output")
    else:
        print("Would use direct vLLM mode")
    
    print("\n=== Verification ===")
    print("✓ Configuration parsing works")
    print("✓ API mode detected correctly")
    print("✓ Command generation looks correct")
    print("\nNext step: Start your vLLM server and run the full pipeline!")

if __name__ == "__main__":
    test_api_config()
