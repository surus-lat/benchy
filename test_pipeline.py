#!/usr/bin/env python3
"""Simple test to check if ZenML pipeline steps are actually executing."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
if os.path.exists('.env'):
    load_dotenv('.env')

# Add src to path
sys.path.insert(0, 'src')

from steps import start_vllm_server, test_vllm_api

def test_individual_steps():
    """Test individual steps to see if they work."""
    print("Testing individual steps...")
    
    try:
        print("1. Testing start_vllm_server step...")
        server_info = start_vllm_server(
            model_name="google/gemma-3n-E4B-it",
            host="0.0.0.0",
            port=8001,  # Use different port to avoid conflicts
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            limit_mm_per_prompt='{"images": 0, "audios": 0}',
            hf_cache="/home/mauro/.cache/huggingface",
            hf_token="",
            lm_eval_path="/home/mauro/dev/lm-evaluation-harness"
        )
        print(f"Server started: {server_info}")
        
        print("2. Testing API...")
        api_result = test_vllm_api(server_info, "google/gemma-3n-E4B-it")
        print(f"API test: {api_result}")
        
        print("3. Cleaning up...")
        import requests
        import psutil
        
        # Kill the server
        try:
            pid = server_info["pid"]
            proc = psutil.Process(pid)
            proc.kill()
            print(f"Killed server process {pid}")
        except:
            print("Failed to kill server process")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_individual_steps()
