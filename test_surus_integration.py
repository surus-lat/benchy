#!/usr/bin/env python3
"""Test SURUS interface integration."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

from src.interfaces.surus_interface import SurusInterface

async def test_surus_interface():
    """Test SURUS interface end-to-end."""
    print("=" * 60)
    print("Testing SURUS Interface Integration")
    print("=" * 60)
    
    # Configuration
    config = {
        "surus": {
            "endpoint": "https://api.surus.dev/functions/v1/extract",
            "api_key_env": "SURUS_API_KEY",
            "timeout": 30,
            "max_retries": 3,
        },
    }
    
    # Initialize interface
    print("\n1. Initializing SurusInterface...")
    interface = SurusInterface(config, "surus-extract", provider_type="surus")
    print("   ✓ Interface initialized")
    
    # Test connection
    print("\n2. Testing connection...")
    connected = await interface.test_connection(max_retries=2, timeout=30)
    if not connected:
        print("   ✗ Connection failed")
        return False
    print("   ✓ Connection successful")
    
    # Test prepare_request (without actual task, just check format)
    print("\n3. Testing prepare_request()...")
    sample = {
        "id": "test_sample",
        "text": "John Doe lives at 123 Main St in New York.",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "address": {"type": "string", "description": "Street address"},
                "city": {"type": "string", "description": "City name"},
            }
        },
        "expected": {"name": "John Doe", "address": "123 Main St", "city": "New York"}
    }
    
    # Mock task with get_prompt (not used by SURUS but needed for interface)
    class MockTask:
        def get_prompt(self, sample):
            return "System prompt", "User prompt"
    
    request = interface.prepare_request(sample, MockTask())
    print(f"   Request format: {list(request.keys())}")
    assert "text" in request
    assert "schema" in request
    assert "sample_id" in request
    assert "system_prompt" not in request  # HTTP interface doesn't use prompts
    print("   ✓ Request format correct")
    
    # Test single generation
    print("\n4. Testing single generation...")
    requests = [request]
    results = await interface.generate_batch(requests)
    
    if results and results[0]:
        result = results[0]
        print(f"   Output: {result.get('output')}")
        print(f"   Raw: {result.get('raw')}")
        print(f"   Error: {result.get('error')}")
        
        if result.get('output'):
            print("   ✓ Generation successful")
            print(f"   ✓ Extracted: {result['output']}")
        else:
            print(f"   ✗ Generation failed: {result.get('error')}")
            return False
    else:
        print("   ✗ No results returned")
        return False
    
    print("\n" + "=" * 60)
    print("✓ SURUS Interface Integration Test PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_surus_interface())
    exit(0 if success else 1)

