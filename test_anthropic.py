#!/usr/bin/env python3
"""Simple test script for Anthropic Claude Haiku API."""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ ERROR: ANTHROPIC_API_KEY not found in .env file")
    print("Please add your key to .env file:")
    print("  ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
    exit(1)

print(f"✅ API key found: {api_key[:20]}... (length: {len(api_key)})")

# Test the API
try:
    from anthropic import Anthropic
    
    print("\n🚀 Testing Anthropic API with Claude 3.5 Haiku...")
    
    client = Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say 'Hello World!' and tell me one interesting fact about AI in exactly one sentence."}
        ]
    )
    
    response_text = message.content[0].text
    
    print("\n✅ SUCCESS! Claude 3.5 Haiku responded:")
    print("=" * 60)
    print(response_text)
    print("=" * 60)
    print(f"\nℹ️  Model: {message.model}")
    print(f"ℹ️  Tokens used: {message.usage.input_tokens} input, {message.usage.output_tokens} output")
    print("\n🎉 Anthropic API is working correctly!")
    
except ImportError:
    print("❌ ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    exit(1)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPossible issues:")
    print("  1. Invalid API key - check your .env file")
    print("  2. Network connection issue")
    print("  3. API quota exceeded")
    print("\nGet your API key at: https://console.anthropic.com/settings/keys")
    exit(1)

