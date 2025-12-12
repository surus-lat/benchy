#!/usr/bin/env python3
"""Test script for image-based invoice extraction."""

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from root .env
load_dotenv(Path(__file__).parent.parent / ".env")

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    # Paths
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompt.txt"
    schema_path = script_dir / "schema.json"
    expected_path = script_dir / "datos.json"
    
    # Load expected data first to get filename
    with open(expected_path, "r") as f:
        expected_data = json.load(f)[0]  # Get first item from array
    
    # Get image path from filename in datos.json
    filename = expected_data.get("filename", "test_image")
    # Try jpgs directory in test_image_request first, then facturas/jpgs
    jpgs_dir = script_dir / "jpgs"
    image_path = jpgs_dir / f"{filename}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load prompt
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()
    
    # Load schema
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    # Encode image
    print("Encoding image...")
    base64_image = encode_image(image_path)
    
    # Initialize client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    client = OpenAI(api_key=api_key)
    
    # Make request
    print("Making request to OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4o",  # Using gpt-4o for vision capabilities
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "factura_extraction",
                "strict": True,
                "schema": schema
            }
        }
    )
    
    # Extract response
    result_text = response.choices[0].message.content
    result_data = json.loads(result_text)
    
    print("\nExtracted data:")
    print(json.dumps(result_data, indent=2, ensure_ascii=False))
    
    # Compare with expected data (skip filename as it's not in schema)
    print("\nComparing with expected data...")
    matching_fields = 0
    # Count fields excluding filename
    fields_to_compare = {k: v for k, v in expected_data.items() if k != "filename"}
    total_fields = len(fields_to_compare)
    
    for key, expected_value in fields_to_compare.items():
        actual_value = result_data.get(key)
        
        # Special handling for fecha_factura: compare only date, ignore time
        if key == "fecha_factura":
            try:
                expected_date = datetime.fromisoformat(expected_value.replace("Z", "+00:00")).date()
                actual_date = datetime.fromisoformat(actual_value.replace("Z", "+00:00")).date()
                is_match = expected_date == actual_date
            except (ValueError, AttributeError):
                is_match = False
        # Special handling for punto_de_venta and numero_de_comprobante: cast to int
        elif key in ("punto_de_venta", "numero_de_comprobante"):
            try:
                expected_int = int(expected_value)
                actual_int = int(actual_value)
                is_match = expected_int == actual_int
            except (ValueError, TypeError):
                is_match = False
        else:
            is_match = actual_value == expected_value
        
        if is_match:
            matching_fields += 1
            print(f"✓ {key}: {actual_value}")
        else:
            print(f"✗ {key}: expected {expected_value}, got {actual_value}")
    
    match_percentage = (matching_fields / total_fields) * 100
    print(f"\nMatch: {matching_fields}/{total_fields} fields ({match_percentage:.1f}%)")

if __name__ == "__main__":
    main()

