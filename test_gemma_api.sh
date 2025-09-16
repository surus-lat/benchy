#!/bin/bash

# Test script for Gemma3n using vLLM API server approach
# This matches your working command structure

echo "=== Testing Gemma3n with vLLM API Server ==="

# Step 1: Start vLLM server (run this in another terminal)
echo "1. Start vLLM server with this command (in another terminal):"
echo "HF_CACHE=/home/mauro/.cache/huggingface HF_TOKEN=\$HF_TOKEN python -m vllm.entrypoints.openai.api_server \\"
echo "    --host 0.0.0.0 \\"
echo "    --model google/gemma-3n-E4B-it \\"
echo "    --enforce-eager -tp 1 \\"
echo "    --max-model-len 8192 \\"
echo "    --limit-mm-per-prompt '{\"images\": 0, \"audios\": 0}'"
echo ""

# Wait for user confirmation
echo "2. Press Enter when the server is running and shows 'Uvicorn running on http://0.0.0.0:8000'..."
read -p "Press Enter to continue..."

# Step 2: Test the API connection
echo "3. Testing API connection..."
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3n-E4B-it",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'

echo -e "\n\n4. If the API test worked, you can now run the evaluation with:"
echo "cd /home/mauro/dev/benchy && python main.py configs/gemma-e4b.yaml"

echo -e "\n5. Or test just the lm_eval command directly:"
echo "cd /home/mauro/dev/lm-evaluation-harness && source .venv/bin/activate"
echo "lm_eval --model local-completions \\"
echo "  --model_args model=google/gemma-3n-E4B-it,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=3 \\"
echo "  --tasks hellaswag \\"
echo "  --batch_size auto \\"
echo "  --limit 1"
