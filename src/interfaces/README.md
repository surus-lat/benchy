# Interfaces

This directory contains interfaces for communicating with AI systems during evaluation.

## Purpose

The interfaces layer provides a clean separation between:
- **Task definition**: What to evaluate (prompts, metrics, datasets)
- **System interface**: How to communicate with the system being evaluated

This design enables benchy to evaluate various AI systems (LLMs, multimodal models, or custom HTTP APIs) using the same task definitions.

## Available Interfaces

### LLMInterface

Async client for LLM providers using OpenAI-compatible APIs.

**Supported Providers:**
- `vllm`: Local vLLM server with guided JSON schema support
- `openai`: OpenAI cloud API
- `anthropic`: Anthropic cloud API

**Features:**
- Async batch processing for high throughput
- Automatic retry with exponential backoff
- JSON extraction from markdown code blocks
- Connection testing before evaluation
- Multi-provider support with unified API

**Usage Example:**

```python
from src.interfaces.llm_interface import LLMInterface

# Configuration
config = {
    "model": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "temperature": 0.0,
        "max_tokens": 2048,
        "timeout": 120,
        "max_retries": 3,
    },
    "performance": {
        "batch_size": 20,
    }
}

# Initialize interface
llm = LLMInterface(config, model_name="my-model", provider_type="vllm")

# Test connection
connected = await llm.test_connection(max_retries=3, timeout=30)

# Generate batch
requests = [
    {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "Extract data from: ...",
        "schema": {"type": "object", "properties": {...}},
        "sample_id": "sample_0",
    }
]

results = await llm.generate_batch(requests)
```

**Configuration Options:**

- `base_url`: API endpoint (e.g., "http://localhost:8000/v1")
- `api_key`: API key ("EMPTY" for vLLM, required for cloud providers)
- `api_key_env`: Environment variable name for API key (default: provider-specific)
- `temperature`: Sampling temperature
- `max_tokens`: Maximum tokens to generate
- `timeout`: Request timeout in seconds
- `max_retries`: Number of retry attempts on failure
- `api_endpoint`: API preference - "auto" (default), "chat", or "completions"

**Provider-Specific Notes:**

**vLLM:**
- Supports guided JSON schema via `extra_body` parameter
- Set `api_key` to "EMPTY"
- Use `api_endpoint: "completions"` for models with chat template issues

**OpenAI:**
- Requires valid `OPENAI_API_KEY` environment variable
- Newer models (GPT-5, o-series) have parameter restrictions
- No guided JSON support (prompt-based only)

**Anthropic:**
- Requires valid `ANTHROPIC_API_KEY` environment variable
- No guided JSON support (prompt-based only)
- Uses different parameter names internally

## Future Interfaces

Additional interfaces can be added for:
- Custom HTTP APIs
- Multimodal systems
- Agent-based systems
- Code execution environments

All interfaces should implement:
1. `async def generate_batch(requests: List[Dict]) -> List[Dict]`
2. `async def test_connection() -> bool`
3. Consistent error handling and logging

## Adding a New Interface

1. Create new file: `src/interfaces/my_interface.py`
2. Implement required methods: `generate_batch()` and `test_connection()`
3. Add to `__init__.py`: `from .my_interface import MyInterface`
4. Document configuration and usage patterns
5. Use in tasks via configuration: `provider_type: "my_provider"`

