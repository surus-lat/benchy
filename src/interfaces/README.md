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

## HTTPInterface

Base class for HTTP-based AI systems with custom endpoints.

**Supported Systems:**
- `surus`: SURUS AI /extract endpoint
- Custom HTTP endpoints (extend HTTPInterface)

**Features:**
- Async HTTP requests using httpx
- Configurable endpoints
- Custom request/response mapping
- Retry logic
- Connection testing

**Usage Example:**

```python
from src.interfaces.surus_interface import SurusInterface

# Configuration
config = {
    "surus": {
        "endpoint": "https://api.surus.dev/functions/v1/extract",
        "api_key_env": "SURUS_API_KEY",
        "timeout": 30,
        "max_retries": 3,
    }
}

# Initialize
interface = SurusInterface(config, "surus-extract", provider_type="surus")

# Test connection
connected = await interface.test_connection()

# Prepare requests (uses raw data, not prompts)
requests = [
    interface.prepare_request(sample, task)
    for sample in batch
]

# Generate
results = await interface.generate_batch(requests)
```

**Key Methods:**

```python
def prepare_request(self, sample: Dict, task) -> Dict:
    """Prepare request for HTTP endpoint."""
    return {
        "text": sample["text"],
        "schema": sample["schema"],
        "sample_id": sample["id"],
    }
```

**SurusInterface:**

Specialized interface for SURUS AI /extract endpoint.

- Uses raw text, not prompts
- Returns clean JSON (no markdown)
- OpenAI-compatible response format
- Optimized for structured extraction

## Request Preparation Pattern

All interfaces implement `prepare_request(sample, task)`:

```python
# LLMInterface - needs prompts
def prepare_request(self, sample, task):
    system_prompt, user_prompt = task.get_prompt(sample)
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "schema": sample["schema"],
        "sample_id": sample["id"],
    }

# HTTPInterface - uses raw data
def prepare_request(self, sample, task):
    return {
        "text": sample["text"],
        "schema": sample["schema"],
        "sample_id": sample["id"],
    }
```

This pattern ensures:
- Tasks stay provider-agnostic
- Interface-specific logic in interfaces
- No conditional code in tasks
- Easy to add new interfaces

## Future Interfaces

Additional interfaces can be added for:
- Custom HTTP APIs (extend HTTPInterface)
- Multimodal systems
- Agent-based systems
- Code execution environments

All interfaces should implement:
1. `def prepare_request(sample, task) -> Dict` - Format requests
2. `async def generate_batch(requests: List[Dict]) -> List[Dict]` - Generate responses
3. `async def test_connection() -> bool` - Test endpoint
4. Consistent error handling and logging

## Adding a New Interface

1. Create new file: `src/interfaces/my_interface.py`
2. Implement required methods: `generate_batch()` and `test_connection()`
3. Add to `__init__.py`: `from .my_interface import MyInterface`
4. Document configuration and usage patterns
5. Use in tasks via configuration: `provider_type: "my_provider"`

