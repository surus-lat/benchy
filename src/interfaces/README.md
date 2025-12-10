# Interfaces

Interfaces handle communication with AI systems during evaluation. They are provider-agnostic and work with standardized `connection_info` dicts.

## Architecture

```
Pipeline (resolves provider) → connection_info → Interface → AI System
                                    ↓
                              BenchmarkRunner
                                    ↓
                                  Task
```

Key principle: **Interfaces adapt task data to their API format**.
- LLM interfaces call `task.get_prompt()` to build chat messages
- HTTP interfaces use raw `sample["text"]` directly
- Tasks don't know which interface is being used

## Available Interfaces

### ChatCompletionsInterface

For OpenAI-compatible chat APIs (vLLM, OpenAI, Anthropic, etc.)

```python
from src.interfaces import ChatCompletionsInterface

# Connection info from engine.build_connection_info()
connection_info = {
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "timeout": 120,
    "max_retries": 3,
    "temperature": 0.0,
    "max_tokens": 2048,
    "use_guided_json": False,  # True for vLLM
}

interface = ChatCompletionsInterface(connection_info, model_name="gpt-4o-mini")

# Test connection
connected = await interface.test_connection()

# Prepare requests (interface calls task.get_prompt())
requests = [interface.prepare_request(sample, task) for sample in batch]

# Generate
results = await interface.generate_batch(requests)
```

### HTTPInterface / SurusInterface

For custom HTTP endpoints that don't use chat completions.

```python
from src.interfaces import SurusInterface

# SURUS-specific config
config = {
    "surus": {
        "endpoint": "https://api.surus.dev/functions/v1/extract",
        "api_key_env": "SURUS_API_KEY",
        "timeout": 30,
        "max_retries": 3,
    }
}

interface = SurusInterface(config, "surus-extract", provider_type="surus")

# Prepare requests (uses raw sample data, not prompts)
requests = [interface.prepare_request(sample, task) for sample in batch]

# Generate
results = await interface.generate_batch(requests)
```

## Integration with Benchmark Engine

The recommended way to get an interface is via the engine:

```python
from src.engine import build_connection_info, get_interface_for_provider

# Build standardized connection info from provider config
connection_info = build_connection_info(
    provider_type="openai",  # or "vllm", "anthropic", "surus"
    provider_config={"base_url": "https://api.openai.com/v1"},
    server_info=None,  # For vLLM, pass server_info here
)

# Get appropriate interface
interface = get_interface_for_provider(
    provider_type="openai",
    connection_info=connection_info,
    model_name="gpt-4o-mini",
)
```

## Creating a New Interface

### 1. Implement the BaseInterface Protocol

```python
# src/interfaces/my_interface.py
from typing import Dict, List, Any

class MyInterface:
    """Interface for MyService API."""
    
    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        """Initialize with connection info."""
        self.base_url = connection_info["base_url"]
        self.model_name = model_name
        # ... setup client
    
    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request from task sample.
        
        For LLM-like interfaces, call task.get_prompt():
            system, user = task.get_prompt(sample)
            return {"system": system, "user": user, ...}
        
        For HTTP interfaces, use raw sample data:
            return {"text": sample["text"], "schema": sample["schema"], ...}
        """
        # Implement based on your API's needs
        pass
    
    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate outputs for batch of requests.
        
        Returns list of dicts with:
        - output: Parsed output (or None on error)
        - raw: Raw response string
        - error: Error message (or None on success)
        """
        pass
    
    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to the API."""
        pass
```

### 2. Register in `__init__.py`

```python
# src/interfaces/__init__.py
from .my_interface import MyInterface

__all__ = [..., "MyInterface"]
```

### 3. Add to Connection Factory (Optional)

If your interface should be auto-selected for a provider type:

```python
# src/engine/connection.py
def get_interface_for_provider(provider_type, connection_info, model_name):
    if provider_type == "my_service":
        from ..interfaces.my_interface import MyInterface
        return MyInterface(connection_info, model_name)
    # ... existing providers
```

## Response Format

All interfaces must return results in this format:

```python
{
    "output": {...},  # Parsed output (dict, list, str, etc.) or None
    "raw": "...",     # Raw response text
    "error": None,    # Error message string or None
}
```

## Best Practices

1. **Keep interfaces stateless** - Don't store sample data between calls
2. **Use async for I/O** - All network calls should be async
3. **Handle errors gracefully** - Return error in result dict, don't raise
4. **Log appropriately** - Info for progress, debug for details, warning for retries
5. **Support batch processing** - Implement concurrent requests in `generate_batch()`

## Connection Info Format

Interfaces receive a standardized `connection_info` dict:

```python
{
    "base_url": "https://api.example.com/v1",
    "api_key": "...",        # Direct key, or None
    "api_key_env": "API_KEY", # Env var name for key
    "timeout": 120,
    "max_retries": 3,
    "temperature": 0.0,
    "max_tokens": 2048,
    "use_guided_json": False,  # vLLM-specific
}
```

This is built by `engine.build_connection_info()` from provider configs.
