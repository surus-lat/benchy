# AI System Configurations

This directory contains configurations for task-optimized AI systems and specialized endpoints.

## Purpose

System configs are for evaluating **AI systems** that are optimized for specific tasks, as opposed to general-purpose language models.

## Directory Organization

```
configs/
├── models/          # General-purpose LLM configurations
│   ├── gpt-4.yaml
│   ├── claude.yaml
│   └── local-model.yaml
│
└── systems/         # Task-optimized AI system configurations
    ├── surus-extract.yaml
    └── custom-api.yaml
```

## Key Differences

### Model Configs (`configs/models/`)
- **Purpose**: Evaluate general-purpose language models
- **Examples**: OpenAI, Anthropic, Llama, Gemma
- **Provider types**: `vllm`, `openai`, `anthropic`, `together`
- **Characteristics**:
  - Support multiple tasks
  - Use standardized prompts
  - Text generation focus
  - Chat completions API

### System Configs (`configs/systems/`)
- **Purpose**: Evaluate task-specialized AI systems
- **Examples**: SURUS /extract, custom extraction APIs
- **Provider types**: `surus`, `surus_ocr`, `surus_factura`, `http`
- **Characteristics**:
  - Optimized for specific tasks
  - Custom API formats
  - Task-specific endpoints
  - Specialized for performance

## System Configuration Format

```yaml
# System identification
system_name: "my-system"
provider_type: "http"  # or "surus", etc.

# Provider-specific configuration
http:
  endpoint: "https://api.example.com/v1/extract"
  api_key_env: "MY_API_KEY"
  timeout: 30
  max_retries: 3
  capabilities:
    supports_schema: true
    supports_files: false

# Model info (for tracking only, not controlling the system)
model:
  name: "my-system"  # System identifier

# Compatible tasks
tasks:
  - "structured_extraction"

# Task configuration
structured_extraction:
  tasks:
    - "paraloq"
  defaults:
    batch_size: 10
```

## Available Systems

### SURUS /extract
**File**: `surus-extract.yaml`
**Purpose**: Optimized structured data extraction
**Endpoint**: https://api.surus.dev/functions/v1/extract
**Compatible tasks**: `structured_extraction`

Task-optimized model for JSON extraction with guided schemas. Significantly faster than general-purpose LLMs for extraction tasks.

**Setup**:
```bash
# Add to .env
SURUS_API_KEY=your-key-here

# Run benchmark
python eval.py --config configs/systems/surus-extract.yaml --limit 10
```

## Creating New System Configs

1. **Create config file**: `configs/systems/my-system.yaml`
2. **Set provider_type**: Determines which interface to use
3. **Configure endpoint**: API URL and authentication
4. **Specify compatible tasks**: Not all tasks work with all systems
5. **Adjust batch size**: Conservative values for external APIs

## Best Practices

1. **Use conservative batch sizes** for external APIs (5-10)
2. **Set appropriate timeouts** based on endpoint speed
3. **Configure retries** for reliability
4. **Test with --limit** before full runs
5. **Declare system capabilities** under the provider block
6. **Track system identifiers** not backend models (we don't control those)

## Testing

```bash
# Test with small limit
python eval.py --config configs/systems/my-system.yaml --limit 5

# Full evaluation
python eval.py --config configs/systems/my-system.yaml
```

## Comparison with Models

| Aspect | Models | Systems |
|--------|--------|---------|
| Flexibility | High - any task | Low - specific tasks |
| API Format | Standardized (OpenAI) | Custom per system |
| Performance | General-purpose | Task-optimized |
| Cost | Per-token | Per-request (often) |
| Setup | vLLM server or API key | API key only |

## Adding New Provider Types

To add a new provider type:

1. Create interface in `src/interfaces/`
2. Implement `prepare_request()` and `generate_batch()`
3. Register it in `src/engine/connection.py` via `get_interface_for_provider`
4. Add provider handling in `eval.py` and `src/config_manager.py` if models should use it
5. Create or update a system config template
6. Document in this README
