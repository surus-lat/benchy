# Cloud Provider Support

This document explains how to use cloud-based LLM providers (OpenAI and Anthropic) instead of locally hosted vLLM models.

## Overview

The benchmarking system now supports three types of providers:
- **vLLM**: Self-hosted models using vLLM server (original behavior)
- **OpenAI**: OpenAI's cloud API (GPT models)
- **Anthropic**: Anthropic's cloud API (Claude models)

## Configuration

### Provider Configuration Files

Provider configurations are stored in `configs/providers/`:

#### OpenAI Cloud Provider (`openai_cloud.yaml`)
```yaml
api_key_env: OPENAI_API_KEY
base_url: https://api.openai.com/v1
timeout: 120
max_retries: 3
temperature: 0.0
max_tokens: 4096
```

#### Anthropic Cloud Provider (`anthropic_cloud.yaml`)
```yaml
api_key_env: ANTHROPIC_API_KEY
base_url: https://api.anthropic.com
timeout: 120
max_retries: 3
temperature: 0.0
max_tokens: 4096
api_version: "2023-06-01"
```

### Model Configuration Files

Model configs reference provider configs using the `provider_config` pattern:

#### OpenAI Model Example (`configs/models/openai_gpt4o_mini.yaml`)
```yaml
model:
  name: gpt-4o-mini
openai:
  provider_config: openai_cloud
  overrides:
    temperature: 0.0
    max_tokens: 4096
task_defaults:
  log_samples: true
tasks:
- structured_extraction
metadata:
  provider: openai
  model_type: gpt4o_mini
  is_cloud: true
```

#### Anthropic Model Example (`configs/models/anthropic_claude_haiku.yaml`)
```yaml
model:
  name: claude-3-5-haiku-20241022
anthropic:
  provider_config: anthropic_cloud
  overrides:
    temperature: 0.0
    max_tokens: 4096
task_defaults:
  log_samples: true
tasks:
- structured_extraction
metadata:
  provider: anthropic
  model_type: claude_haiku
  is_cloud: true
```

## Setup

### 1. Install Required Packages

For OpenAI:
```bash
pip install openai>=1.0.0
```

For Anthropic:
```bash
pip install anthropic>=0.18.0
```

### 2. Set API Keys

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root:

```bash
# Copy the example file
cp env.example .env
```

Then edit `.env` with your actual keys:

```bash
# .env file
OPENAI_API_KEY=sk-your-actual-openai-key
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
```

The `.env` file is automatically loaded when you run `eval.py`.

**Option B: Using environment variables**

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Run Evaluation

Run evaluations the same way as with vLLM models:

```bash
# OpenAI
python eval.py --config configs/models/openai_gpt4o_mini.yaml

# Anthropic
python eval.py --config configs/models/anthropic_claude_haiku.yaml
```

## Supported Tasks

Currently, cloud providers are supported for:
- **structured_extraction**: JSON extraction benchmarks (paraloq, chat_extract)

Support for other tasks (spanish, portuguese, translation) will be added in future updates.

## How It Works

### Architecture Changes

1. **Config Manager** (`src/config_manager.py`):
   - Detects provider type (`openai`, `anthropic`, or `vllm`)
   - Merges provider configs with model-specific overrides
   - Adds `provider_type` field to loaded config

2. **Pipeline** (`src/pipeline.py`):
   - Skips vLLM server start/stop for cloud providers
   - Passes provider config to tasks
   - Handles cloud and local providers transparently

3. **Task Runner** (`src/tasks/structured_extraction.py`):
   - Accepts optional `provider_config` parameter
   - Constructs base URL from provider config instead of server info
   - Passes provider type to benchmark runner

4. **Interface** (`src/tasks/structured/llm/interface.py`):
   - Supports OpenAI-compatible API (vLLM, OpenAI)
   - Supports Anthropic API with separate implementation
   - Handles API key injection from environment variables
   - Automatically falls back to appropriate API methods

### API Compatibility

**OpenAI**:
- Uses `AsyncOpenAI` client
- Supports guided JSON output via `extra_body`
- Compatible with vLLM's OpenAI-compatible endpoint

**Anthropic**:
- Uses `AsyncAnthropic` client
- JSON schema embedded in user prompt (no native guided output)
- Uses Messages API with system/user separation

## Limitations

1. **Guided JSON**: Anthropic doesn't support guided JSON natively, so the schema is included in the prompt for guidance
2. **Tasks**: Currently only `structured_extraction` task is supported for cloud providers
3. **Test Mode**: `--test` flag only works with vLLM provider

## Cost Considerations

**Important**: Cloud providers charge per token. Consider using:
- `--limit N` flag to limit evaluation samples
- Task defaults to control batch size
- Cheaper models for testing (e.g., `gpt-4o-mini`, `claude-3-5-haiku`)

Example with limit:
```bash
python eval.py --config configs/models/openai_gpt4o_mini.yaml --limit 10
```

## Troubleshooting

### API Key Not Found
**Error**: `ValueError: OpenAI API key not found`

**Solution**: Set the environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Module Not Found
**Error**: `ImportError: anthropic package not installed`

**Solution**: Install the package:
```bash
pip install anthropic
```

### Rate Limiting
Cloud providers may rate-limit your requests. The interface includes:
- Automatic retries with exponential backoff
- Configurable `max_retries` in provider config
- Batch size control via task defaults

To reduce rate limit issues:
```yaml
task_defaults:
  batch_size: 5  # Reduce from default 20
```

## Examples

### Quick Test with OpenAI
```bash
export OPENAI_API_KEY="sk-..."
python eval.py \
  --config configs/models/openai_gpt4o_mini.yaml \
  --limit 5 \
  --log-samples
```

### Production Run with Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python eval.py \
  --config configs/models/anthropic_claude_haiku.yaml
```

### Custom Provider Config
Create a custom provider config for different base URLs or settings:

```yaml
# configs/providers/openai_custom.yaml
api_key_env: OPENAI_API_KEY
base_url: https://your-custom-endpoint.com/v1
timeout: 300
max_retries: 5
temperature: 0.0
max_tokens: 8192
```

Then reference it in your model config:
```yaml
model:
  name: custom-model
openai:
  provider_config: openai_custom
```

## Future Enhancements

Planned improvements:
- [ ] Support for more tasks (spanish, portuguese, translation)
- [ ] Support for additional providers (Cohere, Together, etc.)
- [ ] Cost tracking and estimation
- [ ] Batch API support for cost optimization
- [ ] Streaming support for long-running evaluations

