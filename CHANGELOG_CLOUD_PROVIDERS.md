# Cloud Provider Support - Implementation Summary

## Changes Made

### New Files Created

1. **Provider Configurations**:
   - `configs/providers/openai_cloud.yaml` - OpenAI API configuration
   - `configs/providers/anthropic_cloud.yaml` - Anthropic API configuration

2. **Model Configurations**:
   - `configs/models/openai_gpt4o_mini.yaml` - Example OpenAI model
   - `configs/models/anthropic_claude_haiku.yaml` - Example Anthropic model

3. **Documentation**:
   - `docs/CLOUD_PROVIDERS.md` - Comprehensive usage guide

### Modified Files

1. **`src/config_manager.py`**:
   - Added `_merge_cloud_provider_config()` method to handle OpenAI/Anthropic configs
   - Added `provider_type` field to distinguish between vLLM, OpenAI, and Anthropic
   - Supports new provider_config pattern for cloud providers

2. **`src/pipeline.py`**:
   - Added `provider_type` and `provider_config` parameters to `benchmark_pipeline()`
   - Skip vLLM server start/stop for cloud providers
   - Pass provider config to structured_extraction task
   - Conditional cleanup based on provider type

3. **`eval.py`**:
   - Detect provider type from loaded config
   - Extract appropriate provider config (vllm/openai/anthropic)
   - Pass provider parameters to pipeline
   - Handle cloud provider logging

4. **`src/tasks/structured_extraction.py`**:
   - Added `provider_config` parameter
   - Determine base URL from provider config or server info
   - Pass provider_type to BenchmarkRunner
   - Support optional server_info for cloud providers

5. **`src/tasks/structured/benchmark_runner.py`**:
   - Added `provider_type` parameter to `__init__()`
   - Pass provider_type to VLLMInterface

6. **`src/tasks/structured/llm/interface.py`**:
   - Added support for multiple providers in `__init__()`
   - Implemented `_generate_single_anthropic()` for Anthropic API
   - Updated `test_connection()` to handle all provider types
   - API key handling from environment variables
   - Dynamic client initialization based on provider

## Architecture

### Provider Flow

```
eval.py
  ├── Load model config via ConfigManager
  ├── Detect provider_type (vllm/openai/anthropic)
  ├── Extract provider_config
  └── Call benchmark_pipeline()
      ├── [vLLM only] Start vLLM server
      ├── Run tasks (e.g., structured_extraction)
      │   └── BenchmarkRunner
      │       └── VLLMInterface (with provider_type)
      │           ├── OpenAI client (for vLLM & OpenAI)
      │           └── Anthropic client (for Anthropic)
      └── [vLLM only] Stop vLLM server
```

### Provider Detection

```python
# In config_manager.py
if 'openai' in model_config:
    model_config['provider_type'] = 'openai'
elif 'anthropic' in model_config:
    model_config['provider_type'] = 'anthropic'
elif 'vllm' in model_config:
    model_config['provider_type'] = 'vllm'
```

### API Key Injection

```python
# In interface.py
if provider_type == "openai":
    api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
elif provider_type == "anthropic":
    api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
```

## Backward Compatibility

All changes are backward compatible:
- Existing vLLM configs work unchanged
- `provider_type` defaults to `"vllm"`
- All vLLM-specific parameters still supported
- No breaking changes to existing workflows

## Testing

### Manual Testing Steps

1. **Test OpenAI**:
```bash
export OPENAI_API_KEY="your-key"
python eval.py --config configs/models/openai_gpt4o_mini.yaml --limit 5
```

2. **Test Anthropic**:
```bash
export ANTHROPIC_API_KEY="your-key"
python eval.py --config configs/models/anthropic_claude_haiku.yaml --limit 5
```

3. **Verify vLLM still works**:
```bash
python eval.py --config configs/models/openai_gpt-oss-20b_single.yaml --limit 5
```

### Expected Behavior

**For Cloud Providers**:
- ✅ No vLLM server startup logs
- ✅ Direct API connection messages
- ✅ API test connection succeeds
- ✅ Evaluation runs successfully
- ✅ Results saved to output directory

**For vLLM**:
- ✅ Server startup logs appear
- ✅ GPU allocation messages
- ✅ Server test succeeds
- ✅ Evaluation runs
- ✅ Server cleanup at end

## Dependencies

### Required Packages

**For OpenAI support**:
```bash
pip install openai>=1.0.0
```

**For Anthropic support**:
```bash
pip install anthropic>=0.18.0
```

These should be added to `pyproject.toml` or `requirements.txt` as optional dependencies.

## Known Limitations

1. **Task Support**: Currently only `structured_extraction` task supports cloud providers
2. **Guided JSON**: Anthropic doesn't support native guided JSON (schema in prompt instead)
3. **Test Mode**: `--test` flag only works with vLLM
4. **Cost**: Cloud providers charge per token - use `--limit` for testing

## Future Work

1. **Add cloud provider support to other tasks**:
   - spanish evaluation
   - portuguese evaluation  
   - translation evaluation

2. **Enhanced cost tracking**:
   - Token usage logging
   - Cost estimation
   - Budget limits

3. **Additional providers**:
   - Cohere
   - Together AI
   - Groq
   - Azure OpenAI

4. **Optimization**:
   - Batch API support
   - Caching strategies
   - Rate limit handling

## Migration Guide

### Converting a vLLM config to OpenAI

**Before (vLLM)**:
```yaml
model:
  name: local/my-model
vllm:
  host: 0.0.0.0
  port: 8000
  gpu_memory_utilization: 0.9
```

**After (OpenAI)**:
```yaml
model:
  name: gpt-4o-mini
openai:
  provider_config: openai_cloud
  overrides:
    temperature: 0.0
    max_tokens: 4096
```

### Converting to Anthropic

```yaml
model:
  name: claude-3-5-haiku-20241022
anthropic:
  provider_config: anthropic_cloud
  overrides:
    temperature: 0.0
    max_tokens: 4096
```

## Questions & Support

For questions or issues:
1. Check `docs/CLOUD_PROVIDERS.md` for detailed usage
2. Review error messages for API key issues
3. Verify required packages are installed
4. Check API rate limits and quotas

## Version Compatibility

- Python: 3.8+
- OpenAI SDK: 1.0.0+
- Anthropic SDK: 0.18.0+
- Existing vLLM setup: Unchanged

