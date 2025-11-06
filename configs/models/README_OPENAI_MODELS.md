# OpenAI Cloud Models - Quick Reference

## Available Configs

### GPT-4o Mini (Recommended for Testing)
**File**: `openai_gpt-4o-mini.yaml`
- **Model**: `gpt-4o-mini`
- **Best for**: Fast, cheap testing and evaluation
- **Context**: 128K tokens
- **Cost**: ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens

### GPT-5 Mini (Latest!)
**File**: `openai_gpt-5-mini.yaml`
- **Model**: `gpt-5-mini`
- **Best for**: Latest small model performance
- **Note**: Check OpenAI pricing for current rates

## Usage

```bash
# Test with GPT-4o Mini (cheap!)
python eval.py --config configs/models/openai_gpt-4o-mini.yaml --limit 5

# Test with GPT-5 Mini (latest!)
python eval.py --config configs/models/openai_gpt-5-mini.yaml --limit 5

# Full evaluation
python eval.py --config configs/models/openai_gpt-4o-mini.yaml
```

## Creating New Model Configs

Copy and modify any existing config:

```yaml
model:
  name: gpt-4o  # Or gpt-4, gpt-5, etc.
openai:
  provider_config: openai  # Uses configs/providers/openai.yaml
  overrides:
    temperature: 0.0
    max_tokens: 4096
task_defaults:
  log_samples: true
tasks:
- structured_extraction
metadata:
  provider: openai
  model_type: gpt4o
  is_cloud: true
```

## Available OpenAI Models

From your API (as of test):
- `gpt-5-mini` ⭐ Latest small model
- `gpt-5` - Latest flagship
- `gpt-4o` - Optimized GPT-4
- `gpt-4o-mini` ⭐ Cheap and fast
- `gpt-4-turbo` - Previous gen flagship
- `gpt-3.5-turbo` - Budget option

Full list: https://platform.openai.com/docs/models

## Cost Tips

1. **Always test with `--limit 5` first!**
   ```bash
   python eval.py --config configs/models/openai_gpt-5-mini.yaml --limit 5
   ```

2. **Use mini models for testing**:
   - `gpt-4o-mini` - Very cheap (~$0.15 / 1M tokens)
   - `gpt-5-mini` - Latest but still economical

3. **Monitor costs**: Check https://platform.openai.com/usage

## Provider Configs

Both work, use whichever you prefer:
- `openai.yaml` - Simple name
- `openai_cloud.yaml` - Explicit about being cloud

They're identical in functionality.

