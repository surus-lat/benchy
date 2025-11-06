# Quick Start: Cloud Providers

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# For OpenAI
pip install openai

# For Anthropic (Claude)
pip install anthropic

# Or both
pip install openai anthropic
```

### 2. Set API Keys

**Option A: Using .env file (Recommended)**

```bash
# Copy the example file
cp env.example .env

# Edit .env and add your keys
# The file should look like:
OPENAI_API_KEY=sk-your-actual-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

**Option B: Using environment variables**

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run Evaluation

```bash
# Test with OpenAI (5 samples)
python eval.py --config configs/models/openai_gpt4o_mini.yaml --limit 5

# Test with Anthropic (5 samples)
python eval.py --config configs/models/anthropic_claude_haiku.yaml --limit 5

# Full evaluation
python eval.py --config configs/models/openai_gpt4o_mini.yaml
```

## 📝 Creating Your Own Model Config

### OpenAI Model

Create `configs/models/your_openai_model.yaml`:

```yaml
model:
  name: gpt-4o-mini  # or gpt-4, gpt-3.5-turbo, etc.
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
  is_cloud: true
```

### Anthropic Model

Create `configs/models/your_claude_model.yaml`:

```yaml
model:
  name: claude-3-5-haiku-20241022  # or claude-3-5-sonnet, etc.
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
  is_cloud: true
```

## 💰 Cost Management

**Important**: Cloud providers charge per token!

```bash
# Use --limit to test with few samples
python eval.py --config configs/models/openai_gpt4o_mini.yaml --limit 10

# Reduce batch size in config
task_defaults:
  batch_size: 5  # Lower = slower but easier on rate limits
```

## ✅ What Works

- ✅ OpenAI models (GPT-3.5, GPT-4, GPT-4o, etc.)
- ✅ Anthropic models (Claude 3.5 Haiku, Claude 3.5 Sonnet, etc.)
- ✅ Structured extraction tasks (paraloq, chat_extract)
- ✅ All evaluation metrics
- ✅ Checkpointing and resume
- ✅ Batch processing

## ⚠️ Current Limitations

- ⚠️ Only `structured_extraction` task supported (not spanish, portuguese, translation yet)
- ⚠️ `--test` flag only works with vLLM
- ⚠️ Anthropic uses prompt-based JSON guidance (no native guided output)

## 🔍 Troubleshooting

### "API key not found"
```bash
# Make sure you've exported the key
export OPENAI_API_KEY="your-key-here"
# Or add to .env file
```

### "Module not found"
```bash
# Install the required package
pip install openai anthropic
```

### Rate Limiting
Reduce batch size in your model config:
```yaml
task_defaults:
  batch_size: 5
```

## 📚 More Info

- Full documentation: `docs/CLOUD_PROVIDERS.md`
- Implementation details: `CHANGELOG_CLOUD_PROVIDERS.md`
- Existing vLLM configs work unchanged!

## 🎯 Model Name Reference

### OpenAI Models
- `gpt-4o` - Most capable, latest GPT-4
- `gpt-4o-mini` - Faster, cheaper GPT-4
- `gpt-4-turbo` - Previous generation GPT-4
- `gpt-3.5-turbo` - Fast and cheap

### Anthropic Models
- `claude-3-5-sonnet-20241022` - Most capable Claude
- `claude-3-5-haiku-20241022` - Fast and cheap Claude
- `claude-3-opus-20240229` - Previous generation, very capable
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fast and economical

See provider docs for latest model names:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://docs.anthropic.com/en/docs/models-overview

