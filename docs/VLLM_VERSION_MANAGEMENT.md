# vLLM Version Management

This document describes the vLLM version management system that allows you to specify different vLLM versions for different models.

## Overview

The system supports:
- **Per-model vLLM version specification** in YAML configuration files
- **Automatic virtual environment management** - environments are created when needed
- **Default to latest vLLM version** - uses main project environment unless specified otherwise
- **Backward compatibility** with existing configurations
- **Non-Docker approach** using `uv` virtual environments

## How It Works

### Default Behavior (Recommended)
- **No `vllm_version` specified**: Uses the latest vLLM version from the main project environment (`/home/mauro/dev/benchy/.venv`)
- **`vllm_version` specified**: Automatically creates and uses a version-specific virtual environment

### Configuration Flow

1. **Model Config**: You specify `vllm_version` in the model's YAML configuration (optional)
2. **ConfigManager**: Automatically resolves the virtual environment path
3. **Pipeline**: Uses the appropriate virtual environment to start the vLLM server
4. **vLLM Server**: Runs with the specified vLLM version

## Quick Start

### 1. Use Default vLLM Version (Recommended)

Most models should use the default (latest) vLLM version:

```yaml
model:
  name: "your-model-name"

vllm:
  provider_config: "vllm_single_card"
  overrides:
    # No vllm_version specified = uses latest vLLM from main project
    max_model_len: 4096

tasks:
  - "spanish"
  - "portuguese"
```

### 2. Use Specific vLLM Version (When Needed)

Only specify a version when you need a specific one:

```yaml
model:
  name: "tencent/Hunyuan-MT-7B"

vllm:
  provider_config: "vllm_single_card"
  overrides:
    kv_cache_memory: 0
    vllm_version: "0.8.0"  # Only specify when you need a specific version

tasks:
  - "spanish"
  - "portuguese"
  - "translation"
```

### 3. Run the Model

```bash
# The system automatically creates the vLLM 0.8.0 environment if needed
python eval.py --config configs/single_card/Hunyuan-MT-7B.yaml --test
```

## Virtual Environment Structure

```
benchy/
├── .venv/                    # Main project environment (latest vLLM - DEFAULT)
├── venvs/                    # Version-specific environments (created automatically)
│   ├── vllm_0_8_0/         # vLLM 0.8.0 environment (created when needed)
│   └── vllm_0_10_2/        # vLLM 0.10.2 environment (created when needed)
└── configs/
    └── single_card/
        └── Hunyuan-MT-7B.yaml
```

## Configuration Options

### Model Configuration

```yaml
vllm:
  provider_config: "vllm_single_card"  # Base provider config
  overrides:
    # Optional: specify vLLM version (defaults to latest)
    vllm_version: "0.8.0"
    # ... other overrides
```

### When to Specify vLLm Version

Only specify `vllm_version` when:
- **Model compatibility**: The model requires a specific vLLM version
- **Feature requirements**: You need features from a specific vLLM version
- **Testing**: You want to test with a different vLLM version
- **Bug fixes**: You need a specific version that fixes a known issue

## Management Commands (Optional)

The system automatically manages environments, but you can use these commands for manual management:

### List Available Environments

```bash
python scripts/manage_vllm_venvs.py list
```

### Create Environment Manually

```bash
# Create vLLM 0.8.0 environment (usually not needed - created automatically)
python scripts/manage_vllm_venvs.py create 0.8.0
```

### Get Environment Info

```bash
python scripts/manage_vllm_venvs.py info 0.8.0
```

## Backward Compatibility

- **Existing configurations** without `vllm_version` continue to work using the latest vLLM version
- **No breaking changes** to existing model configurations
- **Gradual migration** - you can update models one by one

## Example: Hunyuan-MT-7B with vLLM 0.8.0

The `Hunyuan-MT-7B.yaml` configuration specifies vLLM 0.8.0:

```yaml
model:
  name: "tencent/Hunyuan-MT-7B"

vllm:
  provider_config: "vllm_single_card"
  overrides:
    kv_cache_memory: 0
    vllm_version: "0.8.0"  # Specific version for this model

tasks:
  - "spanish"
  - "portuguese"
  - "translation"
```

To use this configuration:

```bash
# The system automatically creates the vLLM 0.8.0 environment if needed
python eval.py --config configs/single_card/Hunyuan-MT-7B.yaml --test
```

## Benefits

- **Automatic Management**: Environments are created when needed
- **Default to Latest**: Most models use the latest vLLM version by default
- **Version Isolation**: Different models can use different vLLM versions when needed
- **Backward Compatible**: Existing configurations continue to work
- **Non-Docker**: No container overhead, direct GPU access
- **uv Integration**: Leverages your existing `uv` workflow

## Troubleshooting

### Environment Creation Fails

If automatic environment creation fails, the system falls back to the main project environment and logs the error.

### Check Environment Status

```bash
python scripts/manage_vllm_venvs.py list
```

### Manual Environment Creation

If you need to create an environment manually:

```bash
python scripts/manage_vllm_venvs.py create 0.8.0
```