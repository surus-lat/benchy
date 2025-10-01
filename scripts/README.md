# Scripts Directory

This directory contains utility scripts for managing the benchy project.

## Available Scripts

### `manage_vllm_venvs.py`

Optional utility for managing vLLM virtual environments. The system automatically creates environments when needed, so this script is only for manual management.

**Usage:**
```bash
# From the project root
python3 scripts/manage_vllm_venvs.py list
python3 scripts/manage_vllm_venvs.py create 0.8.0
python3 scripts/manage_vllm_venvs.py info 0.8.0
```

**Actions:**
- `list` - List all available vLLM versions with virtual environments
- `create <version>` - Create a virtual environment for a specific vLLM version
- `info <version>` - Get information about a specific vLLM version environment

**Options:**
- `--base-dir` - Base directory for virtual environments (defaults to project root/venvs)
- `--force` - Force recreate the virtual environment if it already exists

## Running Scripts

All scripts should be run from the project root directory:

```bash
cd /path/to/benchy
python3 scripts/script_name.py [arguments]
```
