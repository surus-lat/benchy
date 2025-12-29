# Generation Config Support

## Overview

Benchy can fetch `generation_config.json` from Hugging Face model repositories and save it alongside benchmark outputs. This keeps model-recommended generation settings available for reproducibility without requiring external tools.

## How It Works

### 1. Automatic Fetching

When a benchmark pipeline starts, Benchy:
- Downloads `generation_config.json` from the model's Hugging Face repository (if present)
- Saves it to the run output directory
- Stores the config in `task_config["generation_config"]` for any tasks that choose to use it

### 2. Parameters Captured

The full `generation_config.json` file is saved; Benchy does not filter or rewrite it. Tasks or providers can opt in to use any subset of those settings.

### 3. Where It's Applied

Benchy does not automatically apply generation parameters to every task. Instead, tasks that care about model defaults can read `task_config["generation_config"]` and decide how to apply them.

## Usage

No config changes are required. Benchy fetches and saves the config automatically when running:

```bash
python eval.py -c configs/models/Hunyuan-MT-7B.yaml
```

## Output Files

After running a benchmark, you'll find:

```
outputs/benchmark_outputs/<run_id>/<model_name>/
├── generation_config.json    # Model's generation config (if available)
├── spanish/
├── portuguese/
└── translation/
```

## Logging

Fetch behavior is logged:

- When fetched: `Successfully loaded generation_config.json for <model>`
- When missing: `No generation_config.json found for <model>`

## Implementation Details

### Files
- `src/generation_config.py` - Fetch and save utilities
- `src/pipeline.py` - Loads the config and passes it into task configs

### Key Functions
- `fetch_generation_config()` - Downloads config from the Hub
- `save_generation_config()` - Saves config to the output directory

## Design Notes

- Missing configs are normal and do not fail the run.
- The file is saved for reproducibility even if no task uses it today.
