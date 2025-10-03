# Generation Config Support

## Overview

Benchy now automatically fetches and uses `generation_config.json` from model repositories to ensure evaluations use the model's recommended generation parameters.

## How It Works

### 1. Automatic Fetching

When a benchmark pipeline starts, Benchy automatically:
- Fetches `generation_config.json` from the model's HuggingFace repository
- Saves it to the output directory for reproducibility
- Passes the parameters to lm-evaluation-harness

### 2. Supported Parameters

The following generation parameters are extracted and used:
- `temperature` - Controls randomness in generation
- `top_p` - Nucleus sampling threshold
- `top_k` - Top-k sampling parameter
- `repetition_penalty` - Penalty for token repetition
- `max_tokens` / `max_new_tokens` - Maximum tokens to generate
- `do_sample` - Whether to use sampling
- `seed` - Random seed for reproducibility

### 3. Where It's Applied

Generation parameters are automatically included in all evaluation tasks:
- Spanish language evaluation
- Portuguese language evaluation
- Translation evaluation
- Any future tasks added to the system

## Example

For a model like `arcee-ai/AFM-4.5B` with this `generation_config.json`:

```json
{
  "eos_token_id": [127960, 127967],
  "pad_token_id": 127961,
  "do_sample": true,
  "top_k": 20,
  "top_p": 0.6,
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "transformers_version": "4.56.0"
}
```

The lm_eval command will automatically include:
```bash
lm_eval --model local-completions \
  --model_args "model=arcee-ai/AFM-4.5B,...,temperature=0.7,top_k=20,top_p=0.6,repetition_penalty=1.05,do_sample=True" \
  ...
```

## Output Files

After running a benchmark, you'll find:
```
outputs/benchmark_outputs/<run_id>/<model_name>/
├── generation_config.json    # Model's generation config (if available)
├── spanish/                   # Spanish evaluation results
├── portuguese/               # Portuguese evaluation results
└── translation/              # Translation results
```

## Logging

Generation config information is logged at multiple points:
1. When fetched: `"Successfully loaded generation_config.json for <model>"`
2. Before evaluation: `"Using generation config: {...}"`
3. When added to command: `"Added generation config parameters: ..."`

## Fallback Behavior

If a model doesn't have a `generation_config.json`:
- Benchy logs: `"No generation_config.json found for <model>"`
- Evaluation continues with default lm-evaluation-harness parameters
- No error is raised - this is expected for some models

## Implementation Details

### Files Modified
- `src/generation_config.py` - New utility module for fetching and formatting
- `src/pipeline.py` - Fetches config and passes to tasks
- `src/tasks/lm_harness.py` - Includes parameters in lm_eval commands

### Key Functions
- `fetch_generation_config()` - Downloads config from HF Hub
- `save_generation_config()` - Saves config to output directory
- `format_generation_params_for_lm_eval()` - Converts to command-line format

## Future Enhancements

Potential improvements:
1. Allow per-task generation config overrides
2. Support additional generation parameters as lm-eval adds them
3. Add validation for parameter ranges
4. Support local generation config files (not just from HF Hub)

