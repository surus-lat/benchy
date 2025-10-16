# Generation Config Support

## Overview

Benchy automatically fetches and uses `generation_config.json` from model repositories to ensure evaluations use the model's recommended generation parameters. This feature works transparently - no configuration changes are needed.

## How It Works

### 1. Automatic Fetching

When a benchmark pipeline starts, Benchy automatically:
- Fetches `generation_config.json` from the model's HuggingFace repository
- Saves it to the output directory for reproducibility
- Passes the parameters to lm-evaluation-harness via command-line arguments

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

## Usage

No changes needed to existing configs! The feature works automatically:

```bash
# Just run as normal
python eval.py -c configs/single_card/Hunyuan-MT-7B.yaml

# Generation config is automatically:
# 1. Fetched from HuggingFace
# 2. Applied to all evaluation tasks
# 3. Saved to output directory
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

Example log output:
```
INFO | benchy.pipeline | Fetching generation_config.json for tencent/Hunyuan-MT-7B
INFO | benchy.pipeline | Successfully loaded generation_config.json for tencent/Hunyuan-MT-7B
INFO | benchy.lm_eval  | Using generation config: {'temperature': 0.7, 'top_k': 20, ...}
INFO | benchy.lm_eval  | Added generation config parameters: temperature=0.7,top_k=20,top_p=0.6,...
```

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

### Code Changes Summary
**Lines added:** ~130 total
- New module: 130 lines
- Pipeline changes: ~10 lines
- Task harness changes: ~10 lines

**Complexity:** Minimal
- No new dependencies (huggingface_hub already included)
- No new database tables or state management
- Pure functional implementation

## Testing

To test with a model that has generation_config.json:

```bash
# Test with AFM-4.5B (known to have generation config)
python eval.py -c configs/single_card/AFM-4.5B.yaml --limit 5

# Check logs for:
# - "Successfully loaded generation_config.json"
# - "Added generation config parameters"

# Check output directory for:
# - outputs/benchmark_outputs/<run_id>/AFM-4.5B/generation_config.json
```

## Benefits

1. **Automatic:** No manual configuration needed per model
2. **Reproducible:** Generation config saved with results
3. **Minimal:** Only ~130 lines of code, no new dependencies
4. **Future-proof:** Easy to extend for new tasks
5. **Backwards compatible:** Works with models that don't have generation_config.json

## Design Decisions

### Why fetch before starting vLLM server?
- Generation config is small and fast to fetch
- Doesn't require the model to be loaded
- Fail fast if there are HuggingFace connection issues

### Why pass via model_args instead of API server config?
- lm-evaluation-harness supports these parameters in model_args
- vLLM server can still be started with default parameters
- More flexible - different tasks could theoretically use different generation params

### Why not fail if generation_config.json is missing?
- Many models don't provide this file
- Better to continue with defaults than fail
- User can check logs to see if config was found

## Future Enhancements

Potential improvements:
1. **Model config override:** Allow specifying generation_config in model YAML
2. **Task-specific params:** Override generation params per task
3. **Validation:** Check parameter ranges are valid
4. **Local files:** Support loading generation_config.json from local path
5. **vLLM integration:** Also pass params to vLLM server startup (in addition to lm-eval)