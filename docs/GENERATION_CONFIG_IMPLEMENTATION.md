# Generation Config Implementation Summary

## Problem Statement

Benchy needed to automatically fetch and use `generation_config.json` from model repositories to ensure all evaluations use the model's recommended generation parameters.

## Solution Overview

A minimal, clean implementation that:
1. Fetches `generation_config.json` from HuggingFace before running evaluations
2. Passes parameters to lm-evaluation-harness via command-line arguments
3. Saves the config to output directories for reproducibility

## Implementation Details

### New Files Created

#### `src/generation_config.py` (122 lines)
A single utility module containing three functions:

1. **`fetch_generation_config()`** - Downloads config from HuggingFace Hub
   - Uses `huggingface_hub.hf_hub_download()`
   - Returns `None` if config not found (no error)
   - Logs success/failure appropriately

2. **`save_generation_config()`** - Saves config to output directory
   - Creates JSON file in model output directory
   - For logging and reproducibility

3. **`format_generation_params_for_lm_eval()`** - Converts config to command-line format
   - Maps generation config keys to lm-eval parameter names
   - Returns comma-separated string for `--model_args`
   - Supports: temperature, top_p, top_k, repetition_penalty, max_tokens, do_sample, seed

### Files Modified

#### `src/pipeline.py` (3 changes)
1. Added import: `from .generation_config import fetch_generation_config, save_generation_config`
2. Fetch config before starting vLLM server (Step 0)
3. Save config to output directory
4. Pass `generation_config` to all task configs (spanish, portuguese, translation)

#### `src/tasks/lm_harness.py` (3 changes)
1. Added import: `from ..generation_config import format_generation_params_for_lm_eval`
2. Extract `generation_config` from task_config
3. Format and append generation parameters to `model_args_parts`
4. Log generation config usage

## Code Changes Summary

**Lines added:** ~150 total
- New module: 122 lines
- Pipeline changes: ~10 lines
- Task harness changes: ~10 lines

**Complexity:** Minimal
- No new dependencies (huggingface_hub already included)
- No new database tables or state management
- Pure functional implementation

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

## Example Command Generated

**Before** (without generation config):
```bash
lm_eval --model local-completions \
  --model_args "model=tencent/Hunyuan-MT-7B,max_length=2048,base_url=http://0.0.0.0:20501/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False,tokenizer_backend=huggingface" \
  --tasks latam_es --batch_size 4 --output_path /home/mauro/dev/benchy/outputs/benchmark_outputs/Hunyuan-MT-7B/spanish
```

**After** (with generation config):
```bash
lm_eval --model local-completions \
  --model_args "model=tencent/Hunyuan-MT-7B,max_length=2048,base_url=http://0.0.0.0:20501/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False,tokenizer_backend=huggingface,temperature=0.7,top_k=20,top_p=0.6,repetition_penalty=1.05,do_sample=True" \
  --tasks latam_es --batch_size 4 --output_path /home/mauro/dev/benchy/outputs/benchmark_outputs/Hunyuan-MT-7B/spanish
```

## Output Files

New file in output directory:
```
outputs/benchmark_outputs/<run_id>/<model_name>/
└── generation_config.json    # Model's generation config (NEW)
```

## Logging Examples

```
INFO | benchy.pipeline | Fetching generation_config.json for tencent/Hunyuan-MT-7B
INFO | benchy.pipeline | Successfully loaded generation_config.json for tencent/Hunyuan-MT-7B
INFO | benchy.lm_eval  | Using generation config: {'temperature': 0.7, 'top_k': 20, ...}
INFO | benchy.lm_eval  | Added generation config parameters: temperature=0.7,top_k=20,top_p=0.6,...
```

## Error Handling

- **Model without generation_config.json:** Logs info message, continues normally
- **HuggingFace API error:** Logs warning, continues with default parameters
- **Invalid config format:** Skips problematic parameters, uses valid ones

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
3. **Minimal:** Only ~150 lines of code, no new dependencies
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

1. **Model config override:** Allow specifying generation_config in model YAML
2. **Task-specific params:** Override generation params per task
3. **Validation:** Check parameter ranges are valid
4. **Local files:** Support loading generation_config.json from local path
5. **vLLM integration:** Also pass params to vLLM server startup (in addition to lm-eval)

