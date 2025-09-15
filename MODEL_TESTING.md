# Model Download and Testing Feature

This document describes the new model download and testing step added to the Benchy pipeline.

## Overview

The pipeline now includes a pre-evaluation step that:
1. Downloads the specified model using the same configuration as the evaluation step
2. Runs a simple "Hello, world!" test to verify the model loads and works correctly
3. Extracts model metadata (parameter count, estimated size)
4. Catches configuration errors early before the full evaluation

## Benefits

- **Separates download time from evaluation time**: Model download happens in a dedicated step
- **Early error detection**: Configuration issues are caught before running expensive evaluations
- **Better time tracking**: ZenML tracks download time separately from evaluation time
- **Model validation**: Ensures the model works with your specific configuration before evaluation

## How it Works

### 1. Test Script (`lm-evaluation-harness/scripts/test_model.py`)

A Python script that:
- Loads the model with the same arguments as the evaluation
- Runs a simple text generation test
- Extracts model metadata
- Uses the same virtual environment as lm-evaluation-harness

### 2. ZenML Step (`test_model_download`)

A ZenML step that:
- Calls the test script with the same model configuration
- Streams output in real-time
- Captures model metadata for the pipeline
- Fails fast if there are issues

### 3. Updated Pipeline

The benchmark pipeline now has 3 steps:
1. **Model Test**: Download and test the model
2. **Evaluation**: Run the actual benchmark evaluation
3. **Upload**: Upload results to the leaderboard

## Configuration

No additional configuration is needed. The test step uses the same model configuration from your YAML config file.

Example from `model-test.yaml`:
```yaml
model:
  name: "openai/gpt-oss-20b"
  dtype: "float16" 
  max_length: 16384

performance:
  use_accelerate: true
  num_gpus: 2
  mixed_precision: "no"
```

## Output

The test step provides:
- Model parameter count
- Estimated model size in GB
- Test generation output
- Configuration validation

Example output:
```
✅ Model test successful: openai/gpt-oss-20b
   Parameters: 20,000,000,000
   Size: 37.25 GB
   Generated: Hello, world! This is a test. How are you doing today?
```

## Error Handling

If the model test fails, the pipeline stops before running the expensive evaluation. Common failure scenarios:
- Model not found or access denied
- Insufficient GPU memory
- Invalid model configuration
- Missing HuggingFace token

## Environment Variables

Make sure your HuggingFace token is available:
```bash
export HF_TOKEN=your_huggingface_token
```

The test script will use this token for downloading private models or models requiring authentication.

## Logging

The test step creates detailed logs in the benchy log files:
- All test output is captured and logged
- Model metadata is extracted and stored
- Failures are logged with full error details

## Performance Impact

The test step adds minimal overhead:
- Model download happens once (cached for evaluation step)
- Simple generation test takes seconds
- Early failure detection saves time on problematic configurations

## Manual Testing

You can run the test script manually for debugging:

```bash
cd /path/to/lm-evaluation-harness
source .venv/bin/activate
python scripts/test_model.py --model_name "microsoft/DialoGPT-small" --model_args "pretrained=microsoft/DialoGPT-small,dtype=float16"
```

Or with accelerate:
```bash
python scripts/test_model.py --model_name "openai/gpt-oss-20b" --model_args "pretrained=openai/gpt-oss-20b,dtype=float16" --use_accelerate --num_gpus 2
```
