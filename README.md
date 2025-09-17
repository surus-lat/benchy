# Benchy - vLLM-Powered ML Benchmarking

A streamlined ZenML-powered system for benchmarking ML models using vLLM API server and lm-evaluation-harness, with automatic results upload.

## Overview

Benchy runs a complete evaluation pipeline:
1. **Start vLLM server** with the specified model and configuration
2. **Test API connectivity** to ensure the server is working
3. **Run Spanish evaluation** tasks sequentially 
4. **Run Portuguese evaluation** tasks sequentially
5. **Upload results** to the leaderboard
6. **Clean up** - always stops the vLLM server

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Copy environment template and fill in your values:
   ```bash
   cp env.template .env
   # Edit .env with your HF_TOKEN if needed
   ```

3. Configure your benchmark in a YAML config file (see `configs/` directory)

4. Ensure ZenML server is running:
   ```bash
   sudo docker run -it -d -p 8080:8080 --name zenml zenmldocker/zenml-server
   ```

## Usage

### Single Model Runs

Run with default configuration:
```bash
python main.py
```

Run with specific configuration:
```bash
python main.py --config configs/gemma-e4b.yaml
python main.py -c configs/test-gemma.yaml
```

Show help:
```bash
python main.py --help
```

### Configuration

Create YAML config files in the `configs/` directory. Example configuration:

```yaml
# configs/my-model.yaml
model:
  name: "google/gemma-3n-E4B-it"

evaluation:
  tasks_spanish: "latam_es"      # Spanish evaluation tasks
  tasks_portuguese: "latam_pt"   # Portuguese evaluation tasks  
  batch_size: "20"
  output_path: "/home/mauro/dev/lm-evaluation-harness/output"
  log_samples: true
  cache_requests: true
  trust_remote_code: true
  num_concurrent: 20
  # limit: 5                     # Uncomment for testing (TEST mode)

# vLLM server configuration
vllm:
  host: "0.0.0.0"
  port: 8000
  tensor_parallel_size: 1        # Number of GPUs (-tp parameter)
  max_model_len: 8192
  gpu_memory_utilization: 0.6
  enforce_eager: true            # Better compatibility
  limit_mm_per_prompt: '{"images": 0, "audios": 0}'  # Disable multimodal
  hf_cache: "/home/mauro/.cache/huggingface"
  # hf_token: "hf_..."           # Set if needed

wandb:
  entity: "surus-lat"
  project: "LATAM-leaderboard"

upload:
  script_path: "/home/mauro/dev/leaderboard"
  script_name: "run_pipeline.py"

logging:
  log_dir: "logs"

venvs:
  lm_eval: "/home/mauro/dev/lm-evaluation-harness"
  leaderboard: "/home/mauro/dev/leaderboard"
```

### Available Configurations

- `configs/gemma-e4b.yaml` - Production Gemma model configuration
- `configs/test-gemma.yaml` - Test configuration with limited samples

### Test Mode

Add `limit: N` to the evaluation section to run in TEST mode:
- Only evaluates N examples per task
- Adds "TEST_" prefix to the run name
- Useful for validating configuration before full runs

## Pipeline Steps

The pipeline executes these steps in order:

1. **start_vllm_server** - Starts vLLM API server with model
2. **test_vllm_api** - Validates server is responding
3. **run_lm_evaluation** (Spanish) - Runs Spanish language tasks
4. **run_lm_evaluation** (Portuguese) - Runs Portuguese language tasks  
5. **upload_results** - Uploads combined results
6. **stop_vllm_server** - Always stops server (guaranteed cleanup)

## Features

- **Automatic server management** - vLLM server is always cleaned up
- **Sequential task execution** - Spanish and Portuguese tasks run separately
- **Safe logging** - Handles parallel logging conflicts gracefully
- **Configurable vLLM settings** - Full control over server parameters
- **Error resilience** - Server cleanup happens even if pipeline fails
- **Test mode support** - Quick validation with limited samples

## Logging

All runs generate detailed logs in the `logs/` directory:
- Console output with real-time progress
- File logs with complete execution details
- Separate logs for each pipeline step
- Safe error handling for logging conflicts

## Environment Variables

Set in `.env` file:
- `HF_TOKEN` - Hugging Face token (if needed for model access)
- `BENCHY_CONFIG` - Default config file path (optional)

## Troubleshooting

### vLLM Server Issues
- Check GPU memory usage
- Reduce `gpu_memory_utilization` in config
- Verify model fits in available memory
- Check logs for CUDA errors

### Evaluation Failures
- Verify lm-evaluation-harness virtual environment is set up
- Check task names are correct
- Ensure API server is responding
- Review batch size settings

### Server Cleanup
The pipeline guarantees vLLM server cleanup. If you see warnings about manual cleanup, check running processes:
```bash
ps aux | grep vllm
# Kill any remaining vLLM processes if needed
```

## Development

The codebase is intentionally minimal and focused:
- `main.py` - Entry point and configuration handling
- `src/pipeline.py` - Main ZenML pipeline definition
- `src/steps.py` - Individual pipeline steps
- `src/logging_utils.py` - Logging utilities
- `configs/` - Configuration files