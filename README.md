# Benchy - ZenML-Powered ML Benchmarking

A minimal ZenML-powered system for benchmarking ML models using lm-evaluation-harness and uploading results to datasets.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Copy environment template and fill in your values:
   ```bash
   cp env.template .env
   # Edit .env with your tokens
   ```

3. Configure your benchmark in a YAML config file

## Usage

### Single Model Runs

Run with default configuration:
```bash
python main.py
```

Run with specific configuration:
```bash
python main.py --config configs/example-with-limit.yaml
python main.py -c configs/model-1-qwen-4b-instruct-2507.yaml
```

Show help:
```bash
python main.py --help
```

### Batch Runs

Run multiple models sequentially:
```bash
# Simple shell script (recommended)
./run_models.sh

# Python batch runner
python run_batch.py

# Advanced ZenML pipeline
python batch_runner.py
```

## Configuration

### Single Model Config Files

Create or modify YAML files in the `configs/` directory:

```yaml
# configs/my-model.yaml
model:
  name: "google/gemma-3n-E4B-it"
  dtype: "float16" 
  max_length: 16384

evaluation:
  tasks: "latam"
  device: "cuda"
  batch_size: "auto:4"
  output_path: "/home/mauro/dev/lm-evaluation-harness/output"
  log_samples: true
  limit: 10  # Optional: limit examples for testing

# Logging configuration
logging:
  log_dir: "logs"  # Directory to store log files

wandb:
  entity: "surus-lat"
  project: "LATAM-leaderboard"

upload:
  script_path: "/home/mauro/dev/leaderboard"
  script_name: "run_pipeline.py"

venvs:
  lm_eval: "/home/mauro/dev/lm-evaluation-harness"
  leaderboard: "/home/mauro/dev/leaderboard"
```

### Available Examples

- `configs/config-template.yaml` - Base template
- `configs/example-with-limit.yaml` - Quick testing with limit
- `configs/model-*.yaml` - Specific model configurations

## Features

### Command Line Interface

```bash
python main.py [OPTIONS]

Options:
  -c, --config PATH    Path to configuration YAML file
  -v, --verbose        Enable verbose logging  
  -h, --help          Show help message

Examples:
  python main.py                                    # Use default config.yaml
  python main.py --config configs/my-model.yaml    # Use specific config
  python main.py -c configs/example-with-limit.yaml # Short form
```

### Testing vs Production

**Quick Testing (with `--limit`):**
```yaml
evaluation:
  limit: 10  # Only evaluate 10 examples per task
```

**Full Evaluation:**
```yaml
evaluation:
  # limit: 10  # Comment out or remove for full evaluation
```

### Real-time Output & Logging

The system streams lm-evaluation-harness output in real-time and saves everything to log files:

**Console Output:**
```
[run_lm_evaluation] [lm_eval] Loading model...
[run_lm_evaluation] [lm_eval] Running task: latam
[run_lm_evaluation] [lm_eval] Progress: 5/10 samples
```

**Automatic File Logging:**
- **Single runs**: `logs/benchy_{model_name}_{timestamp}.log`
- **Batch runs**: `logs/batch_run_{timestamp}.log` + individual model logs
- **Complete logs**: All command output, errors, timing, and configuration

**Log Location:**
```bash
logs/
├── benchy_google_gemma-3n-E4B-it_20250112_143022.log
├── benchy_Qwen_Qwen3-4B-Instruct-2507_20250112_144155.log
└── batch_run_20250112_140000.log
```

## Architecture

- `src/steps.py`: ZenML steps for evaluation and upload
- `src/pipeline.py`: Main ZenML pipeline connecting the steps
- `src/batch_pipeline.py`: Multi-model batch processing pipeline
- `main.py`: CLI entry point for single model runs
- `*_runner.py`: Different batch execution approaches
- `configs/`: Configuration files for different models

## Requirements

- Both lm-evaluation-harness and leaderboard repos set up with their respective virtual environments
- Environment variables for HF_TOKEN and WANDB_API_KEY (if needed)
- ZenML initialized in the project

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on extending the system, adding new lm-evaluation-harness parameters, and best practices.