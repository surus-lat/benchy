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

3. Configure your benchmark in `config.yaml` (already set up with defaults from tasks.md)

## Usage

Run the complete benchmark pipeline:

```bash
python main.py
```

This will:
1. Run lm-evaluation-harness with the configured model and tasks
2. Upload the results using the leaderboard script
3. Log everything through ZenML

## Configuration

Edit `config.yaml` to change:
- Model name and parameters
- Evaluation tasks and settings  
- Output paths
- Virtual environment paths

## Architecture

- `src/steps.py`: ZenML steps for evaluation and upload
- `src/pipeline.py`: Main ZenML pipeline connecting the steps
- `main.py`: Entry point that loads config and runs pipeline
- `config.yaml`: Configuration for models, tasks, and paths

## Requirements

- Both lm-evaluation-harness and leaderboard repos set up with their respective virtual environments
- Environment variables for HF_TOKEN and WANDB_API_KEY (if needed)
- ZenML initialized in the project
