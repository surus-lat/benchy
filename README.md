<h1 align="center"><strong>Benchy</strong></h1>
<p align="center"> LATAM Leaderboard Benchmarking Suite </p>

<p align="center">
    <img src="./docs/benchy_2.png" alt="readme_image" style="width:220px;height:220px;" />
</p>


A modular benchmarking suite for evaluating Large Language Models (LLMs) on Spanish and Portuguese tasks for the [LATAM Leaderboard](https://latamboard.ai/). Features a clean, task-based configuration system with vLLM integration and Prefect orchestration.

## 🚀 Features

- **Modular Task System**: Separate configs for Spanish, Portuguese, and translation tasks
- **vLLM Integration**: High-performance model serving with GPU optimization
- **Centralized Configuration**: Global settings in one place, minimal model configs
- **Command Line Control**: Easy testing with `--limit` parameter
- **Prefect Orchestration**: Robust workflow management with monitoring
- **Clean Architecture**: Decoupled components for easy maintenance

## 📋 Prerequisites

### System Requirements
- Python 3.12+
- CUDA-compatible GPU(s)
- Docker (for Prefect server)
- Sufficient disk space for model downloads and outputs

### External Dependencies
- **Prefect server**: Running in Docker container (recommended port 4200)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd benchy
   git submodule update --init --recursive
   ```

2. **Install dependencies**:
   ```bash
   # use our setup script (recommended)
   bash setup.sh

   # ALTERNATIVE: Using uv
   uv sync
   ```

3. **Start Prefect server** (Docker recommended):
   ```bash
   sudo docker run -p 4200:4200 -d --rm prefecthq/prefect:3-python3.12 prefect server start --host 0.0.0.0
   ```

## ⚙️ Configuration

### Configuration Structure

The system uses a modular configuration approach:

```
configs/
├── config.yaml              # Global settings (paths, logging)
├── providers/
│   └── vllm_single_card.yaml # vLLM server defaults
├── tasks/
│   ├── spanish.yaml         # Spanish evaluation config
│   ├── portuguese.yaml      # Portuguese evaluation config
│   └── translation.yaml     # Translation config (ready for use)
├── single_card/             # Pre-configured model configs
│   ├── aya8b.yaml
│   ├── llama3.1.yaml
│   ├── qwen34b.yaml
│   └── ... (15+ models ready)
└── templates/
    └── test-model_new.yaml  # Template for new models
```

### Model Configuration

Create a minimal YAML configuration for each model:

```yaml
# Model information
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"

# vLLM configuration - references base provider config
vllm:
  provider_config: "vllm_single_card"  # Base config
  overrides:
    max_model_len: 2000  # Override only what's different

# Tasks to run
tasks:
  - "spanish"
  - "portuguese"
```

### Global Configuration (`configs/config.yaml`)

```yaml
# Central Configuration File
paths:
  benchmark_outputs: "/home/mauro/dev/benchmark_outputs"
  reference_dir: "/home/mauro/dev/benchy/reference"
  publish_dir: "/home/mauro/dev/publish"
  logs: "logs"

# Hugging Face datasets
datasets:
  results: "LatamBoard/leaderboard-results"

# Default evaluation settings
evaluation:
  default_limit: null  # No limit by default, can be overridden by --limit

# Logging configuration
logging:
  log_dir: "logs"
```

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# Prefect configuration
PREFECT_API_URL=http://localhost:4200/api

# Hugging Face token (if needed)
HF_TOKEN=your_huggingface_token

```

## 🚀 Usage

### Quick Start

1. **Choose a model** from the 15+ pre-configured options in `configs/single_card/`
2. **Test with limited samples**:
   ```bash
   python eval.py --config configs/single_card/qwen34b.yaml --limit 10
   ```
3. **Run full evaluation**:
   ```bash
   python eval.py --config configs/single_card/qwen34b.yaml
   ```
4. **Process results**:
   ```bash
   python ./src/leaderboard/process_all.py
   ```

### Basic Usage

```bash
# Run full evaluation
python eval.py --config configs/single_card/qwen34b.yaml

# Test with limited samples (perfect for testing!)
python eval.py --config configs/single_card/qwen34b.yaml --limit 10

# Test vLLM server only (no evaluation)
python eval.py --config configs/single_card/qwen34b.yaml --test

# Run any of the 15+ pre-configured models
python eval.py --config configs/single_card/llama3.1.yaml --limit 5

# Verbose logging
python eval.py --config configs/single_card/qwen34b.yaml --verbose

# Custom run ID for organized outputs
python eval.py --config configs/single_card/qwen34b.yaml --run-id my_experiment_001
```

### Command Line Options

- `--config` / `-c`: Path to model configuration file
- `--limit`: Limit number of examples per task (useful for testing)
- `--test`: Test vLLM server only, no evaluation
- `--verbose` / `-v`: Enable verbose logging
- `--register` / `-r`: Register flows with Prefect server
- `--run-id`: Custom run ID for organizing outputs (default: auto-generated timestamp)

### Batch Model Evaluation

Use the provided scripts to run multiple models sequentially:

#### Full Evaluation (`run_models.sh`)

Run complete evaluations on multiple models:

```bash
# Run all models in configs/single_card/
./run_models.sh

# Run with custom run ID for organized outputs
./run_models.sh --run-id my_experiment_001

# Run quietly (suppress detailed output)
./run_models.sh --quiet

# Run specific model
./run_models.sh my-model.yaml

# Run models from a list file
./run_models.sh my_model_list.txt

# Combine options
./run_models.sh --quiet --run-id production_batch_2024
```

#### Testing Models (`test_models.sh`)

Quickly test all model configurations to verify they work:

```bash
# Test all models (quick vLLM server tests)
./test_models.sh

# Test with limited evaluation (10 samples per task)
./test_models.sh --limited

# Test quietly
./test_models.sh --quiet

# Test specific model
./test_models.sh my-model.yaml

# Limited evaluation on specific model
./test_models.sh --limited --quiet my-model.yaml
```

**Script Features:**
- **Automatic Discovery**: Finds all `.yaml` files in `configs/single_card/`
- **Custom Run IDs**: Use `--run-id` to organize outputs
- **Detailed Summaries**: Shows which models passed/failed with model names
- **Flexible Testing**: Quick tests or limited evaluations
- **Background Execution**: Use `nohup` for long-running batches

### Download Models

Pre-download models to avoid download delays during evaluation:

```bash
# Edit download_models.sh with your model list
./download_models.sh
```

## 📊 Pipeline Overview

The modular pipeline consists of:

1. **Configuration Loading**: Loads model config, merges with provider defaults
2. **vLLM Server Startup**: Launches server with merged configuration
3. **API Testing**: Verifies server is responding correctly
4. **Task Execution**: Runs only the tasks specified in model config
   - Spanish evaluation (if `"spanish"` in tasks list)
   - Portuguese evaluation (if `"portuguese"` in tasks list)
5. **Result Gathering**: Collects and processes evaluation results
6. **Server Cleanup**: Stops vLLM server and cleans up resources

## 📁 Directory Structure

```
benchy/
├── configs/                 # Configuration files
│   ├── config.yaml         # Global settings
│   ├── providers/          # vLLM provider configs
│   ├── single_card/        # Pre-configured model configs (15+ models)
│   └── templates/          # Example configs
├── src/                   # Source code
│   ├── pipeline.py        # Main Prefect pipeline
│   ├── config_manager.py  # Configuration management
│   ├── inference/         # vLLM server management
│   ├── tasks/             # Evaluation tasks
│   ├── leaderboard/       # Results processing
│   └── logging_utils.py   # Logging utilities
├── outputs/              # Evaluation results
├── logs/                # Log files
├── external/            # Legacy resources (optional)
├── eval.py             # Main entry point
└── scripts/            # Utility scripts
```

## 🔧 Advanced Configuration

### Adding New Tasks

1. Create task config in `src/tasks/my_task/task.json`:
```json
{
  "name": "my_custom_task",
  "description": "Describe the new task",
  "defaults": {
    "batch_size": 20,
    "log_samples": false
  }
}
```

2. Add task to model config:
```yaml
tasks:
  - "spanish"
  - "portuguese" 
  - "my_task"  # New task
```

### GPU Memory Optimization

Override vLLM settings in model config:

```yaml
vllm:
  provider_config: "vllm_single_card"
  overrides:
    gpu_memory_utilization: 0.6  # Reduce from default
    kv_cache_memory: 12934271795  # Explicit KV cache
    tensor_parallel_size: 1       # Single GPU
```

### Testing with Limited Samples

Use the `--limit` parameter for quick testing:

```bash
# Test with 10 samples per task
python eval.py --config configs/single_card/qwen34b.yaml --limit 10
```

## 📈 Monitoring and Logging

- **Prefect Dashboard**: Access at `http://localhost:4200` to monitor pipeline execution
- **Log Files**: Detailed logs stored in `logs/` directory (configurable in `configs/config.yaml`)
- **Output Files**: Results stored in centralized output directory

## 📊 Results Processing

After running evaluations, process the results for the leaderboard:

```bash
# Process all model results and generate leaderboard
python ./src/leaderboard/process_all.py
```

This will:
1. **Parse Results**: Extract scores from Spanish and Portuguese evaluations
2. **Generate Tables**: Create JSON and CSV leaderboard tables
3. **Copy References**: Include task definitions and metadata
4. **Prepare Upload**: Ready files for Hugging Face dataset upload

### Output Structure

```
publish/
├── leaderboard_table.json    # Main results (JSON)
├── leaderboard_table.csv     # Results (CSV)
├── summaries/                # Individual model summaries
│   ├── model1_summary.json
│   └── all_model_summaries.json
├── tasks_list.json          # Task definitions
└── tasks_groups.json        # Task groupings
```

## 🐛 Troubleshooting

### Common Issues

1. **Prefect Server Connection**:
   ```bash
   # Check if Prefect server is running
   curl http://localhost:4200/api/health
   ```

2. **GPU Memory Issues**:
   - Reduce `gpu_memory_utilization`
   - Decrease `batch_size`
   - Lower `num_concurrent`

3. **Model Download Issues**:
   - Check Hugging Face token
   - Verify model name spelling
   - Ensure sufficient disk space

4. **Port Conflicts**:
   - Change vLLM port in configuration
   - Kill existing vLLM processes: `pkill -f vllm`

### Debug Mode

Run with verbose logging for detailed debugging:

```bash
python eval.py --verbose --config configs/single_card/qwen34b.yaml --limit 5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- [Prefect](https://www.prefect.io/) for workflow orchestration
- [Surus](https://surus.lat/) for starting this project
- LATAM community for benchmark development

---

For questions or support, please open an issue or contact the maintainers.
