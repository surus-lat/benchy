# Benchy - LATAM Leaderboard Benchmarking Suite

A modular benchmarking suite for evaluating Large Language Models (LLMs) on Spanish and Portuguese tasks for the LATAM Leaderboard. Features a clean, task-based configuration system with vLLM integration and Prefect orchestration.

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
- **lm-evaluation-harness**: For Spanish task evaluations
- **Portuguese benchmark suite**: For Portuguese task evaluations
- **Prefect server**: Running in Docker container (recommended port 4200)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd benchy
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Set up external evaluation environments**:
   
   For Spanish evaluations (lm-evaluation-harness):
   ```bash
   cd /path/to/lm-evaluation-harness
   uv pip install -e .[api]
   ```
   
   For Portuguese evaluations:
   ```bash
   cd /path/to/portuguese-bench
   uv pip install -e ".[anthropic,openai,sentencepiece]"
   ```

4. **Start Prefect server** (Docker recommended):
   ```bash
   docker run -d --name prefect-server -p 4200:4200 prefecthq/prefect:2-python3.11
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
│   └── translation.yaml     # Translation config (ready)
└── models/
    └── qwen-4b.yaml         # Model config (minimal)
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
  logs: "logs"

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

# Custom config path
BENCHY_CONFIG=configs/my-model.yaml
```

## 🚀 Usage

### Basic Usage

```bash
# Run full evaluation
python eval.py --config configs/models/qwen-4b.yaml

# Test with limited samples (perfect for testing!)
python eval.py --config configs/models/qwen-4b.yaml --limit 10

# Test vLLM server only (no evaluation)
python eval.py --config configs/models/qwen-4b.yaml --test

# Run only Portuguese evaluation
python eval.py --config configs/templates/test-model_new.yaml

# Verbose logging
python eval.py --config configs/models/qwen-4b.yaml --verbose
```

### Command Line Options

- `--config` / `-c`: Path to model configuration file
- `--limit`: Limit number of examples per task (useful for testing)
- `--test`: Test vLLM server only, no evaluation
- `--verbose` / `-v`: Enable verbose logging
- `--register` / `-r`: Register flows with Prefect server

### Batch Model Evaluation

Use the provided script to run multiple models sequentially:

```bash
# Edit run_models.sh to include your config files
./run_models.sh
```

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
│   ├── tasks/              # Task-specific configs
│   ├── models/             # Model configs (minimal)
│   └── templates/          # Example configs
├── src/                   # Source code
│   ├── pipeline.py        # Main Prefect pipeline
│   ├── config_manager.py  # Configuration management
│   ├── inference/         # vLLM server management
│   ├── tasks/             # Evaluation tasks
│   └── logging_utils.py   # Logging utilities
├── outputs/              # Evaluation results
├── logs/                # Log files
├── external/            # External dependencies
├── eval.py             # Main entry point
└── scripts/            # Utility scripts
```

## 🔧 Advanced Configuration

### Adding New Tasks

1. Create task config in `configs/tasks/`:
```yaml
# configs/tasks/my_task.yaml
task_name: "my_custom_task"
lm_eval_path: "/path/to/evaluation/suite"
defaults:
  batch_size: "4"
  num_concurrent: 8
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
python eval.py --config configs/models/qwen-4b.yaml --limit 10
```

## 📈 Monitoring and Logging

- **Prefect Dashboard**: Access at `http://localhost:4200` to monitor pipeline execution
- **Log Files**: Detailed logs stored in `logs/` directory (configurable in `configs/config.yaml`)
- **Output Files**: Results stored in centralized output directory

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
python eval.py --verbose --config configs/models/qwen-4b.yaml --limit 5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation framework
- [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
- [Prefect](https://www.prefect.io/) for workflow orchestration
- LATAM community for benchmark development

---

For questions or support, please open an issue or contact the maintainers.
