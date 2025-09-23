# Benchy - LATAM Leaderboard Benchmarking Suite

A comprehensive benchmarking suite for evaluating Large Language Models (LLMs) on Spanish and Portuguese tasks for the LATAM Leaderboard. This system uses Prefect for workflow orchestration, vLLM for efficient model serving, and lm-evaluation-harness for standardized evaluations.

## 🚀 Features

- **Multi-language Support**: Evaluates models on both Spanish (`latam_es`) and Portuguese (`latam_pr`) tasks
- **vLLM Integration**: High-performance model serving with GPU optimization
- **Prefect Orchestration**: Robust workflow management with monitoring and retry capabilities
- **Flexible Configuration**: YAML-based configuration system for easy model and task management
- **Batch Processing**: Support for running multiple models sequentially
- **Comprehensive Logging**: Detailed logging and result tracking
- **Leaderboard Integration**: Automatic result parsing and upload to the LATAM leaderboard

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

### Model Configuration

Create a YAML configuration file for each model you want to evaluate. See `configs/templates/test-model.yaml` for a template:

```yaml
# Model configuration
model:
  name: "your-model-name"

# Evaluation settings
evaluation:
  tasks_spanish: "latam_es"
  tasks_portuguese: "latam_pr"
  batch_size: "4"
  output_path: "/path/to/outputs"
  log_samples: true
  cache_requests: true
  trust_remote_code: true
  num_concurrent: 8
  limit: null  # Set to a number for testing with limited samples

# vLLM server configuration
vllm:
  host: "0.0.0.0"
  port: 8000
  tensor_parallel_size: 1
  max_model_len: 8192
  gpu_memory_utilization: 0.7
  enforce_eager: true
  limit_mm_per_prompt: '{"images": 0, "audios": 0}'
  hf_cache: "/path/to/huggingface/cache"
  startup_timeout: 900
  cuda_devices: "0"  # Specify GPU device(s)

# Optional: Weights & Biases integration
wandb:
  entity: "your-entity"
  project: "LATAM-leaderboard"

# Logging configuration
logging:
  log_dir: "logs"

# Virtual environment paths
venvs:
  lm_eval_spanish: "/path/to/lm-evaluation-harness"
  lm_eval_portuguese: "/path/to/portuguese-bench"
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

### Single Model Evaluation

```bash
# Run with default config
python eval.py

# Run with specific config
python eval.py --config configs/my-model.yaml

# Test vLLM server only (no evaluation)
python eval.py --test

# Register flows with Prefect for dashboard visibility
python eval.py --register

# Verbose logging
python eval.py --verbose
```

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

The benchmarking pipeline consists of the following steps:

1. **vLLM Server Startup**: Launches a vLLM server with the specified model
2. **API Testing**: Verifies the server is responding correctly
3. **Spanish Evaluation**: Runs Spanish tasks using lm-evaluation-harness
4. **Portuguese Evaluation**: Runs Portuguese tasks using the Portuguese benchmark suite
5. **Result Gathering**: Collects and processes evaluation results
6. **Server Cleanup**: Stops the vLLM server and cleans up resources

## 📁 Directory Structure

```
benchy/
├── configs/                 # Configuration files
│   ├── templates/          # Configuration templates
│   ├── single_card/        # Single GPU configurations
│   └── server/            # Server-specific configs
├── src/                   # Source code
│   ├── pipeline.py        # Main Prefect pipeline
│   ├── steps.py          # Individual pipeline steps
│   ├── logging_utils.py  # Logging utilities
│   └── leaderboard/      # Leaderboard integration
├── outputs/              # Evaluation results
├── logs/                # Log files
├── external/            # External dependencies
├── reference/           # Reference data
├── eval.py             # Main entry point
├── run_models.sh       # Batch evaluation script
└── download_models.sh  # Model download script
```

## 🔧 Advanced Configuration

### GPU Memory Optimization

For memory-constrained setups:

```yaml
vllm:
  gpu_memory_utilization: 0.6  # Reduce from default 0.9
  kv_cache_memory: 12934271795  # Explicit KV cache allocation
  tensor_parallel_size: 1       # Single GPU
```

### Multi-GPU Setup

```yaml
vllm:
  tensor_parallel_size: 2       # Use 2 GPUs
  cuda_devices: "0,1"          # Specify GPU devices
```

### Testing Configuration

For quick testing with limited samples:

```yaml
evaluation:
  limit: 10                    # Only 10 samples per task
  batch_size: "2"             # Smaller batch size
  num_concurrent: 4           # Fewer concurrent requests
```

## 📈 Monitoring and Logging

- **Prefect Dashboard**: Access at `http://localhost:4200` to monitor pipeline execution
- **Log Files**: Detailed logs stored in `logs/` directory
- **Weights & Biases**: Optional integration for experiment tracking
- **Output Files**: Results stored in `outputs/` directory

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
python eval.py --verbose --config configs/debug-config.yaml
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
