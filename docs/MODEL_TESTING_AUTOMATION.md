# Model Testing Automation

This script automates the process of testing models to determine their GPU requirements and compatibility with the benchy testing suite.

## Features

- **Smart Model Detection**: Checks if models already exist before downloading
- **Progress Visibility**: Shows download progress with heartbeats and progress indicators
- **Automatic Model Download**: Downloads models using `huggingface-cli` with timeout protection
- **GPU Testing**: Tests models with both single GPU and two GPU configurations
- **Suite Testing**: Runs limited benchmarking suite (10 samples) on working models
- **Config Generation**: Creates appropriate config files for working models
- **Cleanup**: Removes failed models from cache to save space
- **Comprehensive Logging**: Detailed logs compatible with `nohup` and file streaming
- **Results Summary**: JSON and human-readable summaries of all tests
- **Resume Support**: Can skip downloads and test only existing models
- **Process Management**: PID file for easy process tracking and stopping
- **Signal Handling**: Graceful shutdown with Ctrl+C or kill signals
- **Extended Timeout**: 90-minute timeout for large model downloads
- **Intelligent Disk Space Monitoring**: Model-specific size estimation and space checking
- **Emergency Recovery**: Graceful handling of disk space exhaustion
- **Progress Persistence**: Automatic progress saving for recovery

## Usage

### Basic Usage
```bash
# Test all models in next_models.txt
python test_model_automation.py --models-file next_models.txt

# Test with nohup (recommended for long runs)
nohup python test_model_automation.py --models-file next_models.txt > testing.log 2>&1 &
```

### Advanced Usage
```bash
# Start from model 10 (skip first 10 models)
python test_model_automation.py --models-file next_models.txt --start-from 10

# Test only first 5 models
python test_model_automation.py --models-file next_models.txt --max-models 5

# Test specific range (models 5-15)
python test_model_automation.py --models-file next_models.txt --start-from 5 --max-models 10

# Skip downloads and test only existing models
python test_model_automation.py --models-file next_models.txt --skip-downloads
```

### Process Management
```bash
# Start with nohup (recommended)
nohup python test_model_automation.py --models-file next_models.txt > testing.log 2>&1 &

# Stop the process gracefully
./stop_testing.sh

# Or manually stop using PID
kill $(cat logs/model_testing.pid)

# Or force stop
pkill -f test_model_automation
```

### Disk Space Management
```bash
# Check disk space and find large models
python check_disk_space.py

# Interactive cleanup of large models
python check_disk_space.py --cleanup

# Check specific path
python check_disk_space.py --path /path/to/cache --min-size 2.0
```

### Recovery from Interruption
```bash
# Check what happened in previous run
python test_model_automation.py --models-file next_models.txt --recover

# Resume from specific model (after freeing space)
python test_model_automation.py --models-file next_models.txt --start-from 10
```

## Output Structure

### Generated Files
- **Configs**: `configs/testing/` - Working model configurations
- **Logs**: `logs/model_testing_YYYYMMDD_HHMMSS.log` - Detailed execution logs
- **Results**: `logs/model_testing_results_YYYYMMDD_HHMMSS.json` - Machine-readable results
- **Progress**: `logs/model_testing_progress.json` - Current progress for recovery
- **Emergency**: `logs/model_testing_emergency_results.json` - Results from interrupted runs
- **PID**: `logs/model_testing.pid` - Process ID for management

### Config Naming
- Single GPU: `{model_name}_single.yaml`
- Two GPU: `{model_name}_two.yaml`

### Results Categories
1. **Single GPU Passed**: Models that work with 1 GPU
2. **Two GPU Passed**: Models that work with 2 GPUs
3. **Failed**: Models that don't work with either configuration

## Testing Process

For each model, the script:

1. **Downloads** the model using `huggingface-cli`
2. **Tests Single GPU**: Tries to start vLLM server with 1 GPU
3. **Tests Two GPU**: If single GPU fails, tries with 2 GPUs
4. **Suite Testing**: If model works, runs limited benchmarking suite
5. **Config Generation**: Creates appropriate config file for working models
6. **Cleanup**: Removes failed models from cache

## Example Output

### With Download Progress
```
🧪 Testing model: HuggingFaceH4/zephyr-7b-beta
📥 Downloading model: HuggingFaceH4/zephyr-7b-beta
   This may take several minutes for large models...
   📊 Downloading model.safetensors: 100%|██████████| 2.1G/2.1G [02:15<00:00, 15.6MB/s]
   ⏳ Still downloading... (2m 30s elapsed)
✅ Successfully downloaded: HuggingFaceH4/zephyr-7b-beta
🧪 Testing HuggingFaceH4/zephyr-7b-beta with single GPU config
✅ single GPU test PASSED for HuggingFaceH4/zephyr-7b-beta
🎯 Testing HuggingFaceH4/zephyr-7b-beta with full suite (limited)
✅ Full suite test PASSED for HuggingFaceH4/zephyr-7b-beta
📝 Generated config: configs/testing/HuggingFaceH4_zephyr-7b-beta_single.yaml
🎉 HuggingFaceH4/zephyr-7b-beta WORKED with single GPU ✅
```

### With Existing Model
```
🧪 Testing model: openai/gpt-oss-20b
✅ Model already exists: openai/gpt-oss-20b
🧪 Testing openai/gpt-oss-20b with single GPU config
❌ single GPU test FAILED for openai/gpt-oss-20b
🧪 Testing openai/gpt-oss-20b with two GPU config
✅ two GPU test PASSED for openai/gpt-oss-20b
🎯 Testing openai/gpt-oss-20b with full suite (limited)
✅ Full suite test PASSED for openai/gpt-oss-20b
📝 Generated config: configs/testing/openai_gpt-oss-20b_two.yaml
🎉 openai/gpt-oss-20b WORKED with two GPU ✅
```

## Exit Codes

- **0**: All models passed testing
- **1**: Some models failed (but some may have passed)

## Requirements

- Python 3.7+
- `huggingface-cli` installed
- Benchy virtual environment activated
- Sufficient disk space for model downloads
- GPU access for testing

## Notes

- The script automatically activates the benchy virtual environment
- Failed models are cleaned up to save disk space
- Suite test failures are reported as warnings, not failures
- All operations have timeouts to prevent hanging
- Logs are designed to work with `nohup` and file streaming
- **Disk space is monitored intelligently** - estimates model size before downloading
- **Progress is saved automatically** for recovery from interruptions
- **Emergency summaries** are generated if the process is interrupted
- **Minimum 5GB free space** is required for downloads
