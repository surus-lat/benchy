# LLM Structured Data Extraction Benchmark

A benchmarking system to evaluate Large Language Models on structured data extraction tasks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and preprocess the dataset:
```bash
python download_dataset.py
```
   **Note:** The dataset preprocessing now includes schema sanitization for vLLM compatibility.
   If you previously downloaded the dataset, re-run this script to get the optimized version.

3. Ensure vLLM server is running on port 20502

4. Configure settings in `config.yaml` if needed

## Usage

### Basic evaluation
```bash
python structured_benchmark.py <model-name>
```

### Evaluate with limited samples (for testing)
```bash
python structured_benchmark.py <model-name> --limit 10
```

### Log per-sample outputs for debugging
```bash
python structured_benchmark.py <model-name> --limit 10 --log-samples
```

### Examples
```bash
# Quick test with 10 samples
python structured_benchmark.py meta-llama/Llama-3-8B-Instruct --limit 10

# Full evaluation with sample logging
python structured_benchmark.py meta-llama/Llama-3-8B-Instruct --log-samples

# Debug mode (shows all HTTP requests and detailed errors)
python structured_benchmark.py meta-llama/Llama-3-8B-Instruct --limit 5 --log-level DEBUG
```

## Logging

By default, logging is clean and focused:
- ✅ Connection test results
- ✅ Batch progress
- ✅ Aggregate metrics
- ✅ Warnings for batch errors (if any)
- ❌ No HTTP request spam

Use `--log-level DEBUG` to see all details including individual HTTP requests.

## Results

Results are saved to the `results/` directory:
- `*_metrics.json`: Aggregate metrics for the run
- `*_samples.json`: Per-sample results (if `--log-samples` is used)

## Metrics

The benchmark calculates the following metrics:

- **Validity Rate**: Percentage of outputs that conform to the target schema
- **Exact Match**: Percentage of outputs that exactly match the expected output
- **F1 Score**: Harmonic mean of precision and recall at field level
- **Precision**: Ratio of correct fields to all predicted fields
- **Recall**: Ratio of correct fields to all expected fields
- **Type Accuracy**: Percentage of fields with correct data types
- **Error Rate**: Percentage of samples that failed to generate

## Error Handling

The benchmark is robust against common failures:

### Server Connection Issues
- **Automatic retries**: 3 attempts with exponential backoff
- **Timeout**: 30 seconds per attempt
- **Graceful failure**: Clear error messages if server is down

### Model Generation Errors
- **Per-sample retries**: 3 attempts with exponential backoff
- **JSON parse errors**: Automatically handled (marked as errors, continues)
- **Context length errors**: Filtered during preprocessing (>20K chars)
- **Batch error reporting**: Warnings logged if batch has failures

### What Gets Logged

**Normal mode** (clean output):
- Connection test: success/failure with model verification
- Batch progress: "Processing batch X/Y (N samples)..."
- Batch warnings: Only if errors occur in batch
- Final summary: Aggregate metrics

**Debug mode** (`--log-level DEBUG`):
- All HTTP requests
- Individual sample failures  
- JSON parse errors
- Retry attempts

## Configuration

Edit `config.yaml` to customize:
- Model server URL and parameters
- Dataset settings
- Prompt templates
- Metrics calculation settings
- Output directories
- Performance (batch size, retries, timeouts)

