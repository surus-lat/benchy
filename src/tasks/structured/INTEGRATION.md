# Structured Extraction Benchmark Integration

This document describes how the structured extraction benchmark is integrated into the benchy pipeline.

## Overview

The structured extraction benchmark evaluates LLM performance on extracting structured data from text according to JSON schemas. It has been integrated as a standard task in the benchy benchmarking pipeline.

## Integration Changes

### 1. Task Configuration
- **Location**: `configs/tasks/structured_extraction.yaml`
- **Purpose**: Defines task parameters, prompts, metrics configuration
- **Key Settings**:
  - Batch size: 20 concurrent requests
  - Temperature: 0.0 (deterministic)
  - Max tokens: 2048
  - Timeout: 120 seconds

### 2. Task Runner
- **Location**: `src/tasks/structured.py`
- **Function**: `run_structured_extraction()`
- **Integration**: Uses Prefect @task decorator, follows same pattern as other tasks
- **Features**:
  - Auto-downloads dataset if not present
  - Configures benchmark using pipeline-provided vLLM server
  - Uses shared output path structure
  - Integrates with pipeline logging

### 3. Pipeline Integration
- **Location**: `src/pipeline.py`
- **Changes**:
  - Import: Added `from .tasks.structured import run_structured_extraction`
  - Task Execution: Added conditional block for "structured_extraction" task
  - Configuration: Uses `config_manager.get_task_config()` for task settings

### 4. Dependencies
- **Location**: `pyproject.toml`
- **Added Dependencies**:
  - `openai>=1.0.0` (vLLM client)
  - `datasets>=2.14.0` (HuggingFace datasets)
  - `jsonschema>=4.19.0` (Schema validation)
  - `python-Levenshtein>=0.21.0` (String similarity)
  - `scipy>=1.9.0` (Statistical analysis)
  - `numpy>=1.24.0` (Numerical operations)
  - `rapidfuzz>=3.0.0` (Fuzzy matching)
  - `tqdm>=4.66.0` (Progress bars)

### 5. Setup Script
- **Location**: `setup.sh`
- **Addition**: Auto-downloads paraloq dataset during setup
- **Path**: `src/tasks/structured/.data/paraloq_data.jsonl`

### 6. Task Groups
- **Location**: `configs/config.yaml`
- **Addition**: Created `structured_only` task group
- **Usage**: Can be referenced in model configs as a shortcut

### 7. Import Path Fixes
Fixed all relative imports in structured extraction modules:
- `llm/__init__.py`: Changed to use `.interface`
- `benchmark_runner.py`: Changed to use `.llm`, `.metrics`, `.tasks`
- `structured_benchmark.py`: Changed to use `.benchmark_runner`
- `utils/__init__.py`: Changed to use `.schema_utils`, `.dataset_download`
- `utils/dataset_download.py`: Changed to use `..utils`, `..metrics`

## Usage

### Running the Task

1. **As standalone task in model config**:
```yaml
tasks:
  - "structured_extraction"
```

2. **As part of task group**:
```yaml
tasks:
  - "structured_only"
```

3. **With custom parameters**:
```yaml
task_defaults:
  batch_size: 30
  log_samples: true
  temperature: 0.1

tasks:
  - "structured_extraction"
```

### Testing with test-model_new.yaml

The integration is ready to be tested with your test config:
```yaml
model:
  name: "HuggingFaceH4/zephyr-7b-beta"
vllm:
  provider_config: vllm_single_card
  overrides:
    kv_cache_memory: 0
    gpu_memory_utilization: 0.95
    port: 20502

task_defaults:
  log_samples: true
  cuda_devices: "2"

tasks:
  - "structured_extraction"
```

## Output Structure

Results follow the standard benchy output structure:
```
{output_path}/{run_id}/{model_name}/structured_extraction/
├── {model_name}_{timestamp}_metrics.json    # Aggregate metrics
├── {model_name}_{timestamp}_report.txt      # Human-readable report
└── {model_name}_{timestamp}_samples.json    # Per-sample results (if log_samples=true)
```

## Configuration Flow

1. **Model Config** (`configs/templates/test-model_new.yaml`)
   - Specifies model and vLLM settings
   - Lists tasks to run
   - Provides task_defaults overrides

2. **Task Config** (`configs/tasks/structured_extraction.yaml`)
   - Defines task-specific defaults
   - Contains prompts and metrics config
   - Specifies dataset information

3. **Pipeline Execution** (`src/pipeline.py`)
   - Starts vLLM server with model
   - Loads and merges configurations
   - Executes structured extraction task
   - Saves results to output directory

4. **Task Runner** (`src/tasks/structured.py`)
   - Receives server_info from pipeline
   - Downloads dataset if needed
   - Builds benchmark config
   - Runs evaluation
   - Saves results

## Metrics

The benchmark computes:
- **Extraction Quality Score (EQS)**: Weighted combination of schema validity, field F1, and inverted hallucination
- **Schema Validity Rate**: Percentage of valid JSON outputs
- **Exact Match Rate**: Percentage of perfect extractions
- **Field F1 (Partial)**: Partial credit for field-level matches
- **Hallucination Rate**: Percentage of hallucinated fields
- **Complexity Analysis**: Performance breakdown by schema complexity

## Dataset

- **Source**: HuggingFace `paraloq/json_data_extraction`
- **Auto-download**: Yes, on first run
- **Cache**: `src/tasks/structured/cache/`
- **Processed**: `src/tasks/structured/.data/paraloq_data.jsonl`
- **Size**: ~1000 samples after filtering

## Notes

- Uses async processing for better throughput
- Supports checkpointing for resumable runs
- All dependencies integrated into main benchy venv (no separate venv)
- Compatible with existing pipeline logging and output structure

