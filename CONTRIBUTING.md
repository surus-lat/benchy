# Contributing to Benchy

Thank you for your interest in contributing to Benchy! This document provides guidelines for extending and modifying the codebase.

## 🏗️ Architecture Overview

Benchy is built with ZenML and follows a modular architecture:

```
benchy/
├── src/
│   ├── steps.py           # ZenML steps (evaluation & upload)
│   ├── pipeline.py        # Single model pipeline
│   └── batch_pipeline.py  # Multi-model pipeline
├── configs/               # Model configuration files
├── main.py               # Single model entry point
├── run_batch.py          # Simple batch runner
├── batch_runner.py       # Advanced batch runner
└── run_models.sh         # Shell script batch runner
```

## 🔧 Adding New Features

### Adding New lm-evaluation-harness Parameters

To add support for new lm-eval parameters (like `--limit`), follow this pattern:

1. **Update the ZenML step** (`src/steps.py`):
   ```python
   @step
   def run_lm_evaluation(
       # ... existing parameters ...
       new_param: str = None,  # Add your parameter
       # ... rest of parameters ...
   ):
       # Add to docstring
       # Add to cmd_parts if not None
       if new_param is not None:
           cmd_parts.extend(["--new-param", str(new_param)])
   ```

2. **Update the pipeline** (`src/pipeline.py`):
   ```python
   @pipeline
   def benchmark_pipeline(
       # ... existing parameters ...
       new_param: str = None,  # Add parameter
       # ... rest of parameters ...
   ):
       # Pass to run_lm_evaluation
       eval_results = run_lm_evaluation(
           # ... existing args ...
           new_param=new_param,
           # ... rest of args ...
       )
   ```

3. **Update main.py**:
   ```python
   result = benchmark_pipeline(
       # ... existing args ...
       new_param=eval_config.get('new_param'),
       # ... rest of args ...
   )
   ```

4. **Update configuration templates**:
   ```yaml
   evaluation:
     # ... existing options ...
     new_param: "value"  # Add with comment explaining usage
   ```

5. **Update batch pipeline** (`src/batch_pipeline.py`) if applicable.

### Example: The `--limit` Parameter

Here's how the `--limit` parameter was added:

**1. ZenML Step (`src/steps.py`):**
```python
@step
def run_lm_evaluation(
    # ... other params ...
    limit: int = None,  # ✅ Added parameter
    # ... rest ...
):
    # ✅ Added to docstring
    """
    Args:
        # ... other args ...
        limit: Limit number of examples per task (useful for testing)
    """
    
    # ✅ Added to command building
    if limit is not None:
        cmd_parts.extend(["--limit", str(limit)])
```

**2. Pipeline (`src/pipeline.py`):**
```python
@pipeline
def benchmark_pipeline(
    # ... other params ...
    limit: int = None,  # ✅ Added parameter
    # ... rest ...
):
    eval_results = run_lm_evaluation(
        # ... other args ...
        limit=limit,  # ✅ Pass parameter
        # ... rest ...
    )
```

**3. Main (`main.py`):**
```python
result = benchmark_pipeline(
    # ... other args ...
    limit=eval_config.get('limit'),  # ✅ Get from config
    # ... rest ...
)
```

**4. Configuration:**
```yaml
evaluation:
  limit: 10  # ✅ Added option with comment
```

## 📝 Configuration Management

### Single Model Configs

Create config files in `configs/` directory:

```yaml
# configs/my-model.yaml
model:
  name: "organization/model-name"
  dtype: "float16"
  max_length: 8192

evaluation:
  tasks: "task1,task2"
  device: "cuda"
  batch_size: "auto:4"
  limit: 10  # For testing
  # ... other options
```

### Batch Configs

For multiple models, use `batch_config.yaml`:

```yaml
models:
  - name: "model1"
    # ... model-specific options
  - name: "model2"
    # ... model-specific options

common:
  # ... shared configuration
```

## 🚀 Running Different Scenarios

### Testing Changes

1. **Quick test with limit**:
   ```bash
   # Use example-with-limit.yaml
   BENCHY_CONFIG=configs/example-with-limit.yaml python main.py
   ```

2. **Full evaluation**:
   ```bash
   # Remove or comment out 'limit' in config
   python main.py
   ```

### Batch Evaluations

1. **Simple approach** (recommended for 5 models):
   ```bash
   ./run_models.sh
   ```

2. **Advanced approach** (for complex workflows):
   ```bash
   python batch_runner.py
   ```

## 🧪 Testing Your Changes

### Basic Testing

1. **Test with limit flag**:
   ```bash
   BENCHY_CONFIG=configs/example-with-limit.yaml python main.py
   ```

2. **Check command generation**:
   Look for log output like:
   ```
   [run_lm_evaluation] Executing command: lm_eval --model hf --model_args ... --limit 10
   ```

### Validation Checklist

- [ ] Parameter appears in ZenML step signature
- [ ] Parameter is documented in docstring
- [ ] Parameter is passed through pipeline
- [ ] Parameter is read from configuration
- [ ] Command line includes the new flag when set
- [ ] Configuration examples are updated
- [ ] Batch pipeline supports the parameter (if applicable)

## 🐛 Common Issues

### Missing Parameters

**Problem**: New parameter not appearing in command
**Solution**: Check the parameter flow: config → main.py → pipeline → step

### Configuration Not Loading

**Problem**: Parameter in config but not used
**Solution**: Ensure `eval_config.get('param_name')` is used in main.py

### Type Mismatches

**Problem**: Parameter value is wrong type
**Solution**: Cast parameters appropriately (e.g., `str(limit)` for command line)

## 🎯 Best Practices

1. **Always add defaults**: Use `param: Type = None` for optional parameters
2. **Document everything**: Update docstrings when adding parameters
3. **Test incrementally**: Use `--limit 10` for quick testing
4. **Follow the pattern**: Look at how `limit` was implemented as a reference
5. **Update all runners**: Don't forget batch configurations
6. **Use meaningful names**: Parameter names should match lm-eval flags

## 📚 Files to Modify for New Parameters

| File | Purpose | Required Changes |
|------|---------|-----------------|
| `src/steps.py` | ZenML step | Add parameter, update docstring, add to cmd_parts |
| `src/pipeline.py` | Single model pipeline | Add parameter, pass to step |
| `main.py` | Entry point | Read from config, pass to pipeline |
| `configs/config-template.yaml` | Config template | Add option with comment |
| `src/batch_pipeline.py` | Batch pipeline | Add parameter support (optional) |
| `batch_config.yaml` | Batch config | Add option to models (optional) |

## 🤝 Contributing Guidelines

1. **Follow the existing patterns** shown in this document
2. **Test your changes** with both single and batch modes
3. **Update documentation** when adding new features
4. **Use descriptive commit messages**
5. **Keep changes minimal** and focused

## 📞 Getting Help

- Check existing parameter implementations (like `limit`) as examples
- Review ZenML documentation for step and pipeline patterns
- Test with small configurations first (`limit: 10`)

Happy contributing! 🚀
