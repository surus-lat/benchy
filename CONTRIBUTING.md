# Contributing to Benchy - LATAM Leaderboard Benchmarking Suite

Thank you for your interest in contributing to Benchy! This document will guide you through the process of contributing new tasks, benchmarks, and improvements to our LATAM-focused LLM benchmarking suite.

## 🎯 Project Goals

Benchy is designed to be **task-oriented at its core**. Our benchmarking suite focuses on specific application areas for LLMs that are relevant to LATAM communities:

- **Translation** (Spanish ↔ Portuguese, Indigenous languages)
- **Transcription** (Speech-to-text for LATAM accents and dialects)
- **Structured Data Extraction** (Legal documents, government forms)
- **Summarization** (News, academic papers, legal documents)
- **Question Answering** (Cultural and regional knowledge)
- **And more...**

We welcome contributions that bring **best-in-class tools and benchmarks** for these application areas, especially those incorporating **LATAM language material**.

## 📋 Types of Contributions

### 1. **New Tasks & Benchmarks** (Most Welcome!)
- New evaluation tasks with LATAM language datasets
- Benchmark strategies for specific application areas
- Dataset contributions (with proper licensing)
- Metric definitions for new evaluation types

### 2. **Code Contributions**
- New task modules and processors
- Integration with evaluation frameworks
- Pipeline improvements and optimizations
- Bug fixes and performance enhancements

### 3. **Documentation**
- Task documentation and examples
- Tutorial improvements
- Translation of documentation to Spanish/Portuguese

## 🏗️ Technical Architecture

### Pipeline Architecture (Prefect-based)

Benchy uses **Prefect** for workflow orchestration, where each task is a **Prefect step**:

```python
@task()
def run_spanish_evaluation(
    model_name: str,
    output_path: str,
    server_info: Dict[str, Any],
    task_config: Dict[str, Any],
    limit: Optional[int] = None
) -> Dict[str, Any]:
    # Task-specific evaluation logic
    pass
```

#### Task Configuration System

Each task is configured through YAML files in `configs/tasks/`:

```yaml
# configs/tasks/my_new_task.yaml
name: "my_new_task"
description: "Description of what this task evaluates"

# Task-specific parameters
tasks:
  - "subtask_a"
  - "subtask_b"

# Default evaluation parameters
defaults:
  batch_size: 20
  log_samples: false
  temperature: 0.0
  max_tokens: 512
  timeout: 120
  max_retries: 3

# Output configuration
output:
  subdirectory: "my_new_task"  # Will be appended to main output path
```

#### Adding a New Task to the Pipeline

1. **Implement task processor** in `src/tasks/my_task.py`
2. **Create task config** in `configs/tasks/my_task.yaml`
3. **Add to pipeline** in `src/pipeline.py`
4. **Update task registry** in `src/leaderboard/functions/parse_model_results.py`

### Benchy Task Structure

Each Benchy task consists of:

1. **Dataset loader**: Downloads/preprocesses data into JSONL
2. **Task class**: Implements `load()`, `get_samples()`, `get_prompt()`, and metrics
3. **Run wrapper**: Prefect task that wires the generic engine

### Other Evaluation Tools

We welcome integration with other evaluation frameworks:

- **DeepEval**
- **LangChain Evaluators**
- **Custom evaluation scripts**

**Requirements for new tools:**
- Must produce results in a format compatible with our aggregation system
- Must follow the same modular structure as existing tasks

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/benchy.git
cd benchy
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/my-new-task
# or
git checkout -b add/translation-benchmark
```

## 📝 Contribution Workflow

### 1. **Planning Your Contribution**

Before coding, please:

- **Open an issue** describing your proposed task/contribution
- **Discuss the approach** with maintainers
- **Ensure relevance** and language material inclusion
- **Check for existing similar tasks** to avoid duplication

### 2. **Development Process**

#### For New Tasks:

1. **Create task configuration**:
   ```bash
   # Create new task config
   cp configs/tasks/spanish.yaml configs/tasks/my_new_task.yaml
   # Edit the configuration
   ```

2. **Implement task processor**:
   ```python
   # src/tasks/my_new_task.py
   @task()
   def run_my_new_task_evaluation(
       model_name: str,
       output_path: str,
       server_info: Dict[str, Any],
       task_config: Dict[str, Any],
       limit: Optional[int] = None,
       cuda_devices: Optional[str] = None
   ) -> Dict[str, Any]:
       """Run my new task evaluation."""
       # Implementation here
       pass
   ```

3. **Add to pipeline**:
   ```python
   # src/pipeline.py - Add to benchmark_pipeline function
   if "my_new_task" in tasks:
       logger.info("Running my new task evaluation...")
       my_task_config = config_manager.get_task_config("my_new_task")
       my_task_results = run_my_new_task_evaluation(
           model_name=model_name,
           output_path=model_output_path,
           server_info=server_info,
           api_test_result=api_test_result,
           task_config=my_task_config,
           limit=limit,
           cuda_devices=cuda_devices
       )
       task_results["my_new_task"] = my_task_results
   ```

4. **Add results processor**:
   ```python
   # src/leaderboard/functions/parse_model_results.py
   def process_my_new_task_results(model_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
       """Process my new task results for a model."""
       # Implementation here
       pass
   
   # Add to get_available_task_processors()
   def get_available_task_processors() -> Dict[str, callable]:
       return {
           "spanish": process_spanish_results,
           "portuguese": process_portuguese_results,
           "my_new_task": process_my_new_task_results,  # Add this line
       }
   ```

#### For New Evaluation Tools:

1. **Create module structure**:
   ```
   src/tools/my_tool/
   ├── __init__.py
   ├── evaluator.py
   ├── metrics.py
   └── latam_aggregator.py  # Required: LATAM-compatible output analysis
   ```

2. **Implement LATAM aggregator**:
   ```python
   # src/tools/my_tool/latam_aggregator.py
   def aggregate_latam_results(results: Dict[str, Any]) -> Dict[str, Any]:
       """Convert tool results to LATAM leaderboard format."""
       return {
           "task_scores": {...},
           "category_scores": {...},
           "overall_score": 0.85,
           "model_name": "model_name"
       }
   ```

### 3. **Testing Your Contribution**

```bash
# Test with limited samples
python eval.py --config configs/templates/test-model_new.yaml --limit 10

# Test results processing
python ./src/leaderboard/process_all.py

# Run linting
flake8 src/
black src/
```

### 4. **Documentation**

Update relevant documentation:

- **README.md**: Add your task to examples if significant
- **Task-specific docs**: Create `docs/tasks/my_task.md` if complex
- **API documentation**: Update docstrings and type hints

## 📋 Commit Guidelines

### Commit Message Format

Use descriptive commit messages with tags:

```
[ADD] New translation task for Spanish-Portuguese evaluation
[MOD] Update Portuguese task to support new dataset format
[REF] Refactor task processing to improve modularity
[FIX] Resolve issue with nested result directory parsing
[DOC] Add documentation for new evaluation metrics
[TEST] Add unit tests for translation task processor
```

### Tag Meanings

- **`[ADD]`**: New features, tasks, or functionality
- **`[MOD]`**: Modifications to existing features
- **`[REF]`**: Code refactoring without changing functionality
- **`[FIX]`**: Bug fixes
- **`[DOC]`**: Documentation updates
- **`[TEST]`**: Test additions or modifications
- **`[CI]`**: Continuous integration changes
- **`[DEPS]`**: Dependency updates

### Examples

```bash
git commit -m "[ADD] Spanish legal document extraction task"
git commit -m "[MOD] Update Portuguese task config with new batch size defaults"
git commit -m "[REF] Extract common metric calculation logic"
git commit -m "[FIX] Handle missing results files gracefully"
git commit -m "[DOC] Add contributing guidelines for new tasks"
```

## 🧪 Testing Requirements

For a project focused on evals, we are pretty bad at tests. Please contribute!

### For New Tasks

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test with sample data
3. **End-to-End Tests**: Test full pipeline execution

```python
# tests/test_my_new_task.py
import pytest
from src.tasks.my_new_task import run_my_new_task_evaluation

def test_my_new_task_evaluation():
    """Test my new task evaluation function."""
    # Test implementation
    pass
```

### For New Tools

1. **Tool-specific tests**: Test evaluation logic
2. **LATAM aggregator tests**: Test result conversion
3. **Integration tests**: Test with pipeline

## 📊 Dataset Requirements

### For New Datasets

- **Licensing**: Must be compatible with open source
- **LATAM Focus**: Should include Spanish, Portuguese, or indigenous languages
- **Quality**: High-quality, representative samples
- **Documentation**: Clear description of data sources and collection methods

### Dataset Format

```jsonl
{"input": "Translate to Spanish: Hello world", "output": "Hola mundo", "metadata": {"source": "test"}}
{"input": "What is the capital of Brazil?", "output": "Brasília", "metadata": {"source": "geography"}}
```

## 🔍 Code Review Process

### Before Submitting

1. **Self-review**: Check your code thoroughly
2. **Test locally**: Ensure all tests pass
3. **Update documentation**: Keep docs current
4. **Check formatting**: Run `black` and `flake8`

### Pull Request Guidelines

1. **Clear description**: Explain what your PR does
2. **Reference issues**: Link to related issues
3. **Screenshots**: For UI changes
4. **Test results**: Show that tests pass
5. **Breaking changes**: Clearly document any breaking changes

### Review Criteria

- **Functionality**: Does it work as intended?
- **Code quality**: Is the code clean and maintainable?
- **LATAM relevance**: Does it serve LATAM communities?
- **Documentation**: Is it well-documented?
- **Testing**: Are there adequate tests?

## 🌍 LATAM Community Guidelines

### Language Considerations

- **Primary languages**: English, Spanish, Portuguese
- **Indigenous languages**: Welcome but not required
- **Regional variants**: Consider different LATAM dialects
- **Cultural context**: Include region-specific knowledge

## 🆘 Getting Help

### Resources

- **Issues**: Use GitHub issues for questions and bug reports
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check existing docs first
- **Examples**: Look at existing task implementations

### Contact

- **Maintainers**: @maintainer-handles
- **Community**: Join our Discord/Slack
- **Email**: contact@surus.lat

## 📄 License

By contributing to Benchy, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## 🙏 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributions highlighted
- **Documentation**: Contributors credited in relevant sections
- **Community**: Public recognition for significant contributions

---

Thank you for contributing to Benchy and helping build better LLM evaluation tools for LATAM communities! 🚀
