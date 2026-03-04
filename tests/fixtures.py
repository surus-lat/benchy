"""Shared test fixtures and utilities for Benchy tests."""

from pathlib import Path
from typing import Any, Dict, List
import json
import pytest


@pytest.fixture
def sample_classification_dataset(tmp_path: Path) -> Path:
    """Create a sample classification JSONL dataset.
    
    Returns:
        Path to the created dataset file
    """
    dataset_file = tmp_path / "classification.jsonl"
    samples = [
        {"id": "1", "text": "This is positive", "label": 1},
        {"id": "2", "text": "This is negative", "label": 0},
        {"id": "3", "text": "Another positive", "label": 1},
    ]
    
    with open(dataset_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    return dataset_file


@pytest.fixture
def sample_structured_dataset(tmp_path: Path) -> Path:
    """Create a sample structured extraction JSONL dataset.
    
    Returns:
        Path to the created dataset file
    """
    dataset_file = tmp_path / "structured.jsonl"
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    }
    
    samples = [
        {
            "id": "1",
            "text": "John Doe is 30 years old. Email: john@example.com",
            "schema": schema,
            "expected": {"name": "John Doe", "age": 30, "email": "john@example.com"}
        },
        {
            "id": "2",
            "text": "Jane Smith, age 25. Contact: jane@example.com",
            "schema": schema,
            "expected": {"name": "Jane Smith", "age": 25, "email": "jane@example.com"}
        },
    ]
    
    with open(dataset_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    return dataset_file


@pytest.fixture
def sample_freeform_dataset(tmp_path: Path) -> Path:
    """Create a sample freeform JSONL dataset.
    
    Returns:
        Path to the created dataset file
    """
    dataset_file = tmp_path / "freeform.jsonl"
    samples = [
        {"id": "1", "text": "What is AI?", "expected": "AI is artificial intelligence"},
        {"id": "2", "text": "What is ML?", "expected": "ML is machine learning"},
        {"id": "3", "text": "What is NLP?", "expected": "NLP is natural language processing"},
    ]
    
    with open(dataset_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    return dataset_file


@pytest.fixture
def sample_schema(tmp_path: Path) -> Path:
    """Create a sample JSON schema file.
    
    Returns:
        Path to the created schema file
    """
    schema_file = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "total": {"type": "number"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "price": {"type": "number"}
                    }
                }
            }
        },
        "required": ["invoice_number", "date", "total"]
    }
    
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    return schema_file


@pytest.fixture
def mock_huggingface_download(monkeypatch):
    """Mock HuggingFace dataset download.
    
    Returns a function that can be called to set up the mock with custom data.
    """
    def setup_mock(samples: List[Dict[str, Any]]):
        def fake_download(*args, **kwargs):
            return samples
        
        monkeypatch.setattr(
            "src.tasks.common.utils.dataset_utils.download_huggingface_dataset",
            fake_download
        )
    
    return setup_mock


@pytest.fixture
def fake_cli_args():
    """Create fake argparse.Namespace for testing.
    
    Returns a function that creates Namespace with given kwargs.
    """
    from argparse import Namespace
    
    def create_args(**kwargs):
        # Default values matching benchy_cli_eval.py add_eval_arguments()
        defaults = {
            # Core config
            'config_ref': None,
            'config': None,
            'verbose': False,
            'test': False,
            'register': False,
            'prefect_url': 'http://localhost:4200/api',
            
            # Task selection
            'tasks': [],
            'tasks_file': None,
            'task_group': None,
            'limit': None,
            'batch_size': None,
            'log_samples': False,
            'no_log_samples': False,
            'run_id': None,
            
            # Model config
            'model_name': 'test-model',
            'model_path': None,
            'provider': None,
            'base_url': None,
            'api_key_env': None,
            'api_key': None,
            'timeout': None,
            'max_retries': None,
            'max_concurrent': None,
            'temperature': None,
            'max_tokens': None,
            'max_tokens_param_name': None,
            'api_endpoint': None,
            'use_structured_outputs': None,
            'probe_mode': 'skip',
            'image_max_edge': None,
            'organization': None,
            'url': None,
            'vllm_config': None,
            'compatibility': None,
            'exit_policy': 'relaxed',
            'output_path': None,
            'dataset': None,
            
            # Ad-hoc task / dataset config
            'task_type': None,
            'dataset_name': None,
            'dataset_source': 'auto',
            'dataset_split': 'test',
            'dataset_input_field': None,
            'dataset_output_field': None,
            'dataset_id_field': None,
            'dataset_label_field': None,
            'dataset_labels': None,
            'dataset_choices_field': None,
            'dataset_schema_field': None,
            'dataset_schema_path': None,
            'dataset_schema_json': None,
            'multimodal_input': False,
            'multimodal_image_field': 'image_path',
            'system_prompt': None,
            'user_prompt_template': None,
            'save_config': None,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)
    
    return create_args
