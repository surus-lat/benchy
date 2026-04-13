"""Unit tests for src/benchmark_compiler.py.

Tests pure compiler logic — no API calls, no eval pipeline invoked.
"""

from __future__ import annotations

import json
import textwrap
import tempfile
from pathlib import Path

import pytest
import yaml

from src.benchmark_compiler import (
    _resolve_task_type,
    _build_json_schema,
    _build_dataset_config,
    _build_target_config,
    compile_benchmark,
    validate_benchmark_yaml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_spec(tmp_path: Path, spec: dict) -> str:
    p = tmp_path / "benchmark.yaml"
    p.write_text(yaml.dump({"benchmark": spec}))
    return str(p)


def _minimal_spec(**overrides) -> dict:
    base = {
        "name": "test-bench",
        "description": "test",
        "task": {
            "type": "extraction",
            "input": {"type": "text", "description": "some input"},
            "output": {"type": "structured", "fields": [{"name": "foo", "type": "string", "required": True}]},
        },
        "scoring": {"type": "per_field"},
        "data": {"source": "local", "path": ".data/test/train.jsonl"},
        "target": {"type": "model", "provider": "together", "model": "google/gemma-4-31B-it"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Task type mapping
# ---------------------------------------------------------------------------

class TestResolveTaskType:
    def _make(self, task_type, output_type, input_type="text"):
        return {
            "type": task_type,
            "input": {"type": input_type},
            "output": {"type": output_type},
        }

    def test_extraction_structured(self):
        t, mm = _resolve_task_type(self._make("extraction", "structured"))
        assert t == "structured"
        assert mm is False

    def test_classification_label(self):
        t, mm = _resolve_task_type(self._make("classification", "label"))
        assert t == "classification"
        assert mm is False

    def test_qa_text(self):
        t, _ = _resolve_task_type(self._make("qa", "text"))
        assert t == "freeform"

    def test_translation_text(self):
        t, _ = _resolve_task_type(self._make("translation", "text"))
        assert t == "freeform"

    def test_freeform_text(self):
        t, _ = _resolve_task_type(self._make("freeform", "text"))
        assert t == "freeform"

    def test_summarization_text(self):
        t, _ = _resolve_task_type(self._make("summarization", "text"))
        assert t == "freeform"

    def test_image_input_sets_multimodal(self):
        t, mm = _resolve_task_type(self._make("extraction", "structured", input_type="image"))
        assert t == "structured"
        assert mm is True

    def test_pdf_input_sets_multimodal(self):
        _, mm = _resolve_task_type(self._make("extraction", "structured", input_type="pdf"))
        assert mm is True

    def test_unknown_type_falls_back_to_output_type(self):
        t, _ = _resolve_task_type(self._make("unknown_task", "text"))
        assert t == "freeform"


# ---------------------------------------------------------------------------
# JSON Schema builder
# ---------------------------------------------------------------------------

class TestBuildJsonSchema:
    def test_basic_string_field(self):
        fields = [{"name": "vendor", "type": "string", "required": True}]
        schema = _build_json_schema(fields)
        assert schema["type"] == "object"
        assert "vendor" in schema["properties"]
        assert schema["properties"]["vendor"]["type"] == "string"
        assert "vendor" in schema["required"]

    def test_optional_field_excluded_from_required(self):
        fields = [
            {"name": "required_field", "type": "string", "required": True},
            {"name": "optional_field", "type": "string", "required": False},
        ]
        schema = _build_json_schema(fields)
        assert "required_field" in schema["required"]
        assert "optional_field" not in schema["required"]

    def test_number_type_mapping(self):
        fields = [{"name": "amount", "type": "number", "required": True}]
        schema = _build_json_schema(fields)
        assert schema["properties"]["amount"]["type"] == "number"

    def test_type_aliases(self):
        fields = [
            {"name": "a", "type": "float", "required": False},
            {"name": "b", "type": "int", "required": False},
            {"name": "c", "type": "bool", "required": False},
        ]
        schema = _build_json_schema(fields)
        assert schema["properties"]["a"]["type"] == "number"
        assert schema["properties"]["b"]["type"] == "integer"
        assert schema["properties"]["c"]["type"] == "boolean"

    def test_description_and_format_forwarded(self):
        fields = [{"name": "date", "type": "string", "description": "ISO date", "format": "date", "required": True}]
        schema = _build_json_schema(fields)
        assert schema["properties"]["date"]["description"] == "ISO date"
        assert schema["properties"]["date"]["format"] == "date"


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

class TestBuildDatasetConfig:
    def _task(self, output_type="structured"):
        return {"type": "extraction", "input": {"type": "text"}, "output": {"type": output_type, "fields": []}}

    def _scoring(self):
        return {"type": "per_field"}

    def _call(self, data_section, internal_type="structured", is_multimodal=False, name="my-bench"):
        return _build_dataset_config(data_section, self._task(), self._scoring(), name, internal_type, is_multimodal)

    def test_local_source_sets_field_mappings(self):
        dc = self._call({"source": "local", "path": ".data/my-bench/train.jsonl"})
        assert dc["input_field"] == "input"
        assert dc["output_field"] == "expected_output"

    def test_local_source_name_from_dotdata_path(self):
        dc = self._call({"source": "local", "path": ".data/my-bench/train.jsonl"})
        assert dc["name"] == "my-bench"
        assert dc["source"] == "local"

    def test_generate_source(self):
        dc = self._call({"source": "generate", "count": 20}, name="gen-bench")
        assert dc["name"] == "gen-bench"
        assert dc["source"] == "local"
        assert dc["input_field"] == "input"
        assert dc["output_field"] == "expected_output"

    def test_huggingface_source(self):
        dc = self._call({"source": "huggingface", "dataset": "org/mydata", "split": "validation"})
        assert dc["name"] == "org/mydata"
        assert dc["source"] == "huggingface"
        assert dc["split"] == "validation"

    def test_multimodal_adds_flags(self):
        dc = self._call({"source": "local", "path": ".data/x/train.jsonl"}, is_multimodal=True)
        assert dc["multimodal_input"] is True
        assert "multimodal_image_field" in dc

    def test_structured_type_builds_schema(self):
        task = {
            "type": "extraction",
            "input": {"type": "text"},
            "output": {
                "type": "structured",
                "fields": [{"name": "vendor", "type": "string", "required": True}],
            },
        }
        dc = _build_dataset_config(
            {"source": "local", "path": ".data/x/train.jsonl"},
            task, self._scoring(), "bench", "structured", False,
        )
        schema = json.loads(dc["schema_json"])
        assert "vendor" in schema["properties"]

    def test_classification_type_builds_labels(self):
        task = {
            "type": "classification",
            "input": {"type": "text"},
            "output": {"type": "label", "labels": ["cat", "dog"]},
        }
        dc = _build_dataset_config(
            {"source": "local", "path": ".data/x/train.jsonl"},
            task, self._scoring(), "bench", "classification", False,
        )
        labels = json.loads(dc["labels"])
        assert "cat" in labels
        assert "dog" in labels


# ---------------------------------------------------------------------------
# Target config
# ---------------------------------------------------------------------------

class TestBuildTargetConfig:
    def test_api_type(self):
        cfg = _build_target_config(
            {"type": "api", "url": "https://api.example.com/extract",
             "body_template": '{"text": "{{text}}"}', "response_path": "data"},
            "my-bench",
        )
        assert cfg["provider_type"] == "api"
        assert cfg["api"]["endpoint"] == "https://api.example.com/extract"
        assert cfg["api"]["response_path"] == "data"
        assert cfg["api"]["body_template"] == '{"text": "{{text}}"}'

    def test_model_type_together(self):
        cfg = _build_target_config(
            {"type": "model", "provider": "together", "model": "google/gemma-4-31B-it"},
            "my-bench",
        )
        assert cfg["provider_type"] == "together"
        assert cfg["model"]["name"] == "google/gemma-4-31B-it"

    def test_model_type_openai(self):
        cfg = _build_target_config(
            {"type": "model", "provider": "openai", "model": "gpt-4o"},
            "my-bench",
        )
        assert cfg["provider_type"] == "openai"

    def test_local_type(self):
        cfg = _build_target_config(
            {"type": "local", "url": "http://localhost:8000/v1", "model": "llama3"},
            "my-bench",
        )
        assert cfg["provider_type"] == "openai"
        assert cfg["openai"]["base_url"] == "http://localhost:8000/v1"
        assert cfg["model"]["name"] == "llama3"

    def test_model_type_optional_overrides(self):
        cfg = _build_target_config(
            {"type": "model", "provider": "together", "model": "x",
             "temperature": 0.5, "max_tokens": 512},
            "bench",
        )
        assert cfg["together"]["temperature"] == 0.5
        assert cfg["together"]["max_tokens"] == 512


# ---------------------------------------------------------------------------
# Full compile_benchmark
# ---------------------------------------------------------------------------

class TestCompileBenchmark:
    def test_compiles_extraction_spec(self, tmp_path):
        path = _write_spec(tmp_path, _minimal_spec())
        compiled = compile_benchmark(path)
        assert compiled.name == "test-bench"
        assert compiled.internal_task_type == "structured"
        assert compiled.dataset_config["input_field"] == "input"
        assert compiled.dataset_config["output_field"] == "expected_output"
        assert compiled.target_config["provider_type"] == "together"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            compile_benchmark("/nonexistent/benchmark.yaml")

    def test_system_prompt_extracted(self, tmp_path):
        spec = _minimal_spec()
        spec["target"]["system_prompt"] = "You are an extractor."
        path = _write_spec(tmp_path, spec)
        compiled = compile_benchmark(path)
        assert compiled.system_prompt == "You are an extractor."


# ---------------------------------------------------------------------------
# validate_benchmark_yaml
# ---------------------------------------------------------------------------

class TestValidateBenchmarkYaml:
    def test_valid_spec_returns_no_errors(self, tmp_path):
        path = _write_spec(tmp_path, _minimal_spec())
        errors = validate_benchmark_yaml(path)
        assert errors == []

    def test_missing_file_returns_error(self):
        errors = validate_benchmark_yaml("/nonexistent/benchmark.yaml")
        assert any("not found" in e.lower() or "exist" in e.lower() for e in errors)

    def test_missing_task_section(self, tmp_path):
        spec = _minimal_spec()
        del spec["task"]
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("task" in e.lower() for e in errors)

    def test_missing_scoring_section(self, tmp_path):
        spec = _minimal_spec()
        del spec["scoring"]
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("scoring" in e.lower() for e in errors)

    def test_invalid_task_type(self, tmp_path):
        spec = _minimal_spec()
        spec["task"]["type"] = "invented_type"
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("type" in e.lower() for e in errors)

    def test_structured_output_without_fields(self, tmp_path):
        spec = _minimal_spec()
        spec["task"]["output"] = {"type": "structured"}  # no fields
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("fields" in e.lower() for e in errors)

    def test_api_target_without_url(self, tmp_path):
        spec = _minimal_spec()
        spec["target"] = {"type": "api", "body_template": '{"x": "{{text}}"}'}
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("url" in e.lower() for e in errors)

    def test_local_source_without_path(self, tmp_path):
        spec = _minimal_spec()
        spec["data"] = {"source": "local"}
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("path" in e.lower() for e in errors)

    def test_huggingface_source_without_dataset(self, tmp_path):
        spec = _minimal_spec()
        spec["data"] = {"source": "huggingface"}
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("dataset" in e.lower() for e in errors)

    def test_invalid_scoring_type(self, tmp_path):
        spec = _minimal_spec()
        spec["scoring"] = {"type": "magic"}
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("scoring" in e.lower() or "type" in e.lower() for e in errors)

    def test_model_target_without_model_name(self, tmp_path):
        spec = _minimal_spec()
        spec["target"] = {"type": "model", "provider": "together"}
        path = _write_spec(tmp_path, spec)
        errors = validate_benchmark_yaml(path)
        assert any("model" in e.lower() for e in errors)
