"""Benchmark compiler: translates benchmark.yaml into engine-ready configuration.

This module is the bridge between the second-layer user-facing spec and the
existing BenchmarkRunner/handler system. It does NOT modify the engine.

Usage::

    from src.benchmark_compiler import compile_benchmark, validate_benchmark_yaml

    compiled = compile_benchmark("benchmark.yaml")
    errors = validate_benchmark_yaml("benchmark.yaml")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Task-type mapping
# ---------------------------------------------------------------------------

# Maps (benchmark task type, output type, has_image_input) → internal handler type
_TASK_TYPE_MAP: Dict[Tuple[str, str, bool], str] = {
    ("extraction", "structured", False): "structured",
    ("extraction", "structured", True): "structured",
    ("classification", "label", False): "classification",
    ("classification", "label", True): "classification",
    ("qa", "text", False): "freeform",
    ("qa", "text", True): "freeform",
    ("translation", "text", False): "freeform",
    ("translation", "text", True): "freeform",
    ("freeform", "text", False): "freeform",
    ("freeform", "text", True): "freeform",
    # Extra aliases
    ("summarization", "text", False): "freeform",
    ("summarization", "text", True): "freeform",
}

# Fallback: derive from output type alone
_OUTPUT_TYPE_FALLBACK: Dict[str, str] = {
    "structured": "structured",
    "label": "classification",
    "text": "freeform",
    "score": "freeform",
}

_IMAGE_INPUT_TYPES = {"image", "document", "pdf"}


def _resolve_task_type(task_section: Dict[str, Any]) -> Tuple[str, bool]:
    """Return (internal_task_type, is_multimodal)."""
    task_type = task_section.get("type", "").lower()
    input_type = task_section.get("input", {}).get("type", "text").lower()
    output_type = task_section.get("output", {}).get("type", "text").lower()
    is_multimodal = input_type in _IMAGE_INPUT_TYPES

    internal = _TASK_TYPE_MAP.get((task_type, output_type, is_multimodal))
    if internal is None:
        # Try without image flag
        internal = _TASK_TYPE_MAP.get((task_type, output_type, False))
    if internal is None:
        internal = _OUTPUT_TYPE_FALLBACK.get(output_type, "freeform")

    return internal, is_multimodal


# ---------------------------------------------------------------------------
# JSON Schema builder
# ---------------------------------------------------------------------------

_FIELD_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "number": "number",
    "float": "number",
    "integer": "integer",
    "int": "integer",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


def _build_json_schema(fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a JSON Schema object from benchmark.yaml field definitions."""
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for f in fields:
        name = f["name"]
        json_type = _FIELD_TYPE_MAP.get(f.get("type", "string").lower(), "string")
        prop: Dict[str, Any] = {"type": json_type}
        if "description" in f:
            prop["description"] = f["description"]
        if "format" in f:
            prop["format"] = f["format"]
        if f.get("enum"):
            prop["enum"] = f["enum"]
        properties[name] = prop
        # Default required=True for all fields unless explicitly false
        if f.get("required", True):
            required.append(name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_dataset_config(
    data_section: Dict[str, Any],
    task_section: Dict[str, Any],
    scoring_section: Dict[str, Any],
    benchmark_name: str,
    internal_task_type: str,
    is_multimodal: bool,
) -> Dict[str, Any]:
    """Build a dataset_config dict compatible with _build_dataset_config_from_args output."""
    source = data_section.get("source", "local")
    dc: Dict[str, Any] = {}

    if source == "local":
        path = data_section.get("path", "")
        p = Path(path) if path else Path(".data") / benchmark_name
        if str(p).startswith(".data/") or str(p).startswith(".data\\"):
            parts = p.parts
            dc["name"] = parts[1] if len(parts) > 1 else benchmark_name
        elif path.endswith(".jsonl") or path.endswith(".csv"):
            dc["name"] = path
        else:
            dc["name"] = path or benchmark_name
        dc["source"] = "local"

    elif source == "generate":
        # Data will be in .data/<benchmark_name>/ after synthesize-data runs
        dc["name"] = benchmark_name
        dc["source"] = "local"

    elif source == "huggingface":
        dc["name"] = data_section.get("dataset", "")
        dc["source"] = "huggingface"
        dc["split"] = data_section.get("split", "test")

    # Multimodal
    if is_multimodal:
        dc["multimodal_input"] = True
        image_field = task_section.get("input", {}).get("field", "image_path")
        dc["multimodal_image_field"] = image_field

    # Structured: inline JSON schema from field definitions
    if internal_task_type == "structured":
        fields = task_section.get("output", {}).get("fields", [])
        if fields:
            schema = _build_json_schema(fields)
            dc["schema_json"] = json.dumps(schema)

    # Classification: inline labels
    if internal_task_type == "classification":
        labels = task_section.get("output", {}).get("labels", [])
        if labels:
            label_map = {str(lbl): str(lbl) for lbl in labels}
            dc["labels"] = json.dumps(label_map)

    return dc


def _build_target_config(
    target_section: Dict[str, Any],
    benchmark_name: str,
) -> Dict[str, Any]:
    """Return a config dict ready to merge into run_eval's config."""
    target_type = target_section.get("type", "model")
    cfg: Dict[str, Any] = {}

    if target_type == "api":
        body_template = target_section.get(
            "body_template",
            '{"input": "{{text}}"}',
        )
        api_cfg: Dict[str, Any] = {
            "endpoint": target_section["url"],
            "body_template": body_template,
            "timeout": target_section.get("timeout", 120),
            "max_retries": 3,
        }
        if "response_path" in target_section:
            api_cfg["response_path"] = target_section["response_path"]
        if "headers" in target_section:
            api_cfg["headers"] = target_section["headers"]
        if "api_key_env" in target_section:
            api_cfg["api_key_env"] = target_section["api_key_env"]
        cfg["provider_type"] = "api"
        cfg["model"] = {"name": target_section.get("name", benchmark_name)}
        cfg["api"] = api_cfg

    elif target_type == "local":
        # OpenAI-compatible local endpoint (vLLM, llama.cpp, etc.)
        base_url = target_section.get("url", "http://localhost:8000/v1")
        model_name = target_section.get("model", "local-model")
        cfg["provider_type"] = "openai"
        cfg["model"] = {"name": model_name}
        cfg["openai"] = {
            "base_url": base_url,
            "api_key_env": "OPENAI_API_KEY",
            "timeout": target_section.get("timeout", 120),
            "max_retries": 3,
            "max_concurrent": target_section.get("max_concurrent", 5),
            "temperature": target_section.get("temperature", 0.0),
            "max_tokens": target_section.get("max_tokens", 2048),
            "max_tokens_param_name": "max_tokens",
            "api_endpoint": "auto",
        }

    else:
        # target.type: model — named cloud provider
        provider = target_section.get("provider", "openai").lower()
        model_name = target_section.get("model", "")
        cfg["provider_type"] = provider
        cfg["model"] = {"name": model_name}
        # Provider-specific config defaults; keys will be resolved by _build_cli_only_config
        # We only need to set base_url if overriding the default.
        provider_cfg: Dict[str, Any] = {}
        if "base_url" in target_section:
            provider_cfg["base_url"] = target_section["base_url"]
        if "api_key_env" in target_section:
            provider_cfg["api_key_env"] = target_section["api_key_env"]
        if "temperature" in target_section:
            provider_cfg["temperature"] = target_section["temperature"]
        if "max_tokens" in target_section:
            provider_cfg["max_tokens"] = target_section["max_tokens"]
        if provider_cfg:
            cfg[provider] = provider_cfg

    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CompiledBenchmark:
    """Result of compiling benchmark.yaml. Ready to inject into the eval pipeline."""

    name: str
    internal_task_type: str          # structured | freeform | classification
    dataset_config: Dict[str, Any]   # equivalent to _build_dataset_config_from_args output
    target_config: Dict[str, Any]    # provider/model config dict for run_eval
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    scoring: Dict[str, Any] = field(default_factory=dict)


def compile_benchmark(path: str = "benchmark.yaml") -> CompiledBenchmark:
    """Load and compile benchmark.yaml into engine-ready structures.

    Args:
        path: Path to benchmark.yaml (default: ./benchmark.yaml)

    Returns:
        CompiledBenchmark dataclass with all fields populated.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is invalid or missing required sections.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"benchmark.yaml not found at {p.resolve()}. "
            f"Run 'benchy create' to create one, or check the path."
        )

    with open(p) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"{path} is empty.")

    # Support both top-level 'benchmark:' key and flat format
    spec = raw.get("benchmark", raw)

    benchmark_name = spec.get("name", "benchmark")
    task_section = spec.get("task", {})
    scoring_section = spec.get("scoring", {})
    data_section = spec.get("data", {})
    target_section = spec.get("target", {})

    internal_task_type, is_multimodal = _resolve_task_type(task_section)

    dataset_config = _build_dataset_config(
        data_section,
        task_section,
        scoring_section,
        benchmark_name,
        internal_task_type,
        is_multimodal,
    )

    target_config = _build_target_config(target_section, benchmark_name)

    system_prompt = target_section.get("system_prompt") or spec.get("system_prompt")
    user_prompt_template = (
        spec.get("prompt", {}).get("user_template")
        or spec.get("user_prompt_template")
    )

    return CompiledBenchmark(
        name=benchmark_name,
        internal_task_type=internal_task_type,
        dataset_config=dataset_config,
        target_config=target_config,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        scoring=scoring_section,
    )


def validate_benchmark_yaml(path: str = "benchmark.yaml") -> List[str]:
    """Pre-flight validation of benchmark.yaml.

    Returns:
        List of error message strings. Empty list means valid.
    """
    p = Path(path)
    if not p.exists():
        return [f"File not found: {p}"]

    try:
        with open(p) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]

    if not raw:
        return ["benchmark.yaml is empty"]

    spec = raw.get("benchmark", raw)
    errors: List[str] = []

    # Required top-level sections
    for section in ("task", "scoring", "data", "target"):
        if section not in spec:
            errors.append(f"Missing required section: '{section}:'")

    if "name" not in spec:
        errors.append("Missing 'name:' field")

    # --- task ---
    if "task" in spec:
        t = spec["task"]
        if "type" not in t:
            errors.append("task.type is required")
        valid_task_types = {"extraction", "classification", "qa", "translation", "freeform", "summarization"}
        if t.get("type") and t["type"] not in valid_task_types:
            errors.append(
                f"task.type '{t['type']}' is not recognized. "
                f"Valid values: {', '.join(sorted(valid_task_types))}"
            )
        if "input" not in t:
            errors.append("task.input is required")
        if "output" not in t:
            errors.append("task.output is required")
        elif t.get("output", {}).get("type") == "structured":
            if not t["output"].get("fields"):
                errors.append(
                    "task.output.fields is required when task.output.type is 'structured'. "
                    "List the fields your AI should extract."
                )

    # --- scoring ---
    if "scoring" in spec:
        s = spec["scoring"]
        if "type" not in s:
            errors.append("scoring.type is required")
        valid_scoring = {"binary", "per_field", "semantic", "custom"}
        if s.get("type") and s["type"] not in valid_scoring:
            errors.append(
                f"scoring.type '{s['type']}' is not recognized. "
                f"Valid values: {', '.join(sorted(valid_scoring))}"
            )

    # --- data ---
    if "data" in spec:
        d = spec["data"]
        if "source" not in d:
            errors.append("data.source is required")
        valid_sources = {"local", "generate", "huggingface"}
        if d.get("source") and d["source"] not in valid_sources:
            errors.append(
                f"data.source '{d['source']}' is not recognized. "
                f"Valid values: {', '.join(sorted(valid_sources))}"
            )
        if d.get("source") == "local" and "path" not in d:
            errors.append("data.path is required when data.source is 'local'")
        if d.get("source") == "huggingface" and "dataset" not in d:
            errors.append("data.dataset is required when data.source is 'huggingface'")

    # --- target ---
    if "target" in spec:
        tgt = spec["target"]
        if "type" not in tgt:
            errors.append("target.type is required")
        valid_target_types = {"api", "model", "local"}
        if tgt.get("type") and tgt["type"] not in valid_target_types:
            errors.append(
                f"target.type '{tgt['type']}' is not recognized. "
                f"Valid values: {', '.join(sorted(valid_target_types))}"
            )
        if tgt.get("type") == "api":
            if "url" not in tgt:
                errors.append("target.url is required when target.type is 'api'")
            if "body_template" not in tgt:
                errors.append(
                    "target.body_template is required when target.type is 'api'. "
                    "Example: '{\"input\": \"{{text}}\"}'"
                )
        elif tgt.get("type") == "model":
            if "provider" not in tgt:
                errors.append(
                    "target.provider is required when target.type is 'model'. "
                    "Example: openai, anthropic, together"
                )
            if "model" not in tgt:
                errors.append("target.model is required when target.type is 'model'")
        elif tgt.get("type") == "local":
            if "url" not in tgt:
                errors.append(
                    "target.url is required when target.type is 'local'. "
                    "Example: http://localhost:8000/v1"
                )

    return errors
