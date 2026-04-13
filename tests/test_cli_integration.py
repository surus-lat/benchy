"""Integration tests for the second-layer CLI pipeline.

Tests _resolve_benchmark_path and _apply_benchmark_to_args without running
a real eval or making any API calls. Uses existing specs in benchmarks/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

import pytest

# Run all tests from the project root so discovery paths work
pytestmark = pytest.mark.usefixtures("chdir_to_project_root")


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def chdir_to_project_root(tmp_path, project_root, monkeypatch):
    """All discovery tests must run from the project root."""
    monkeypatch.chdir(project_root)


# ---------------------------------------------------------------------------
# _resolve_benchmark_path
# ---------------------------------------------------------------------------

class TestResolveBenchmarkPath:
    def _resolve(self, arg):
        from src.benchy_cli_eval import _resolve_benchmark_path
        return _resolve_benchmark_path(arg)

    def test_explicit_path_returned_as_is(self):
        result = self._resolve("benchmarks/request-by-email.yaml")
        assert result == "benchmarks/request-by-email.yaml"

    def test_multiple_specs_raises_system_exit(self):
        # benchmarks/ contains multiple specs → ambiguous
        with pytest.raises(SystemExit) as exc:
            self._resolve(None)
        assert "Multiple" in str(exc.value) or "benchmark" in str(exc.value).lower()

    def test_nonexistent_explicit_path_still_returned(self):
        # _resolve doesn't check existence; compile_benchmark does
        result = self._resolve("benchmarks/made-up.yaml")
        assert result == "benchmarks/made-up.yaml"


# ---------------------------------------------------------------------------
# _apply_benchmark_to_args
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "benchmark": "benchmarks/request-by-email.yaml",
        "config": None,
        "config_ref": None,
        "api_url": None,
        "task_type": None,
        "system_prompt": None,
        "user_prompt_template": None,
        "dataset_schema_json": None,
        "dataset_labels": None,
        "base_url": None,
        "provider": None,
        "model_name": None,
        "dataset_name": None,
        "dataset": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestApplyBenchmarkToArgs:
    def _apply(self, spec_path="benchmarks/request-by-email.yaml"):
        from src.benchy_cli_eval import _apply_benchmark_to_args
        args = _make_args(benchmark=spec_path)
        _apply_benchmark_to_args(args)
        return args

    def test_task_type_is_set(self):
        args = self._apply()
        assert args.task_type == "structured"

    def test_provider_is_set(self):
        args = self._apply()
        assert args.provider == "together"

    def test_model_name_is_set(self):
        args = self._apply()
        assert args.model_name  # non-empty

    def test_dataset_name_is_set(self):
        args = self._apply()
        assert args.dataset_name  # non-empty string

    def test_schema_json_injected_for_structured(self):
        args = self._apply()
        # request-by-email has structured output → schema_json must be set
        assert hasattr(args, "dataset_schema_json")
        assert args.dataset_schema_json  # non-empty

    def test_input_field_in_dataset_config(self):
        from src.benchmark_compiler import compile_benchmark
        compiled = compile_benchmark("benchmarks/request-by-email.yaml")
        assert compiled.dataset_config["input_field"] == "input"
        assert compiled.dataset_config["output_field"] == "expected_output"

    def test_api_target_sets_api_url(self, tmp_path, monkeypatch):
        import yaml
        spec = {
            "benchmark": {
                "name": "api-test",
                "description": "test",
                "task": {
                    "type": "qa",
                    "input": {"type": "text", "description": "q"},
                    "output": {"type": "text"},
                },
                "scoring": {"type": "semantic"},
                "data": {"source": "local", "path": ".data/api-test/train.jsonl"},
                "target": {
                    "type": "api",
                    "url": "https://api.example.com/ask",
                    "body_template": '{"q": "{{text}}"}',
                    "response_path": "answer",
                },
            }
        }
        p = tmp_path / "api-test.yaml"
        p.write_text(yaml.dump(spec))

        from src.benchy_cli_eval import _apply_benchmark_to_args
        args = _make_args(benchmark=str(p))
        _apply_benchmark_to_args(args)

        assert args.api_url == "https://api.example.com/ask"
        assert args.api_response_path == "answer"

    def test_local_target_sets_base_url(self, tmp_path):
        import yaml
        spec = {
            "benchmark": {
                "name": "local-test",
                "description": "test",
                "task": {
                    "type": "qa",
                    "input": {"type": "text", "description": "q"},
                    "output": {"type": "text"},
                },
                "scoring": {"type": "semantic"},
                "data": {"source": "local", "path": ".data/local-test/train.jsonl"},
                "target": {
                    "type": "local",
                    "url": "http://localhost:9999/v1",
                    "model": "my-local-model",
                },
            }
        }
        p = tmp_path / "local-test.yaml"
        p.write_text(yaml.dump(spec))

        from src.benchy_cli_eval import _apply_benchmark_to_args
        args = _make_args(benchmark=str(p))
        _apply_benchmark_to_args(args)

        assert args.base_url == "http://localhost:9999/v1"
        assert args.model_name == "my-local-model"


# ---------------------------------------------------------------------------
# benchy validate CLI command
# ---------------------------------------------------------------------------

class TestBenchyValidateCLI:
    def _run(self, *extra_args):
        result = subprocess.run(
            [sys.executable, "-m", "src.benchy_cli", "validate"] + list(extra_args),
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        return result

    def test_valid_spec_exits_zero(self):
        result = self._run("--benchmark", "benchmarks/request-by-email.yaml")
        assert result.returncode == 0

    def test_nonexistent_spec_exits_nonzero(self):
        result = self._run("--benchmark", "benchmarks/does-not-exist.yaml")
        assert result.returncode != 0

    def test_valid_spec_prints_success(self):
        result = self._run("--benchmark", "benchmarks/request-by-email.yaml")
        output = result.stdout + result.stderr
        assert "valid" in output.lower() or "✓" in output
