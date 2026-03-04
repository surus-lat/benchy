from pathlib import Path

import pytest

from src.engine.protocols import InterfaceCapabilities
from src.tasks.group_runner import (
    SubtaskContext,
    TaskGroupSpec,
    _run_default_subtask_async,
    run_task_group,
)


class CompatTask:
    answer_type = "freeform"
    requires_logprobs = False
    prefers_logprobs = False
    requires_multimodal = False
    requires_schema = False
    requires_files = False

    def get_task_name(self):
        return "compat_task"


class IncompatTask(CompatTask):
    requires_multimodal = True


class DummyInterface:
    def __init__(self, supports_multimodal=True):
        self.capabilities = InterfaceCapabilities(
            supports_multimodal=supports_multimodal,
            supports_logprobs=True,
            supports_schema=True,
            supports_files=True,
            supports_batch=True,
            request_modes=["chat"],
        )


def test_run_task_group_custom_runner_writes_status_and_aggregates(tmp_path):
    def run_subtask(context):
        return {
            "aggregate_metrics": {
                "total_samples": 5,
                "valid_samples": 5,
                "error_rate": 0.0,
                "score": 0.8 if context.subtask_name == "a" else 0.6,
            }
        }

    def aggregate_metrics(all_metrics, subtasks):
        total = sum(m.get("score", 0.0) for m in all_metrics.values())
        return {
            "total_samples": 10,
            "valid_samples": 10,
            "error_rate": 0.0,
            "score": total / len(subtasks),
        }

    spec = TaskGroupSpec(
        name="demo_group",
        display_name="Demo Group",
        default_subtasks=["a", "b"],
        run_subtask=run_subtask,
        aggregate_metrics=aggregate_metrics,
    )

    result = run_task_group(
        spec=spec,
        model_name="demo/model",
        output_path=str(tmp_path),
        server_info=None,
        task_config={
            "tasks": ["a", "b"],
            "defaults": {},
            "prompts": {},
            "task_configs": {"a": {}, "b": {}},
            "output": {"subdirectory": "demo_group"},
        },
        limit=2,
        provider_config={"provider_type": "http", "base_url": "http://localhost:9999"},
    )

    assert result["status"] == "passed"
    assert result["metrics"]["score"] == pytest.approx(0.7)

    status_path = Path(result["output_path"]) / "task_status.json"
    assert status_path.exists()


@pytest.mark.asyncio
async def test_default_subtask_skips_on_incompatibility(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.tasks.group_runner.get_interface_for_provider",
        lambda provider_type, connection_info, model_name: DummyInterface(supports_multimodal=False),
    )

    spec = TaskGroupSpec(
        name="demo",
        display_name="Demo",
        prepare_task=lambda context: IncompatTask(),
    )

    context = SubtaskContext(
        subtask_name="x",
        subtask_config={},
        task_config={},
        defaults={},
        prompts={},
        model_name="m",
        provider_type="openai",
        connection_info={},
        output_dir=tmp_path,
        subtask_output_dir=tmp_path / "x",
        limit=1,
        compatibility_mode="skip",
        shared=None,
    )

    result = await _run_default_subtask_async(spec, context)

    assert result["skipped"] is True
    assert "requires multimodal" in result["skip_reason"].lower()


@pytest.mark.asyncio
async def test_default_subtask_runs_benchmark_and_saves_results(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.tasks.group_runner.get_interface_for_provider",
        lambda provider_type, connection_info, model_name: DummyInterface(supports_multimodal=True),
    )

    saved = {}

    class FakeRunner:
        def __init__(self, task, interface, config):
            self.task = task
            self.interface = interface
            self.config = config

        async def run(self, limit=None, no_resume=False):
            return {
                "aggregate_metrics": {"total_samples": 1, "valid_samples": 1, "error_rate": 0.0},
                "per_sample_metrics": [{"valid": True}],
                "samples": [],
            }

    def fake_save_results(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr("src.tasks.group_runner.BenchmarkRunner", FakeRunner)
    monkeypatch.setattr("src.tasks.group_runner.save_results", fake_save_results)

    spec = TaskGroupSpec(
        name="demo",
        display_name="Demo",
        prepare_task=lambda context: CompatTask(),
    )

    context = SubtaskContext(
        subtask_name="x",
        subtask_config={},
        task_config={},
        defaults={"batch_size": 2, "log_samples": False},
        prompts={},
        model_name="m",
        provider_type="openai",
        connection_info={},
        output_dir=tmp_path,
        subtask_output_dir=tmp_path / "x",
        limit=1,
        compatibility_mode="skip",
        shared=None,
    )

    result = await _run_default_subtask_async(spec, context)

    assert result["aggregate_metrics"]["total_samples"] == 1
    assert saved["model_name"] == "m"
    assert saved["task_name"] == "compat_task"
