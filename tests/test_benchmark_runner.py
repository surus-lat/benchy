import json

import pytest

from src.engine.benchmark_runner import BenchmarkRunner
from src.engine.benchmark_runner import save_results
from src.engine.protocols import InterfaceCapabilities


class FakeTask:
    answer_type = "freeform"
    requires_logprobs = False
    prefers_logprobs = False
    requires_multimodal = False
    requires_schema = False
    requires_files = False

    def __init__(self, samples, fail_metric_for=None):
        self._samples = list(samples)
        self._fail_metric_for = fail_metric_for or set()
        self.loaded = False

    def load(self):
        self.loaded = True

    def get_samples(self, limit=None):
        data = self._samples[:limit] if limit else self._samples
        for sample in data:
            yield sample

    def get_prompt(self, sample):
        return "system", sample.get("text", "")

    def get_task_name(self):
        return "fake_task"

    def calculate_metrics(self, prediction, expected, sample, error=None, error_type=None):
        if sample["id"] in self._fail_metric_for:
            raise ValueError("metric computation failed")
        return {
            "valid": error is None,
            "score": 1.0 if prediction == expected and error is None else 0.0,
            "error_type": error_type,
        }

    def get_error_metrics(self, error, error_type=None):
        return {
            "valid": False,
            "score": 0.0,
            "error": error,
            "error_type": error_type,
        }

    def aggregate_metrics(self, all_metrics):
        total = len(all_metrics)
        valid = sum(1 for m in all_metrics if m.get("valid"))
        errors = total - valid
        return {
            "valid_samples": valid,
            "error_count": errors,
            "error_rate": (errors / total) if total else 0.0,
            "score": (sum(m.get("score", 0.0) for m in all_metrics) / total) if total else 0.0,
        }


class FakeInterface:
    capabilities = InterfaceCapabilities(
        supports_multimodal=True,
        supports_logprobs=True,
        supports_schema=True,
        supports_files=True,
        supports_batch=True,
        request_modes=["chat"],
    )

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.generate_calls = 0
        self.closed = False

    def prepare_request(self, sample, task):
        return {"sample_id": sample["id"]}

    async def generate_batch(self, requests):
        self.generate_calls += 1
        return [self._outputs.pop(0) for _ in requests]

    async def test_connection(self, max_retries=3, timeout=30):
        return True

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_runner_handles_metric_exceptions_with_task_error_metrics(tmp_path):
    task = FakeTask(
        samples=[
            {"id": "s1", "text": "a", "expected": "A"},
            {"id": "s2", "text": "b", "expected": "B"},
        ],
        fail_metric_for={"s2"},
    )
    interface = FakeInterface(
        outputs=[
            {"output": "A", "raw": "A", "error": None, "error_type": None},
            {"output": "B", "raw": "B", "error": None, "error_type": "invalid_response"},
        ]
    )

    runner = BenchmarkRunner(
        task=task,
        interface=interface,
        config={"model_name": "m", "batch_size": 2, "output_dir": str(tmp_path), "log_samples": False},
    )

    results = await runner.run(limit=None, no_resume=True)

    assert task.loaded is True
    assert interface.closed is True
    assert len(results["per_sample_metrics"]) == 2
    assert results["per_sample_metrics"][1]["valid"] is False
    assert results["per_sample_metrics"][1]["error"] == "metric computation failed"
    assert results["aggregate_metrics"]["total_samples"] == 2


@pytest.mark.asyncio
async def test_runner_uses_checkpoint_and_skips_generation_when_already_complete(tmp_path, monkeypatch):
    samples = [
        {"id": "s1", "text": "a", "expected": "A"},
        {"id": "s2", "text": "b", "expected": "B"},
    ]
    task = FakeTask(samples=samples)
    interface = FakeInterface(outputs=[])

    import src.engine.benchmark_runner as runner_module

    completed_ids = {"s1", "s2"}
    metrics_by_id = {
        "s1": {"valid": True, "score": 1.0},
        "s2": {"valid": True, "score": 1.0},
    }

    monkeypatch.setattr(
        runner_module,
        "load_checkpoint",
        lambda path, config_hash: (completed_ids, metrics_by_id),
    )

    runner = BenchmarkRunner(
        task=task,
        interface=interface,
        config={"model_name": "m", "batch_size": 2, "output_dir": str(tmp_path), "log_samples": False},
    )

    results = await runner.run(limit=None, no_resume=False)

    assert interface.generate_calls == 0
    assert len(results["per_sample_metrics"]) == 2
    assert results["aggregate_metrics"]["total_samples"] == 2
    assert results["aggregate_metrics"]["performance_summary"]["primary_metric"] == "score"


@pytest.mark.asyncio
async def test_runner_adds_performance_summary_to_aggregate_metrics(tmp_path):
    task = FakeTask(
        samples=[
            {"id": "s1", "text": "a", "expected": "A"},
            {"id": "s2", "text": "b", "expected": "B"},
            {"id": "s3", "text": "c", "expected": "X"},
            {"id": "s4", "text": "d", "expected": "Y"},
        ]
    )
    interface = FakeInterface(
        outputs=[
            {"output": "A", "raw": "A", "error": None, "error_type": None},
            {"output": "B", "raw": "B", "error": None, "error_type": None},
            {"output": "C", "raw": "C", "error": None, "error_type": None},
            {"output": "D", "raw": "D", "error": None, "error_type": None},
        ]
    )

    runner = BenchmarkRunner(
        task=task,
        interface=interface,
        config={"model_name": "m", "batch_size": 4, "output_dir": str(tmp_path), "log_samples": False},
    )

    results = await runner.run(limit=None, no_resume=True)

    performance_summary = results["aggregate_metrics"]["performance_summary"]
    assert performance_summary["status"] == "ok"
    assert performance_summary["primary_metric"] == "score"
    assert performance_summary["samples_with_metric"] == 4
    assert performance_summary["lowest_samples"][0]["sample_id"] in {"s3", "s4"}


def test_save_results_calls_task_additional_artifact_hook(tmp_path):
    class FakeTaskWithArtifacts:
        def __init__(self):
            self.called = False

        def build_additional_artifacts(
            self,
            *,
            results,
            output_dir,
            safe_model_name,
            timestamp,
            task_name,
        ):
            self.called = True
            marker = output_dir / f"{safe_model_name}_{timestamp}_marker.txt"
            marker.write_text(task_name, encoding="utf-8")
            return [marker]

    task = FakeTaskWithArtifacts()
    results = {
        "aggregate_metrics": {"score": 1.0},
        "samples": [],
    }

    save_results(
        results=results,
        output_dir=tmp_path,
        model_name="provider/model",
        task_name="fake_task",
        log_samples=False,
        task_instance=task,
    )

    assert task.called is True
    assert any(path.name.endswith("_marker.txt") for path in tmp_path.iterdir())


def test_save_results_writes_per_sample_metrics_and_performance_summary_artifacts(tmp_path):
    results = {
        "aggregate_metrics": {"score": 0.5},
        "per_sample_metrics": [
            {"sample_id": "s1", "valid": True, "document_extraction_score": 0.95, "field_f1_partial": 0.9},
            {"sample_id": "s2", "valid": True, "document_extraction_score": 0.70, "field_f1_partial": 0.8},
            {"sample_id": "s3", "valid": False, "document_extraction_score": 0.10, "field_f1_partial": 0.1},
            {"sample_id": "s4", "valid": True, "document_extraction_score": 0.40, "field_f1_partial": 0.5},
        ],
        "samples": [],
    }

    save_results(
        results=results,
        output_dir=tmp_path,
        model_name="provider/model",
        task_name="fake_task",
        log_samples=False,
    )

    per_sample_metrics_files = list(tmp_path.glob("*_per_sample_metrics.json"))
    performance_summary_files = list(tmp_path.glob("*_performance_summary.json"))
    report_files = list(tmp_path.glob("*_report.txt"))

    assert len(per_sample_metrics_files) == 1
    assert len(performance_summary_files) == 1
    assert len(report_files) == 1

    per_sample_payload = json.loads(per_sample_metrics_files[0].read_text(encoding="utf-8"))
    assert per_sample_payload["performance_summary"]["primary_metric"] == "document_extraction_score"
    assert per_sample_payload["entries"][0]["performance_metric"] == "document_extraction_score"
    assert per_sample_payload["entries"][0]["performance_bucket"].startswith("q")

    performance_summary_payload = json.loads(performance_summary_files[0].read_text(encoding="utf-8"))
    assert performance_summary_payload["performance_summary"]["quartiles"]["median"] == pytest.approx(0.55)

    report_text = report_files[0].read_text(encoding="utf-8")
    assert "Performance Summary:" in report_text
    assert "primary_metric: document_extraction_score" in report_text
