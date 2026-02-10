import pytest

from src.engine.benchmark_runner import BenchmarkRunner
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
