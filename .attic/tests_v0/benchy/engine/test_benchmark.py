"""benchy.benchmark — the run loop, concurrency, retry, timeout, checkpoint."""

from __future__ import annotations

import asyncio
import json

import pytest

from benchy.benchmark import Benchmark, RetryPolicy, _Checkpoint
from benchy.core import Capabilities, Prediction, Record, Response, Score

from .conftest import FakeData, FakeScorer, FakeSystem, FakeTask, erroring_system, flaky_system, slow_system


def make_bench(task=None, data=None, scoring=None, **kw) -> Benchmark:
    task = task or FakeTask()
    data = data or FakeData.of_texts(["a", "b", "c"])
    scoring = scoring or FakeScorer()
    return Benchmark(task=task, data=data, scoring=scoring, **kw)


class TestConstructorValidation:
    def test_rejects_a_task_missing_protocol_members(self):
        class NotATask:
            pass

        with pytest.raises(TypeError, match="Task"):
            Benchmark(task=NotATask(), data=FakeData([]), scoring=FakeScorer())

    def test_rejects_a_data_missing_protocol_members(self):
        class NotData:
            pass

        with pytest.raises(TypeError, match="Data"):
            Benchmark(task=FakeTask(), data=NotData(), scoring=FakeScorer())

    def test_rejects_a_scorer_missing_protocol_members(self):
        class NotAScorer:
            pass

        with pytest.raises(TypeError, match="Scorer"):
            Benchmark(task=FakeTask(), data=FakeData([]), scoring=NotAScorer())


class TestLabel:
    def test_uses_explicit_name_first(self):
        bench = make_bench(name="Explicit", ontology="a/b/c")
        assert bench.label == "Explicit"

    def test_falls_back_to_ontology(self):
        bench = make_bench(ontology="a/b/c")
        assert bench.label == "a/b/c"

    def test_falls_back_to_task_name(self):
        bench = make_bench()
        assert bench.label == "fake-task"


class TestRunBasics:
    async def test_all_samples_succeed(self):
        bench = make_bench(data=FakeData.of_texts(["ok", "ok", "ok"]))
        system = FakeSystem(respond=lambda req, n: Response(text="ok"))
        report = await bench.run(system)
        assert report.n_samples == 3
        assert report.n_errors == 0
        assert report.fitness == pytest.approx(1.0)
        assert len(report.records) == 3

    async def test_record_order_matches_sample_order(self):
        bench = make_bench(data=FakeData.of_texts(["a", "b", "c", "d", "e"]))
        system = FakeSystem()
        report = await bench.run(system)
        assert [r.sample_id for r in report.records] == ["0", "1", "2", "3", "4"]

    async def test_limit_is_honoured(self):
        bench = make_bench(data=FakeData.of_texts(["a", "b", "c", "d"]))
        report = await bench.run(FakeSystem(), limit=2)
        assert report.n_samples == 2
        assert [r.sample_id for r in report.records] == ["0", "1"]

    async def test_run_returns_a_report_with_system_and_scorer_identity(self):
        bench = make_bench()
        system = FakeSystem(url="fake://x")
        report = await bench.run(system)
        assert report.system == "fake://x"
        assert report.scorer == "fake_scorer()"
        assert report.benchmark == bench.label


class TestOrderingUnderOutOfOrderCompletion:
    async def test_records_stay_in_sample_order_even_when_later_samples_finish_first(self):
        # Sample "0" is slow, everyone else is fast -- so completion order is
        # reversed relative to sample order, but Report.records must not be.
        def delay_for(req):
            return 0.06 if req.meta.get("sample_id") == "0" else 0.0

        def respond(req, n):
            return Response(text=f"v{req.meta['sample_id']}")

        data = FakeData.of_texts(["a", "b", "c", "d"], expected=["v0", "v1", "v2", "v3"])
        bench = make_bench(data=data)
        system = FakeSystem(respond=respond, delay=delay_for)
        report = await bench.run(system, concurrency=4)
        assert [r.sample_id for r in report.records] == ["0", "1", "2", "3"]
        assert report.fitness == pytest.approx(1.0)


class TestConcurrency:
    async def test_default_concurrency_comes_from_system_capabilities(self):
        data = FakeData.of_texts([str(i) for i in range(8)])
        bench = make_bench(data=data)
        system = FakeSystem(delay=0.02, capabilities=Capabilities(max_concurrency=2))
        await bench.run(system)
        assert system.max_inflight <= 2

    async def test_explicit_concurrency_overrides_capabilities(self):
        data = FakeData.of_texts([str(i) for i in range(10)])
        bench = make_bench(data=data)
        system = FakeSystem(delay=0.02, capabilities=Capabilities(max_concurrency=8))
        await bench.run(system, concurrency=3)
        assert system.max_inflight <= 3

    async def test_concurrency_is_actually_used_not_just_capped(self):
        """A weak bound (max <= N) would pass even with no concurrency at all."""
        data = FakeData.of_texts([str(i) for i in range(6)])
        bench = make_bench(data=data)
        system = FakeSystem(delay=0.03, capabilities=Capabilities(max_concurrency=4))
        await bench.run(system, concurrency=4)
        assert system.max_inflight == 4


class TestErrorsDoNotAbortTheRun:
    async def test_a_system_error_response_produces_an_error_record_not_a_crash(self):
        data = FakeData.of_texts(["a", "b", "c"])
        bench = make_bench(data=data)
        system = erroring_system({"1"}, capabilities=Capabilities(max_concurrency=4))
        report = await bench.run(system, retries=1)
        assert report.n_samples == 3
        assert report.n_errors == 1
        bad = next(r for r in report.records if r.sample_id == "1")
        assert bad.error is not None
        assert bad.score is None
        good_ids = {r.sample_id for r in report.records if r.error is None}
        assert good_ids == {"0", "2"}

    async def test_errors_are_excluded_from_fitness_not_scored_as_zero(self):
        """Documented design decision: see benchy/benchmark.py module docstring."""
        data = FakeData.of_texts(["a", "b"], expected=["ok:0", "unused"])
        bench = make_bench(data=data)
        system = erroring_system({"1"})
        report = await bench.run(system, retries=1)
        # Sample "0" succeeds and its prediction ("ok:0") matches expected exactly
        # -> if the errored sample were folded in as a 0.0, fitness would be 0.5.
        # Excluded, it's the mean of the one successful score alone: 1.0.
        assert report.fitness == pytest.approx(1.0)

    async def test_a_raised_exception_from_invoke_also_produces_an_error_record(self):
        data = FakeData.of_texts(["a"])
        bench = make_bench(data=data)

        def respond(req, n):
            raise RuntimeError("boom")

        system = FakeSystem(respond=respond)
        report = await bench.run(system, retries=1)
        assert report.n_errors == 1
        assert "boom" in report.records[0].error


class TestValidateSampleFailsLoud:
    async def test_invalid_sample_raises_and_does_not_return_a_report(self):
        task = FakeTask(invalid_ids=frozenset({"1"}))
        data = FakeData.of_texts(["a", "b", "c"])
        bench = make_bench(task=task, data=data)
        system = FakeSystem()
        from benchy.core import SchemaViolation

        with pytest.raises(SchemaViolation):
            await bench.run(system)

    async def test_validation_happens_before_any_system_invocation(self):
        task = FakeTask(invalid_ids=frozenset({"0"}))
        data = FakeData.of_texts(["a", "b"])
        bench = make_bench(task=task, data=data)
        system = FakeSystem()
        from benchy.core import SchemaViolation

        with pytest.raises(SchemaViolation):
            await bench.run(system)
        assert system.calls == 0


class TestRetry:
    async def test_backs_off_and_eventually_succeeds(self):
        data = FakeData.of_texts(["a"], expected=["ok"])
        bench = make_bench(data=data)
        system = flaky_system(fail_times=2)
        report = await bench.run(system, retries=3)
        assert report.n_errors == 0
        assert system.calls == 3
        assert report.fitness == pytest.approx(1.0)

    async def test_gives_up_after_max_attempts(self):
        data = FakeData.of_texts(["a"])
        bench = make_bench(data=data)
        system = flaky_system(fail_times=10)
        report = await bench.run(system, retries=3)
        assert system.calls == 3
        assert report.n_errors == 1

    def test_retry_policy_delay_is_bounded_and_backs_off(self):
        policy = RetryPolicy(max_attempts=5, base_delay=0.1, multiplier=2.0, max_delay=1.0, jitter=0.0)
        assert policy.delay(1) == pytest.approx(0.1)
        assert policy.delay(2) == pytest.approx(0.2)
        assert policy.delay(3) == pytest.approx(0.4)
        assert policy.delay(10) == pytest.approx(1.0)  # capped


class TestTimeout:
    async def test_a_slow_sample_times_out_and_becomes_an_error_record(self):
        data = FakeData.of_texts(["a"])
        bench = make_bench(data=data)
        system = slow_system(delay=0.3)
        report = await bench.run(system, timeout=0.05, retries=1)
        assert report.n_errors == 1
        assert "TimeoutError" in report.records[0].error


class TestOnRecordAndProgress:
    async def test_on_record_fires_once_per_sample(self):
        data = FakeData.of_texts(["a", "b", "c"])
        bench = make_bench(data=data)
        seen: list[str] = []
        await bench.run(FakeSystem(), on_record=lambda r: seen.append(r.sample_id))
        assert set(seen) == {"0", "1", "2"}
        assert len(seen) == 3

    async def test_on_record_may_be_async(self):
        data = FakeData.of_texts(["a", "b"])
        bench = make_bench(data=data)
        seen: list[str] = []

        async def on_record(record: Record) -> None:
            await asyncio.sleep(0)
            seen.append(record.sample_id)

        await bench.run(FakeSystem(), on_record=on_record)
        assert set(seen) == {"0", "1"}

    async def test_progress_prints_to_stderr(self, capsys):
        data = FakeData.of_texts(["a", "b"])
        bench = make_bench(data=data)
        await bench.run(FakeSystem(), progress=True)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "0" in captured.err or "1" in captured.err

    async def test_no_progress_output_by_default(self, capsys):
        data = FakeData.of_texts(["a"])
        bench = make_bench(data=data)
        await bench.run(FakeSystem())
        captured = capsys.readouterr()
        assert captured.err == ""


class TestCheckpoint:
    def test_checkpoint_round_trips_a_record(self, tmp_path):
        path = tmp_path / "ckpt.jsonl"
        ckpt = _Checkpoint(path)
        record = Record(
            sample_id="1",
            prediction=Prediction(value="x"),
            score=Score(value=1.0, scorer="fake_scorer()"),
            latency_ms=1.5,
            usage=None,
            error=None,
            raw_text="x",
        )
        ckpt.append(record)
        loaded = ckpt.load()
        assert set(loaded) == {"1"}
        assert loaded["1"].prediction.value == "x"
        ckpt.clear()
        assert ckpt.load() == {}

    async def test_run_skips_samples_already_present_in_the_checkpoint(self, tmp_path):
        path = tmp_path / "ckpt.jsonl"
        data = FakeData.of_texts(["a", "b", "c"], expected=["ok:0", "ok:1", "ok:2"])
        bench = make_bench(data=data)

        preloaded = Record(
            sample_id="0",
            prediction=Prediction(value="ok:0"),
            score=Score(value=1.0, scorer="fake_scorer()"),
            latency_ms=1.0,
            usage=None,
            error=None,
            raw_text="ok:0",
        )
        _Checkpoint(path).append(preloaded)

        system = FakeSystem(respond=lambda req, n: __import__("benchy.core", fromlist=["Response"]).Response(
            text=f"ok:{req.meta['sample_id']}"
        ))
        report = await bench.run(system, checkpoint=path)

        assert system.calls == 2  # sample "0" was skipped
        assert report.n_samples == 3
        assert not path.exists()  # cleared after a fully successful run

    async def test_checkpoint_file_is_kept_if_a_sample_errors(self, tmp_path):
        path = tmp_path / "ckpt.jsonl"
        data = FakeData.of_texts(["a", "b"])
        bench = make_bench(data=data)
        system = erroring_system({"1"})
        await bench.run(system, checkpoint=path, retries=1)
        assert path.exists()
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        ids = {row["sample_id"] for row in lines}
        assert ids == {"0", "1"}  # even the errored sample's Record gets checkpointed


class TestRunSyncAndCompare:
    def test_run_sync_matches_run(self):
        data = FakeData.of_texts(["a", "b"])
        bench = make_bench(data=data)
        report = bench.run_sync(FakeSystem())
        assert report.n_samples == 2

    async def test_compare_preserves_system_order_and_grades_each(self):
        data = FakeData.of_texts(["a"], expected=["match"])
        bench = make_bench(data=data)
        good = FakeSystem(url="good", respond=lambda req, n: Response_match())
        bad = FakeSystem(url="bad")
        reports = await bench.compare([good, bad])
        assert [r.system for r in reports] == ["good", "bad"]
        assert reports[0].fitness > reports[1].fitness


def Response_match():
    from benchy.core import Response

    return Response(text="match")


class TestRunIsRepeatable:
    async def test_run_can_be_called_many_times_without_mutating_state(self):
        data = FakeData.of_texts(["a", "b", "c"])
        bench = make_bench(data=data)
        r1 = await bench.run(FakeSystem())
        r2 = await bench.run(FakeSystem())
        assert r1.n_samples == r2.n_samples == 3
        assert len(bench.data) == 3  # not consumed

    async def test_as_loss_is_safe_to_call_repeatedly_like_an_optimizer_would(self):
        data = FakeData.of_texts(["a", "b"], expected=["ok:0", "ok:1"])
        bench = make_bench(data=data)
        loss = bench.as_loss()
        system_factory = lambda: FakeSystem(respond=lambda req, n: __import__(
            "benchy.core", fromlist=["Response"]
        ).Response(text=f"ok:{req.meta['sample_id']}"))
        values = [await loss(system_factory()) for _ in range(3)]
        assert values == [pytest.approx(1.0)] * 3


class TestAsLossExactness:
    async def test_as_loss_equals_report_fitness_exactly(self):
        data = FakeData.of_texts(["a", "b", "c"], expected=["ok:0", "x", "ok:2"])
        bench = make_bench(data=data)
        system = FakeSystem(respond=lambda req, n: __import__(
            "benchy.core", fromlist=["Response"]
        ).Response(text=f"ok:{req.meta['sample_id']}"))
        loss_value = await bench.as_loss()(system)
        report = await bench.run(system)
        assert loss_value == report.fitness


class TestOptionalOptimizerAdapters:
    """The lazy dspy/textgrad adapters' contract: when the framework is
    absent, a clean, actionable ImportError naming the package — never an
    opaque NameError from inside a lazily-imported module."""

    def test_dspy_adapter_raises_actionable_importerror_when_dspy_is_absent(self):
        from benchy.loss import as_dspy_metric

        make_bench_ = make_bench  # local alias, linter calm
        bench = make_bench_(data=FakeData.of_texts(["a"], expected=["ok:0"]))
        with pytest.raises(ImportError, match="dspy"):
            as_dspy_metric(bench)

    def test_textgrad_adapter_raises_actionable_importerror_when_textgrad_is_absent(self):
        from benchy.loss import as_textgrad_loss

        bench = make_bench(data=FakeData.of_texts(["a"], expected=["ok:0"]))
        with pytest.raises(ImportError, match="textgrad"):
            as_textgrad_loss(bench)
