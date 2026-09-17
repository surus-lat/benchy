"""P7/P8 — the engine loop, status classification, scoring and aggregation.

Covers conformance cases C20–C27 and C29–C31.
"""

from __future__ import annotations

import json

import pytest
from conftest import edit

from benchy import adapter
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run

INVOICE = edit(
    program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
    scoring={"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
)
IR = compile_benchmark(INVOICE)

GOOD = {"supplier": "ACME", "total": 121.0}


def workspace(tmp_path, *expected):
    """A workspace whose exam has one row per `expected` output."""
    rows = [{"input": {"text": f"row-{i}"}, "expected": e} for i, e in enumerate(expected)]
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


async def execute(tmp_path, adapter_impl, *expected, ir=IR):
    return await run(ir, workspace(tmp_path, *expected), adapter_impl)


# ---------------------------------------------------------------------------
# C20 — a valid output
# ---------------------------------------------------------------------------

async def test_c20_valid_output_is_status_valid(tmp_path):
    result = await execute(tmp_path, lambda _: dict(GOOD), GOOD)
    (only,) = result["results"]
    assert only["status"] == "valid"
    assert only["error"] is None
    assert only["prediction"] == GOOD


async def test_perfect_prediction_scores_one(tmp_path):
    result = await execute(tmp_path, lambda _: dict(GOOD), GOOD)
    assert result["benchmark_score"] == 1.0
    assert result["results"][0]["score"] == 1.0
    assert result["results"][0]["contribution"] == 1.0


async def test_async_and_sync_adapters_both_work(tmp_path):
    async def async_adapter(_):
        return dict(GOOD)

    assert (await execute(tmp_path, async_adapter, GOOD))["benchmark_score"] == 1.0
    assert (await execute(tmp_path, lambda _: dict(GOOD), GOOD))["benchmark_score"] == 1.0


async def test_adapter_object_with_invoke_is_accepted(tmp_path):
    class Impl:
        async def invoke(self, _input_object):
            return dict(GOOD)

    assert (await execute(tmp_path, Impl(), GOOD))["benchmark_score"] == 1.0


async def test_adapter_receives_the_validated_input_object(tmp_path):
    seen = []
    await execute(tmp_path, lambda inp: seen.append(inp) or dict(GOOD), GOOD)
    assert seen == [{"text": "row-0"}]


# ---------------------------------------------------------------------------
# C21, C22 — invalid outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prediction,expected_code",
    [
        ({"supplier": "ACME", "total": "lots"}, "wrong_type"),          # C21
        ({"supplier": "ACME", "total": 1.0, "debug": "x"}, "extra_field"),  # C22
        ({"supplier": "ACME"}, "missing_field"),
        ({"debug": "unexpected"}, "missing_field"),
        ("not an object", "wrong_type"),
    ],
)
async def test_c21_c22_c25_invalid_outputs_are_status_invalid_output(tmp_path, prediction, expected_code):
    result = await execute(tmp_path, lambda _: prediction, GOOD)
    (only,) = result["results"]
    assert only["status"] == "invalid_output"
    assert only["error"]["code"] == expected_code
    assert only["score"] is None
    assert only["field_scores"] is None
    assert only["contribution"] == 0.0


async def test_invalid_output_retains_the_raw_prediction(tmp_path):
    result = await execute(tmp_path, lambda _: {"debug": "unexpected"}, GOOD)
    assert result["results"][0]["prediction"] == {"debug": "unexpected"}


async def test_invalid_output_error_carries_the_field_path(tmp_path):
    result = await execute(tmp_path, lambda _: {"supplier": "ACME"}, GOOD)
    assert result["results"][0]["error"]["path"] == ["total"]


# ---------------------------------------------------------------------------
# C23 — execution errors
# ---------------------------------------------------------------------------

async def test_c23_c26_adapter_exception_is_an_execution_error(tmp_path):
    def boom(_):
        raise RuntimeError("provider exploded")

    result = await execute(tmp_path, boom, GOOD)
    (only,) = result["results"]
    assert only["status"] == "execution_error"
    assert only["error"]["code"] == "adapter_error"
    assert "provider exploded" in only["error"]["message"]
    assert only["prediction"] is None
    assert only["score"] is None
    assert only["field_scores"] is None
    assert only["contribution"] == 0.0


async def test_adapter_raising_a_benchy_error_is_still_an_execution_error(tmp_path):
    def boom(_):
        raise BenchyError("runtime", "provider_error", "429")

    result = await execute(tmp_path, boom, GOOD)
    assert result["results"][0]["status"] == "execution_error"


async def test_missing_adapter_binding_aborts_before_any_example(tmp_path):
    with pytest.raises(BenchyError) as exc:
        await run(IR, workspace(tmp_path, GOOD), None)
    assert exc.value.code == "adapter_not_bound"


# ---------------------------------------------------------------------------
# C24, C25, C26 — scoring of failures
# ---------------------------------------------------------------------------

async def test_c24_valid_but_entirely_wrong_output_scores_zero(tmp_path):
    wrong = {"supplier": "OTHER", "total": 0.0}
    result = await execute(tmp_path, lambda _: wrong, GOOD)
    (only,) = result["results"]
    assert only["status"] == "valid"
    assert only["score"] == 0.0
    assert only["contribution"] == 0.0
    assert result["benchmark_score"] == 0.0


async def test_partial_credit_uses_the_declared_weights(tmp_path):
    # supplier weight 1 correct, total weight 3 wrong -> 1/4
    result = await execute(tmp_path, lambda _: {"supplier": "ACME", "total": 0.0}, GOOD)
    assert result["results"][0]["score"] == 0.25


async def test_field_scores_record_path_score_and_weight(tmp_path):
    result = await execute(tmp_path, lambda _: {"supplier": "ACME", "total": 0.0}, GOOD)
    assert result["results"][0]["field_scores"] == [
        {"path": ["supplier"], "score": 1, "weight": 1.0},
        {"path": ["total"], "score": 0, "weight": 3.0},
    ]


async def test_a_zero_weight_field_does_not_affect_the_score(tmp_path):
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"note": "string", "total": "float"}},
        scoring={"weights": {"note": 0, "total": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
    ))
    expected = {"note": "reference", "total": 1.0}
    result = await execute(tmp_path, lambda _: {"note": "different", "total": 1.0}, expected, ir=ir)
    assert result["results"][0]["score"] == 1.0


# ---------------------------------------------------------------------------
# C27 — aggregation keeps failures in the denominator
# ---------------------------------------------------------------------------

async def test_c27_one_failure_among_two_perfect_examples_scores_one_half(tmp_path):
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second call fails")
        return dict(GOOD)

    result = await execute(tmp_path, flaky, GOOD, GOOD)
    assert result["benchmark_score"] == 0.5
    assert result["summary"] == {"examples": 2, "valid": 1, "invalid_outputs": 0, "execution_errors": 1}


async def test_invalid_output_stays_in_the_denominator(tmp_path):
    calls = {"n": 0}

    def half_bad(_):
        calls["n"] += 1
        return dict(GOOD) if calls["n"] == 1 else {"debug": "x"}

    result = await execute(tmp_path, half_bad, GOOD, GOOD)
    assert result["benchmark_score"] == 0.5
    assert result["summary"]["invalid_outputs"] == 1


async def test_summary_counts_every_status(tmp_path):
    seq = [dict(GOOD), {"debug": "x"}, "boom", dict(GOOD)]

    def cycle(_):
        nxt = seq.pop(0)
        if nxt == "boom":
            raise RuntimeError("no")
        return nxt

    result = await execute(tmp_path, cycle, GOOD, GOOD, GOOD, GOOD)
    assert result["summary"] == {"examples": 4, "valid": 2, "invalid_outputs": 1, "execution_errors": 1}
    assert result["benchmark_score"] == 0.5


# ---------------------------------------------------------------------------
# result shape (spec §16) and ordering
# ---------------------------------------------------------------------------

async def test_run_result_has_the_documented_shape(tmp_path):
    result = await execute(tmp_path, lambda _: dict(GOOD), GOOD)
    assert set(result) == {"version", "benchmark_score", "summary", "results"}
    assert result["version"] == "1.0"
    assert set(result["results"][0]) == {
        "index", "status", "prediction", "field_scores", "score", "contribution", "error",
    }


async def test_results_are_emitted_in_dataset_order_with_their_indices(tmp_path):
    result = await execute(tmp_path, lambda _: dict(GOOD), GOOD, GOOD, GOOD)
    assert [r["index"] for r in result["results"]] == [0, 1, 2]


async def test_result_is_json_serializable(tmp_path):
    result = await execute(tmp_path, lambda _: {"debug": "x"}, GOOD)
    json.dumps(result)


async def test_dataset_error_aborts_the_run_without_a_benchmark_score(tmp_path):
    (tmp_path / "exam.jsonl").write_text('{"input": {"text": "a"}, "expected": {}}')
    with pytest.raises(BenchyError) as exc:
        await run(IR, tmp_path, lambda _: dict(GOOD))
    assert exc.value.phase == "dataset"


# ---------------------------------------------------------------------------
# C29, C30, C31 — typed equality through the engine
# ---------------------------------------------------------------------------

async def test_c29_date_equality_is_semantic(tmp_path):
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"when": "datetime"}},
        scoring={"weights": {"when": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
    ))
    expected = {"when": "2026-09-13T12:00:00Z"}
    same = await execute(tmp_path, lambda _: {"when": "2026-09-13T09:00:00-03:00"}, expected, ir=ir)
    assert same["benchmark_score"] == 1.0


@pytest.mark.parametrize("content,score", [(b"same", 1.0), (b"different", 0.0)])
async def test_c30_c31_artifact_equality_is_byte_for_byte(tmp_path, content, score):
    (tmp_path / "ref.png").write_bytes(b"same")
    (tmp_path / "out.png").write_bytes(content)
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"image": "image"}},
        scoring={"weights": {"image": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
    ))
    (tmp_path / "exam.jsonl").write_text(json.dumps({"input": {"text": "a"}, "expected": {"image": "ref.png"}}))
    result = await run(ir, tmp_path, lambda _: {"image": str(tmp_path / "out.png")})
    assert result["benchmark_score"] == score


async def test_adapter_artifact_output_must_stay_inside_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "outside.png").write_bytes(b"x")
    (ws / "ref.png").write_bytes(b"x")
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"image": "image"}},
        scoring={"weights": {"image": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
    ))
    (ws / "exam.jsonl").write_text(json.dumps({"input": {"text": "a"}, "expected": {"image": "ref.png"}}))
    result = await run(ir, ws, lambda _: {"image": str(tmp_path / "outside.png")})
    (only,) = result["results"]
    assert only["status"] == "invalid_output"
    assert only["error"]["code"] == "path_escape"


# ---------------------------------------------------------------------------
# IR shape validation (spec §17)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drop", ["version", "program", "scoring", "data", "ai-system"])
async def test_malformed_ir_is_rejected(tmp_path, drop):
    ir = dict(IR)
    ir.pop(drop)
    with pytest.raises(BenchyError) as exc:
        await run(ir, workspace(tmp_path, GOOD), lambda _: dict(GOOD))
    assert exc.value.code == "invalid_ir"


async def test_unsupported_data_format_in_ir_is_rejected(tmp_path):
    ir = json.loads(json.dumps(IR))
    ir["data"]["format"] = "csv"
    with pytest.raises(BenchyError) as exc:
        await run(ir, workspace(tmp_path, GOOD), lambda _: dict(GOOD))
    assert exc.value.code == "invalid_ir"


# ---------------------------------------------------------------------------
# adapter registry — the CLI's lookup path, off the execution path
# ---------------------------------------------------------------------------

def test_registry_binds_and_resolves_an_external_ai_system():
    adapter.clear()

    def sentinel(_):
        return dict(GOOD)

    adapter.bind("external:invoice-extractor-v7", sentinel)
    assert adapter.resolve({"type": "external", "id": "invoice-extractor-v7"}) is sentinel
    adapter.clear()


def test_registry_binds_a_model_provider():
    adapter.clear()

    def sentinel(_):
        return dict(GOOD)

    adapter.bind("model:openai", sentinel)
    assert adapter.resolve({"type": "model", "provider": "openai", "model": "m"}) is sentinel
    adapter.clear()


def test_unbound_ai_system_is_a_setup_error():
    adapter.clear()
    with pytest.raises(BenchyError) as exc:
        adapter.resolve({"type": "external", "id": "nobody"})
    assert exc.value.code == "adapter_not_bound"
