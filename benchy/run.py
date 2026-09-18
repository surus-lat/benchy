"""The engine: IR + adapter -> result (handoff §16).

The loop is deliberately flat and readable, because the whole of Benchy's execution
semantics is visible in it: validate the input, invoke, classify what came back,
score it if it is a valid program output, and keep every example in the denominator.

Three statuses, and the difference between them is load-bearing (paper A.12):

    valid            a schema-valid output — *whatever* it scored, including 0
    invalid_output   the adapter returned something that is not a program output
    execution_error  the adapter never produced an output at all

Dataset failures are none of those. They abort (paper A.15), and they do so naturally
here: `data.examples` is a generator, so its diagnostics are raised outside the `try`
blocks that classify adapter failures. No flag distinguishes the two cases; the
control flow does.

Execution is sequential. Spec §16 permits concurrency as an optimization provided
results keep their dataset indices and serialize in dataset order — an optimization
worth adding when a real run needs it, and not before.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchy import adapter as _adapter
from benchy import data, score, types
from benchy.errors import BenchyError

__all__ = ["run"]

#: Spec §17: what the engine requires of an IR it did not just compile.
_IR_KEYS = ("version", "ontology_version", "benchmark", "program", "scoring", "data", "ai-system")


async def run(ir: Mapping, workspace: Path | str, adapter: object) -> dict:
    """Evaluate the AI-system behind `adapter` against the benchmark in `ir`."""
    _check_ir(ir)
    invoke = _adapter.invoker(adapter)
    root = Path(workspace).resolve()
    output_schema = ir["program"]["output"]

    def resolve_output(reference: str, field: tuple[str, ...]) -> str:
        # An adapter's artifact output must be a local file inside the run's
        # workspace (spec §10); an escape surfaces as invalid_output.
        return str(data.resolve_within(reference, root, root, list(field), phase="runtime"))

    results: list[dict] = []
    for index, inputs, expected in data.examples(ir, root):
        try:
            prediction = await invoke(inputs)
        except Exception as exc:  # noqa: BLE001 - any failure to produce an output
            results.append(_failed(index, "execution_error", None, _adapter_error(exc)))
            continue
        try:
            validated = types.validate(prediction, output_schema, phase="runtime", resolve=resolve_output)
        except BenchyError as exc:
            results.append(_failed(index, "invalid_output", prediction, exc.to_dict()))
            continue
        field_scores, instance = score.score_example(validated, expected, ir)
        results.append({
            "index": index,
            "status": "valid",
            "prediction": validated,
            "field_scores": field_scores,
            "score": instance,
            "contribution": instance,
            "error": None,
        })

    return {
        "version": ir["version"],
        "benchmark_score": score.benchmark_score([r["contribution"] for r in results]),
        "summary": {
            "examples": len(results),
            "valid": sum(r["status"] == "valid" for r in results),
            "invalid_outputs": sum(r["status"] == "invalid_output" for r in results),
            "execution_errors": sum(r["status"] == "execution_error" for r in results),
        },
        "results": results,
    }


def _failed(index: int, status: str, prediction: object, error: dict) -> dict:
    """A non-scoring example: `null` score preserved, zero contribution enforced."""
    return {
        "index": index,
        "status": status,
        "prediction": prediction,
        "field_scores": None,
        "score": None,
        "contribution": 0.0,
        "error": error,
    }


def _adapter_error(exc: Exception) -> dict:
    return {"phase": "runtime", "code": "adapter_error", "path": None, "message": f"{type(exc).__name__}: {exc}"}


def _check_ir(ir: Mapping) -> None:
    if not isinstance(ir, Mapping):
        raise BenchyError("runtime", "invalid_ir", f"IR must be a mapping, got {type(ir).__name__}")
    missing = [key for key in _IR_KEYS if key not in ir]
    if missing:
        raise BenchyError("runtime", "invalid_ir", f"IR is missing required keys: {', '.join(missing)}")
    if ir["data"].get("format") != "jsonl":
        raise BenchyError("runtime", "invalid_ir", f"unsupported data format {ir['data'].get('format')!r}", ["data", "format"])
    for key in ("dimensions", "evaluator", "instance_aggregator", "benchmark_aggregator"):
        if key not in ir["scoring"]:
            raise BenchyError("runtime", "invalid_ir", f"IR scoring is missing {key!r}", ["scoring"])
