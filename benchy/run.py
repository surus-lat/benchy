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

Baselines, not decorations: the report also publishes, per field, the score a
*trivial* predictor would get on this very exam (majority for enums/bools/artifacts,
the mean for numbers, the empty string for text and temporals), the signal
`score - baseline`, and whether the field counts as signal at all:
`score >= baseline + epsilon` (default 0.01, sealed by the compiler in (0, 0.10]).
A field that does not beat doing nothing is reported as such — everything published
here is computed from the exam, nothing is a constant.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchy import adapter as _adapter
from benchy import data, metrics, score, types
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
    dimensions = ir["scoring"]["dimensions"]

    def resolve_output(reference: str, field: tuple[str, ...]) -> str:
        # An adapter's artifact output must be a local file inside the run's
        # workspace (spec §10); an escape surfaces as invalid_output.
        return str(data.resolve_within(reference, root, root, list(field), phase="runtime"))

    # Per-field accumulators, filled in the same pass as scoring: the expected side
    # of every row (the exam itself feeds the baselines) and each field's
    # contribution (a failed example contributes 0 to every field, the same
    # denominator discipline as the benchmark score).
    expected_by_field: dict[tuple, list] = {tuple(d["path"]): [] for d in dimensions}
    contrib_by_field: dict[tuple, list] = {tuple(d["path"]): [] for d in dimensions}

    results: list[dict] = []
    for index, inputs, expected in data.examples(ir, root):
        for dimension in dimensions:
            expected_by_field[tuple(dimension["path"])].append(score.value_at(expected, dimension["path"]))
        try:
            prediction = await invoke(inputs)
        except Exception as exc:  # noqa: BLE001 - any failure to produce an output
            for dimension in dimensions:
                contrib_by_field[tuple(dimension["path"])].append(0.0)
            results.append(_failed(index, "execution_error", None, _adapter_error(exc)))
            continue
        try:
            validated = types.validate(prediction, output_schema, phase="runtime", resolve=resolve_output)
        except BenchyError as exc:
            for dimension in dimensions:
                contrib_by_field[tuple(dimension["path"])].append(0.0)
            results.append(_failed(index, "invalid_output", prediction, exc.to_dict()))
            continue
        field_scores, instance = score.score_example(validated, expected, ir)
        for field in field_scores:
            contrib_by_field[tuple(field["path"])].append(field["score"])
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
        "fields": _field_report(ir, expected_by_field, contrib_by_field),
        "summary": {
            "examples": len(results),
            "valid": sum(r["status"] == "valid" for r in results),
            "invalid_outputs": sum(r["status"] == "invalid_output" for r in results),
            "execution_errors": sum(r["status"] == "execution_error" for r in results),
        },
        "results": results,
    }


def _field_report(
    ir: Mapping,
    expected_by_field: Mapping[tuple, list],
    contrib_by_field: Mapping[tuple, list],
) -> list[dict]:
    """Per-field score vs. the trivial baseline computed from the exam itself.

    Dimensions are walked in IR order. A field's score is the mean of its
    contributions over every example — failures included as zero, so this number is
    the field-level analogue of the benchmark score, not a mean over valid outputs
    (which would let a system that fails often look better by answering less).
    """
    output_schema = ir["program"]["output"]
    epsilon = float(ir["scoring"].get("signal_epsilon", 0.01))
    fields = []
    for dimension in ir["scoring"]["dimensions"]:
        path = tuple(dimension["path"])
        node = types.at(output_schema, dimension["path"])
        name = dimension.get("metric", "exact")
        values = expected_by_field[path]
        field_score = sum(contrib_by_field[path]) / len(contrib_by_field[path])
        baseline = _baseline_score(name, dimension.get("params"), node, values)
        fields.append({
            "path": dimension["path"],
            "metric": name,
            "weight": dimension["weight"],
            "score": field_score,
            "baseline": baseline,
            "signal": field_score - baseline,
            "counts_as_signal": bool(field_score >= baseline + epsilon),
        })
    return fields


def _trivial_prediction(node: Mapping, values: Sequence) -> object:
    """The do-nothing predictor for this field's type, fitted on the exam's expecteds."""
    kind = node["type"]
    if kind in ("enum", "bool") or kind in types.ARTIFACTS:
        # majority: the most frequent expected value (Counter is stable by first
        # appearance, so ties break deterministically).
        return Counter(values).most_common(1)[0][0]
    if kind in ("int", "float"):
        # mean: the constant that minimizes absolute error over the exam.
        return sum(values) / len(values)
    return ""  # string, date, time, datetime: the empty answer


def _baseline_score(name: str, params: Mapping | None, node: Mapping, values: Sequence) -> float:
    """Mean score of the trivial prediction under the field's own metric.

    A trivial prediction the metric cannot read (e.g. an empty string where a date
    is expected) simply misses: it scores 0 for that example, which is what a system
    answering nothing would get.
    """
    trivial = _trivial_prediction(node, values)
    total = 0.0
    for expected in values:
        try:
            if name == "exact":
                total += 1.0 if types.equal(trivial, expected, node) else 0.0
            else:
                total += metrics.score(name, trivial, expected, params)
        except Exception:  # noqa: BLE001 - an unreadable trivial prediction misses
            continue
    return total / len(values)


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
