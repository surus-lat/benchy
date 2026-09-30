"""`benchy compile` and `benchy run` — the human surface.

Two verbs, because the paper has two phases: compilation turns source into IR, and
execution turns IR plus an adapter into a result. Keeping them separate on the
command line is what makes paper A.8's invariant observable from a shell — compile
once, delete the YAML, and the IR still runs.

`--adapter module:attr` names your AI-system's adapter explicitly: no searching a
module for something that looks adapter-shaped. Naming the object is one word longer
and never wrong.

It may be omitted when `ai-system.type` is `model`, in which case the runtime selects
a built-in provider adapter (A.11). An `external` AI-system always needs one, because
only the runtime knows what that identifier means.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from benchy import data, providers, score
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchy", description="Benchmark AI-systems.")
    verbs = parser.add_subparsers(dest="verb", required=True)

    compile_verb = verbs.add_parser("compile", help="compile benchmark YAML into canonical JSON IR")
    compile_verb.add_argument("source", type=Path, help="benchmark YAML")
    compile_verb.add_argument("-o", "--output", type=Path, help="write IR here instead of stdout")

    run_verb = verbs.add_parser("run", help="evaluate an AI-system against a benchmark")
    run_verb.add_argument("source", type=Path, help="benchmark YAML, or a compiled .json IR")
    run_verb.add_argument(
        "--adapter", metavar="MODULE:ATTR",
        help="the AI-system's adapter; optional when ai-system.type is 'model'",
    )
    run_verb.add_argument("-w", "--workspace", type=Path, help="benchmark workspace root (default: source directory)")
    run_verb.add_argument("-o", "--output", type=Path, help="write the result here instead of stdout")

    args = parser.parse_args(argv)
    try:
        return _compile(args) if args.verb == "compile" else _run(args)
    except BenchyError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1


def _compile(args: argparse.Namespace) -> int:
    return _emit(compile_benchmark(_read(args.source)), args.output)


def _run(args: argparse.Namespace) -> int:
    ir = (
        json.loads(_read(args.source, "runtime"))
        if args.source.suffix == ".json"
        else compile_benchmark(_read(args.source))
    )
    workspace = args.workspace or args.source.parent
    adapter = _load_adapter(args.adapter) if args.adapter else providers.for_system(ir, workspace)
    result = asyncio.run(run(ir, workspace, adapter))
    _warn_if_degenerate(result)
    _warn_if_uninformative_field(result, ir, workspace)
    return _emit(result, args.output)


def _warn_if_degenerate(result: dict) -> None:
    """Say out loud when every example produced the same output.

    An adapter that ignores its input — or reads the wrong input key — returns a constant
    for every example and still reports 100% valid, because each output on its own
    conforms to the schema. A constant predictor is not a measurement of the AI-system.

    This is a warning and not a field in the report on purpose: the report's shape is a
    pinned contract, and a machine reader must not have to learn a new key to keep working.
    It goes to stderr; stdout stays the report alone.
    """
    results = result.get("results") or []
    valid = [item for item in results if item.get("status") == "valid"]
    predictions = {
        json.dumps(item.get("prediction"), sort_keys=True, ensure_ascii=False)
        for item in valid
    }
    if len(valid) > 1 and len(predictions) == 1 and len(valid) == len(results):
        print(
            "warning: degenerate_constant_output — every example produced the same "
            "output: the adapter is not reading its input (or reads the wrong key), so "
            "this run does not measure the AI-system",
            file=sys.stderr,
        )


#: A field whose expected value has more distinct values than this gets its distribution
#: summarized: the diagnostic is meant to be read, not to flood stderr.
_SMALL_ALPHABET = 3


def _seen(value: object) -> str:
    """A value written the way the report writes it, so two equal values compare equal."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _distribution(counts: Counter, rows: int) -> str:
    """`'yes' 30/39, 'no' 9/39`, largest first, with the tail summarized."""
    shown = counts.most_common(4)
    text = ", ".join(f"{value} {number}/{rows}" for value, number in shown)
    remaining = len(counts) - len(shown)
    return f"{text} (+{remaining} more values)" if remaining else text


def _warn_if_uninformative_field(result: dict, ir: Mapping, workspace: Path | str) -> None:
    """One field can be a non-measurement while the whole prediction object still varies.

    The object-level check misses a field whose value never changes across examples: the
    benchmark score is a mean over fields, so an uninformative 1.0 there reads as a perfect
    field, and the reader cannot tell a field that measures the AI-system from a field that
    measures the label distribution of the exam.

    Per field, in this order, one line at most:

    - the exam's own expected value is the same in every row: the field cannot discriminate,
      so any predictor that emits that constant scores 1.0 on it;
    - the prediction is the same in every valid example: a constant is not a measurement at
      field level either;
    - the field is perfect in every valid example over a small alphabet: the score has to be
      read next to the trivial majority baseline, which is printed.

    The exam is streamed a second time -- one row of memory, no adapter call. The
    distributions are bounded by the report, which already holds every prediction. This is a
    warning on stderr and never a key in the report, whose shape is a pinned contract.
    """
    results = result.get("results") or []
    valid = [item for item in results if item.get("status") == "valid"]
    if len(valid) < 2:
        return
    dimensions = [tuple(dimension["path"]) for dimension in ir["scoring"]["dimensions"]]
    counts = {path: Counter() for path in dimensions}
    rows = 0
    for _, _, expected in data.examples(ir, workspace):
        rows += 1
        for path in dimensions:
            counts[path][_seen(score.value_at(expected, path))] += 1
    for path in dimensions:
        name = ".".join(path)
        distribution = _distribution(counts[path], rows)
        predicted = {_seen(score.value_at(item["prediction"], path)) for item in valid}
        if len(counts[path]) == 1:
            print(
                f"warning: degenerate_exam_field — every example's expected value for field "
                f"{name} is {next(iter(counts[path]))} ({distribution}): the field cannot "
                f"discriminate, so a constant predictor scores 1.0 on it",
                file=sys.stderr,
            )
            continue
        if len(predicted) == 1:
            print(
                f"warning: degenerate_constant_field — field {name} is {next(iter(predicted))} "
                f"in every valid example ({len(valid)}/{len(results)}); the exam's expected "
                f"value there: {distribution}. A constant prediction does not measure the field",
                file=sys.stderr,
            )
            continue
        scores = [
            field["score"]
            for item in valid
            for field in (item.get("field_scores") or [])
            if tuple(field["path"]) == path
        ]
        if (len(scores) == len(valid) and all(value == 1 for value in scores)
                and len(counts[path]) <= _SMALL_ALPHABET):
            majority = counts[path].most_common(1)[0][1] / rows
            print(
                f"warning: field_score_baseline — field {name} scores 1 in every valid example "
                f"({len(valid)}/{len(results)}); the exam's expected value there: "
                f"{distribution}, so a majority-only predictor scores {majority:.3f} on this field",
                file=sys.stderr,
            )


def _read(path: Path, phase: str = "compile") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchyError(phase, "data_not_found", f"cannot read {path}: {exc.strerror}") from None


def _emit(document: dict, output: Path | None) -> int:
    text = json.dumps(document, indent=2, ensure_ascii=False, default=str)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _load_adapter(spec: str) -> object:
    """Import `module:attr`, where `module` is a dotted name or a .py file path."""
    module_name, separator, attribute = spec.rpartition(":")
    if not separator or not module_name or not attribute:
        raise BenchyError("runtime", "adapter_not_bound", f"--adapter must be MODULE:ATTR, got {spec!r}")
    source = Path(module_name)
    if source.suffix == ".py":
        if not source.is_file():
            raise BenchyError("runtime", "adapter_not_bound", f"no such adapter module: {source}")
        spec_obj = importlib.util.spec_from_file_location(source.stem, source)
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise BenchyError("runtime", "adapter_not_bound", f"cannot import {module_name!r}: {exc}") from None
    if not hasattr(module, attribute):
        raise BenchyError("runtime", "adapter_not_bound", f"{module_name!r} has no attribute {attribute!r}")
    return getattr(module, attribute)
