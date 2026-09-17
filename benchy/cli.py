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
from pathlib import Path

from benchy import providers
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
    return _emit(result, args.output)


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
