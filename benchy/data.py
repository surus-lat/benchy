"""The exam: streaming JSONL, validated against the compiled schemas.

The dataset is *the exam* (paper §5): each row is a pair `(x_i, y_i*)`. This module
streams those rows, validates both sides against the IR's schemas, and resolves
artifact references — and does all three in a single pass over each row, because
`types.validate` takes the artifact resolver as a hook.

**Dataset failures abort the run** (paper A.15). They are benchmark errors, not
AI-system failures: a malformed row says nothing about the system under test, so
scoring it as a zero would corrupt the measurement. Every diagnostic here carries
`phase="dataset"`.

Two path bases, per amendment §2 — they are genuinely different and conflating them
is an easy, silent bug:

    data.path        -> resolved from the benchmark workspace root
    artifact paths   -> resolved from the JSONL file's own directory

Both must canonicalize to somewhere inside the workspace. `Path.resolve()` follows
symlinks, so comparing the real path against the real workspace root rejects
traversal and symlink escape with the same check.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from benchy import types
from benchy.errors import BenchyError

__all__ = ["examples", "resolve_within"]


def resolve_within(
    reference: str,
    base: Path,
    workspace: Path,
    path: list[str] | None = None,
    *,
    phase: str = "dataset",
) -> Path:
    """Resolve `reference` against `base` and require the result to stay in `workspace`.

    Absolute references replace `base` on join, and `resolve()` collapses `..` and
    follows symlinks, so one containment check covers traversal and link escape.
    Shared with the run loop, which confines adapter artifact *outputs* the same way.
    """
    resolved = (base / reference).resolve()
    if not resolved.is_relative_to(workspace):
        raise BenchyError(
            phase, "path_escape",
            f"{reference!r} resolves to {resolved}, outside the benchmark workspace {workspace}",
            path,
        )
    return resolved


def examples(ir: Mapping, workspace: Path | str) -> Iterator[tuple[int, dict, dict]]:
    """Yield `(index, input, expected)` for each exam example, in dataset order.

    A generator: rows are streamed, never preloaded, so a large exam costs one row
    of memory and a malformed row fails only when reached.
    """
    root = Path(workspace).resolve()
    dataset = resolve_within(ir["data"]["path"], root, root, ["data", "path"])
    if not dataset.is_file():
        raise BenchyError("dataset", "data_not_found", f"dataset is not a readable file: {dataset}", ["data", "path"])

    input_schema = ir["program"]["input"]
    output_schema = ir["program"]["output"]
    here = dataset.parent

    def resolve(reference: str, field: tuple[str, ...]) -> str:
        return str(resolve_within(reference, here, root, list(field)))

    index = 0
    with dataset.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = _row(line, lineno, dataset)
            yield (
                index,
                types.validate(row["input"], input_schema, phase="dataset", resolve=resolve),
                types.validate(row["expected"], output_schema, phase="dataset", resolve=resolve),
            )
            index += 1

    if index == 0:
        raise BenchyError("dataset", "empty_dataset", f"dataset contains no examples: {dataset}", ["data", "path"])


def _row(line: str, lineno: int, dataset: Path) -> Mapping:
    def reject(message: str) -> BenchyError:
        return BenchyError("dataset", "invalid_dataset_record", f"{dataset.name} line {lineno}: {message}")

    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise reject(f"malformed JSON ({exc.msg})") from None
    if not isinstance(row, Mapping):
        raise reject(f"expected an object, got {type(row).__name__}")
    if set(row) != {"input", "expected"}:
        raise reject(f"expected exactly 'input' and 'expected', got {', '.join(sorted(row)) or '<empty>'}")
    return row
