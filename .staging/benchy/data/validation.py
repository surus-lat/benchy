"""``Data.validate(schema)`` — report, don't raise.

Checks every sample's ``input`` against a JSON Schema (the Task's
``input_schema``). This is a diagnostic pass an author runs explicitly; it
is unrelated to the warn-and-skip filtering ``Data.__iter__``/the mapping
layer already does for structurally malformed rows (missing columns,
invalid JSON lines) — this is about samples that parsed fine but don't
satisfy the *shape* a Task expects. Per the build plan, only
``Benchmark.run`` (a sibling module) fails loud; this only reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchy.data.dataset import Data


@dataclass(frozen=True, slots=True)
class ValidationReport:
    n_total: int
    n_valid: int
    n_invalid: int
    errors: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def validate_data(data: "Data", schema: dict[str, Any]) -> ValidationReport:
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema)
    n_total = 0
    n_invalid = 0
    errors: list[tuple[str, str]] = []
    for sample in data:
        n_total += 1
        sample_errors = sorted(validator.iter_errors(sample.input), key=lambda e: e.path)
        if sample_errors:
            n_invalid += 1
            message = "; ".join(e.message for e in sample_errors)
            errors.append((sample.id, message))
    return ValidationReport(
        n_total=n_total,
        n_valid=n_total - n_invalid,
        n_invalid=n_invalid,
        errors=tuple(errors),
    )
