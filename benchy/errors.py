"""The one diagnostic shape, per spec v1.1 §18.

Every rejection in benchy — a malformed YAML document, a dataset row that does
not match the compiled schema, a run that has no adapter bound — is reported as
a `BenchyError` carrying the same four fields:

    phase    "compile" | "dataset" | "runtime"
    code     a stable machine-readable identifier
    path     the semantic key path it applies to, or None
    message  human wording; explicitly *not* part of benchmark semantics

There are deliberately no subclasses. `phase` already discriminates, and the one
place that needs to tell failures apart — the run loop — distinguishes "benchy
rejected this" (`BenchyError`, abort) from "the AI-system blew up" (any other
exception, `execution_error`). A class hierarchy would add three parts to express
what one field already says.

The code vocabulary (paper A.16), listed here rather than as a frozenset so
there is no second place to edit when a code is added:

    compile   invalid_yaml duplicate_key unknown_key missing_key
              unsupported_version unsupported_ontology_version unknown_task
              unknown_domain unknown_language task_program_mismatch
              invalid_schema missing_weight extra_weight invalid_weight
              invalid_ai_system invalid_value
    dataset   data_not_found invalid_dataset_record empty_dataset path_escape
              artifact_not_found missing_field extra_field wrong_type
              invalid_enum invalid_value
    runtime   adapter_not_bound adapter_error timeout invalid_ir
              missing_field extra_field wrong_type invalid_enum invalid_value

`missing_key` and `path_escape` extend paper A.16, whose list is explicitly
"representative": the first is the counterpart of `unknown_key` for a required key,
the second distinguishes a workspace escape (spec §10) from a merely absent file.
"""

from __future__ import annotations

__all__ = ["BenchyError"]


class BenchyError(Exception):
    """A structured, serializable benchy diagnostic."""

    def __init__(
        self,
        phase: str,
        code: str,
        message: str,
        path: list[str] | None = None,
    ) -> None:
        self.phase = phase
        self.code = code
        self.message = message
        self.path = list(path) if path else None
        super().__init__(f"[{phase}:{code}] {self._where()}{message}")

    def _where(self) -> str:
        return f"{'.'.join(self.path)}: " if self.path else ""

    def to_dict(self) -> dict[str, object]:
        """The `error` object as it appears in a result artifact (spec §16/§18)."""
        return {
            "phase": self.phase,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
