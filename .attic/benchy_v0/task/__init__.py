"""benchy.task -- the exam contract: what shape does a solution have.

Owns the Task<->System bridge (`render`/`parse`) described in
`docs/superpowers/plans/2026-09-01-new-benchy-build.md`, section "There is
a Task<->System bridge, and the Task owns it", and the ontology registry
that resolves `/<task>/<domain>/<language>` lookups (`benchy.core.OntologyPath`).

Public surface -- siblings import exactly these names::

    from benchy.task import Task, load, register, registry, builtin

    task = Task(
        name="invoice_extraction",
        ontology="image_extraction/invoices/es-AR",
        input=InvoiceInput,        # pydantic BaseModel | dict JSON Schema
        output=InvoiceExtraction,  # pydantic BaseModel | dict JSON Schema
        instructions="Extract the fields from this Argentine invoice.",
    )
    task.input_schema      # -> dict, JSON Schema
    task.output_schema     # -> dict, JSON Schema
    task.render(sample, caps)   # -> Request
    task.parse(response, caps)  # -> Prediction
    task.validate_sample(sample)  # raises SchemaViolation

    register(task)
    load("image_extraction/invoices/es-AR")  # -> Task

See `base.py` for the render/parse negotiation, `media.py` for the audio /
image `Sample.input` shapes, `repair.py` for the JSON-from-prose
heuristics, `choice.py` for tolerant label parsing, and `builtin.py` for
the ready-made task shapes (`transcription`, `structured_extraction`,
`classification`, `freeform`).
"""

from __future__ import annotations

from . import builtin
from .base import Task
from .registry import TaskRegistry, load, register, registry

__all__ = ["Task", "TaskRegistry", "builtin", "load", "register", "registry"]
