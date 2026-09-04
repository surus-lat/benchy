"""Ready-made Task shapes so an author reuses instead of rebuilding.

Each function returns a plain `Task` (see `base.py`), pre-wired for a
common exam shape:

    transcription(...)           audio in, plain text out    (mode="text")
    structured_extraction(...)   the invoice/lead-extraction case (mode="schema")
    classification(...)          a fixed label set, tolerant parsing (mode="choice")
    freeform(...)                bare text in, bare text out (mode="text")

Two design calls worth flagging (both deliberate, not oversights):

- `transcription`'s output schema is the *object* `{"text": <str>}`, so
  `Prediction.value` is always a dict a downstream scorer can key off with
  `["text"]`. `freeform` is bare "text in, text out" -- no object wrapper
  beyond what `Sample.input` already requires (it is a `Mapping`). This
  mirrors the literal wording of the build brief: transcription is written
  with braces (`{text: str}`), freeform is written in prose ("text in,
  text out").
- None of these functions require `input=`/`output=` beyond what's needed
  to do their one job. Anything left unspecified resolves through
  `schema.resolve_schema(None)` to a fully permissive `{"type": "object"}`
  schema, so a Data source can hand samples through with harmless extra
  fields without tripping `SchemaViolation`.

`input=`/`output=` accept a pydantic `BaseModel` subclass or a raw JSON
Schema `dict` everywhere `Task(...)` does -- `structured_extraction` in
particular is exercised with a raw dict in the cross-module seam suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchy.core import OntologyPath

from .base import ParseFn, RenderFn, Task

TRANSCRIPTION_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

TRANSCRIPTION_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"audio": {}, "audio_path": {"type": "string"}},
}

FREEFORM_INPUT: dict[str, Any] = {"type": "object", "properties": {"text": {"type": "string"}}}
FREEFORM_OUTPUT: dict[str, Any] = {"type": "string"}


def _default_name(ontology: str | OntologyPath) -> str:
    return str(ontology if isinstance(ontology, OntologyPath) else OntologyPath.parse(ontology))


def transcription(
    *,
    ontology: str | OntologyPath,
    language: str | None = None,
    name: str | None = None,
    instructions: str | None = None,
    input: Any = None,
    template: str | None = None,
    render_fn: RenderFn | None = None,
    parse_fn: ParseFn | None = None,
) -> Task:
    """Audio in, plain text out. See `Sample.input` shapes in `.media`."""
    default_instructions = "Transcribe the audio exactly as spoken" + (
        f" in {language}." if language else "."
    )
    return Task(
        name=name or _default_name(ontology),
        ontology=ontology,
        input=input if input is not None else TRANSCRIPTION_INPUT,
        output=TRANSCRIPTION_OUTPUT,
        instructions=instructions if instructions is not None else default_instructions,
        mode="text",
        template=template,
        render_fn=render_fn,
        parse_fn=parse_fn,
    )


def structured_extraction(
    output: Any,
    *,
    ontology: str | OntologyPath,
    instructions: str = "",
    input: Any = None,
    name: str | None = None,
    template: str | None = None,
    render_fn: RenderFn | None = None,
    parse_fn: ParseFn | None = None,
) -> Task:
    """The invoice/lead-extraction case: JSON schema in, JSON out."""
    return Task(
        name=name or _default_name(ontology),
        ontology=ontology,
        input=input,
        output=output,
        instructions=instructions,
        mode="schema",
        template=template,
        render_fn=render_fn,
        parse_fn=parse_fn,
    )


def classification(
    labels: Sequence[str],
    *,
    ontology: str | OntologyPath,
    instructions: str = "",
    input: Any = None,
    name: str | None = None,
    template: str | None = None,
    render_fn: RenderFn | None = None,
    parse_fn: ParseFn | None = None,
) -> Task:
    """Output constrained to a fixed label set, with tolerant parsing."""
    label_list = list(labels)
    return Task(
        name=name or _default_name(ontology),
        ontology=ontology,
        input=input,
        output={"type": "string", "enum": label_list},
        instructions=instructions or "Classify the input.",
        mode="choice",
        labels=label_list,
        template=template,
        render_fn=render_fn,
        parse_fn=parse_fn,
    )


def freeform(
    *,
    ontology: str | OntologyPath,
    instructions: str = "",
    input: Any = None,
    output: Any = None,
    name: str | None = None,
    template: str | None = None,
    render_fn: RenderFn | None = None,
    parse_fn: ParseFn | None = None,
) -> Task:
    """Text in, text out -- no schema, no object wrapper, no ceremony."""
    return Task(
        name=name or _default_name(ontology),
        ontology=ontology,
        input=input if input is not None else FREEFORM_INPUT,
        output=output if output is not None else FREEFORM_OUTPUT,
        instructions=instructions,
        mode="text",
        template=template,
        render_fn=render_fn,
        parse_fn=parse_fn,
    )
