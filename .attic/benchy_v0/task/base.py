"""The exam contract: what shape does a solution have.

`Task` is the bridge between an opaque `benchy.core.System` and a typed
Sample/Prediction pair -- the seam described in the build plan's "There is
a Task<->System bridge, and the Task owns it":

    request    = task.render(sample, system.capabilities)   # Sample  -> Request
    response   = await system.invoke(request)                # opaque
    prediction = task.parse(response, system.capabilities)   # Response -> Prediction

Design decisions worth knowing before reading the code:

- ``input=``/``output=`` accept either a pydantic v2 ``BaseModel`` subclass
  (the authoring surface) or a raw JSON Schema ``dict`` (the exchange
  format every sibling module and the cross-module seam tests actually
  pass around) -- both are normalized by ``benchy.task.schema.resolve_schema``.
  ``None`` resolves to a fully permissive ``{"type": "object"}`` schema.

- Three internal rendering *modes* (``self.mode``), because "structured
  JSON in, JSON out" is not the only exam shape the build plan asks for:

    "schema"  Full structured negotiation: native ``output_schema`` when
              ``caps.structured_output``, else the schema is rendered into
              the prompt and the response is repaired/parsed as JSON.
              ``Task(...)`` used directly, and ``builtin.structured_extraction``.

    "text"    No JSON at all -- the response's raw text *is* the answer
              (or is wrapped into a single-field object, see below).
              ``builtin.freeform`` and ``builtin.transcription``.

    "choice"  Tolerant label resolution against a fixed label set, with a
              lettered-options block in the prompt. ``builtin.classification``.

  A hand-authored ``Task(...)`` defaults to "schema" -- that is the
  headline authoring surface the vision sells (structured extraction).
  The other two modes are ordinarily reached through ``builtin.*``, but
  nothing stops an author passing ``mode=`` directly.

- Reserved input keys ``audio``/``audio_path`` and ``image``/``image_path``
  (see ``benchy.task.media``) are pulled out of ``Sample.input`` and
  rendered as ``AudioPart``/``ImagePart`` instead of prompt text, then
  checked against ``Capabilities``: a missing *capability* raises
  ``CapabilityError``; a present key with an unusable *value* raises
  ``SchemaViolation``.

- ``Request.meta["sample_id"]`` is always exactly ``sample.id`` -- this is
  a hard contract from the integration seam suite (the engine correlates
  records on it, and the `echo:` system test double keys per-sample
  failure injection off of it) -- ``render()`` guarantees it even when a
  custom ``render_fn`` forgets to set it.

- ``parse()`` never raises. Every code path -- a system error, an empty
  response, unparsable prose, a schema-shape mismatch -- returns a
  ``Prediction`` with ``parse_ok=False`` and a human-readable
  ``parse_error`` instead. A benchmark run must survive one bad response.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import jsonschema
from pydantic import ValidationError

from benchy.core import (
    Capabilities,
    Message,
    OntologyPath,
    Prediction,
    Request,
    Response,
    Sample,
    SchemaViolation,
    TextPart,
)

from . import media
from . import render as render_mod
from .choice import parse_label
from .repair import extract_json
from .schema import resolve_schema

Mode = Literal["schema", "text", "choice"]
RenderFn = Callable[[Sample, Capabilities, "Task"], Request]
ParseFn = Callable[[Response, Capabilities, "Task"], Prediction]


class Task:
    """See module docstring. Satisfies `benchy.core.Task` structurally."""

    def __init__(
        self,
        *,
        name: str,
        ontology: str | OntologyPath,
        input: Any = None,
        output: Any = None,
        instructions: str = "",
        mode: Mode = "schema",
        labels: Sequence[str] | None = None,
        template: str | None = None,
        render_fn: RenderFn | None = None,
        parse_fn: ParseFn | None = None,
    ) -> None:
        self.name = name
        self.ontology = ontology if isinstance(ontology, OntologyPath) else OntologyPath.parse(ontology)
        self.instructions = instructions
        self.mode: Mode = mode
        self.labels: list[str] | None = list(labels) if labels is not None else None
        self.template = template
        self.render_fn = render_fn
        self.parse_fn = parse_fn

        self.input_schema, self._input_model = resolve_schema(input)
        self.output_schema, self._output_model = resolve_schema(output)

        if mode == "choice" and not self.labels:
            raise ValueError("mode='choice' requires a non-empty `labels` sequence")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Task(name={self.name!r}, ontology={str(self.ontology)!r}, mode={self.mode!r})"

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def validate_sample(self, sample: Sample) -> None:
        """Raise `SchemaViolation` if `sample.input` doesn't fit this Task.

        Reserved media keys (see `benchy.task.media`) carry values -
        filesystem paths, numpy arrays - that are not themselves valid
        JSON Schema instances, so they are checked only for *presence*
        (via `required`) and excluded from the structural JSON Schema
        validation applied to everything else.
        """
        data = dict(sample.input)
        schema = self.input_schema
        if not isinstance(schema, Mapping):
            return

        media_keys = set(media.AUDIO_KEYS) | set(media.IMAGE_KEYS)
        required = list(schema.get("required", []))
        missing = [key for key in required if key not in data]
        if missing:
            raise SchemaViolation(
                f"task {self.name!r}: sample {sample.id!r} is missing required "
                f"input field(s) {missing!r}"
            )

        properties = dict(schema.get("properties", {}))
        pruned_schema = {
            **schema,
            "properties": {k: v for k, v in properties.items() if k not in media_keys},
            "required": [r for r in required if r not in media_keys],
        }
        projection = {k: v for k, v in data.items() if k not in media_keys}
        try:
            jsonschema.validate(instance=projection, schema=pruned_schema)
        except jsonschema.exceptions.ValidationError as exc:
            raise SchemaViolation(
                f"task {self.name!r}: sample {sample.id!r} failed input schema "
                f"validation: {exc.message}"
            ) from exc

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render(self, sample: Sample, caps: Capabilities) -> Request:
        request = (
            self.render_fn(sample, caps, self)
            if self.render_fn is not None
            else self._default_render(sample, caps)
        )
        # Hard contract (integration seam suite): the engine and the
        # `echo:` test double correlate on this. Guaranteed here so a
        # custom render_fn cannot silently break it.
        if request.meta.get("sample_id") != sample.id:
            request = dataclasses.replace(request, meta={**request.meta, "sample_id": sample.id})
        return request

    def _default_render(self, sample: Sample, caps: Capabilities) -> Request:
        text_fields, parts = media.extract_media(dict(sample.input))
        for part in parts:
            if not caps.accepts(part):
                raise media.capability_error(self.name, part, caps)

        body = render_mod.render_input_body(text_fields)
        blocks = [self.instructions, body]
        output_schema = None

        if self.mode == "schema":
            if caps.structured_output:
                output_schema = self.output_schema
            else:
                blocks.append(render_mod.render_schema_block(self.output_schema))
        elif self.mode == "choice":
            if caps.structured_output:
                output_schema = self.output_schema
            else:
                blocks.append(render_mod.render_choice_block(self.labels or []))
        # "text" mode: no schema, no extra block - the raw answer is the point.

        if self.template is not None:
            text = self.template.format(
                instructions=self.instructions,
                input=body,
                schema=self.output_schema,
            )
        else:
            text = render_mod.compose(*blocks)

        message_parts: list = []
        if text:
            message_parts.append(TextPart(text))
        message_parts.extend(parts)
        if not message_parts:
            message_parts.append(TextPart(""))

        return Request(
            messages=(Message(role="user", parts=tuple(message_parts)),),
            output_schema=output_schema,
            meta={"sample_id": sample.id, "task": self.name},
        )

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------

    def parse(self, response: Response, caps: Capabilities) -> Prediction:
        if self.parse_fn is not None:
            return self.parse_fn(response, caps, self)
        if response.error is not None:
            return Prediction(value=None, raw_text=response.text, parse_ok=False, parse_error=response.error)
        if self.mode == "text":
            return self._parse_text(response)
        if self.mode == "choice":
            return self._parse_choice(response, caps)
        return self._parse_schema(response, caps)

    def _parse_text(self, response: Response) -> Prediction:
        text = response.text
        if text is None and response.data is not None:
            text = str(response.data)
        if text is None or not text.strip():
            return Prediction(value=None, raw_text=text, parse_ok=False, parse_error="empty response")
        text = text.strip()
        value: Any = text
        if isinstance(self.output_schema, Mapping) and self.output_schema.get("type") == "object":
            props = list(self.output_schema.get("properties", {}))
            if len(props) == 1:
                value = {props[0]: text}
        return Prediction(value=value, raw_text=text, parse_ok=True)

    def _parse_choice(self, response: Response, caps: Capabilities) -> Prediction:
        raw = response.data if (caps.structured_output and response.data is not None) else response.text
        label, error = parse_label(raw, self.labels or [])
        if error is not None:
            return Prediction(value=None, raw_text=response.text, parse_ok=False, parse_error=error)
        return Prediction(value=label, raw_text=response.text, parse_ok=True)

    def _parse_schema(self, response: Response, caps: Capabilities) -> Prediction:
        if caps.structured_output and response.data is not None:
            data: Any = response.data
            error = None
        else:
            data, error = extract_json(response.text)
        if error is not None:
            return Prediction(value=None, raw_text=response.text, parse_ok=False, parse_error=error)

        data = self._reconcile_shape(data)

        if self._output_model is not None:
            try:
                instance = self._output_model.model_validate(data)
            except ValidationError as exc:
                return Prediction(
                    value=data,
                    raw_text=response.text,
                    parse_ok=False,
                    parse_error=f"output failed schema validation: {exc}",
                )
            return Prediction(value=instance.model_dump(mode="json"), raw_text=response.text, parse_ok=True)

        try:
            jsonschema.validate(instance=data, schema=self.output_schema)
        except Exception as exc:  # noqa: BLE001 - includes non-ValidationError shape mismatches
            return Prediction(
                value=data,
                raw_text=response.text,
                parse_ok=False,
                parse_error=f"output failed schema validation: {exc}",
            )
        return Prediction(value=data, raw_text=response.text, parse_ok=True)

    def _reconcile_shape(self, data: Any) -> Any:
        """Unwrap a singleton array when an object was asked for.

        Models sometimes wrap a single structured answer in a list even
        when told the shape is an object. Unwrapping here means a
        one-element ``[{"name": "Ana"}]`` doesn't fail validation over a
        formatting tic that has nothing to do with the actual content.
        """
        schema = self.output_schema if isinstance(self.output_schema, Mapping) else {}
        wants_object = schema.get("type") == "object" or "properties" in schema
        if wants_object and isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        return data
