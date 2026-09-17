"""Pydantic <-> JSON Schema conversion.

Pydantic v2 is the authoring surface; JSON Schema is the exchange format
carried on ``Task.input_schema`` / ``output_schema`` and on
``Request.output_schema``. ``benchy.core`` stays pydantic-free and so do
those attributes -- they are plain dicts by the time anything outside this
module sees them.

Callers may also hand us a raw JSON Schema ``dict`` directly with no
pydantic involved at all. Both are treated identically once resolved: the
dict is what every sibling module and the cross-module seam tests actually
exchange, so a bare mapping must work everywhere a pydantic model works.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def is_model(obj: Any) -> bool:
    """True if `obj` is a pydantic BaseModel *class* (not an instance)."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def resolve_schema(spec: Any) -> tuple[dict[str, Any], type[BaseModel] | None]:
    """Normalize a Task's ``input=``/``output=`` argument.

    Returns ``(json_schema, pydantic_model_or_None)``. `spec` may be:

    - a pydantic ``BaseModel`` subclass -> ``(model.model_json_schema(), model)``
    - a mapping (already a JSON Schema dict) -> ``(dict(spec), None)``
    - ``None`` -> a fully permissive ``{"type": "object"}`` schema, so a
      Task that doesn't care about its input/output shape (e.g. a
      builtin that only needs one field) doesn't reject samples that
      merely carry harmless extra keys.
    """
    if spec is None:
        return {"type": "object"}, None
    if is_model(spec):
        return spec.model_json_schema(), spec
    if isinstance(spec, Mapping):
        return dict(spec), None
    raise TypeError(
        "input/output must be a pydantic BaseModel subclass or a JSON Schema "
        f"mapping, got {type(spec)!r}"
    )
