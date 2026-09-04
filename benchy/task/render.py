"""Prompt rendering: instructions + input body + (optional) schema contract.

Kept as pure string assembly, deliberately separate from `Task` so an
author's `template=`/`render_fn=` override has an obvious seam to replace,
and so it can be unit-tested without constructing a whole Task.

The rendered prompt is meant to read like an instruction to a competent
model, not a data dump: a single input field renders as its bare value (so
a freeform question still reads like a question), multiple fields render
as short labeled lines, and the JSON Schema contract -- only emitted when
the system lacks native structured output -- is pretty-printed with an
explicit "reply with JSON only" instruction bracketing it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def render_input_body(text_fields: Mapping[str, Any]) -> str:
    if not text_fields:
        return ""
    if len(text_fields) == 1:
        return str(next(iter(text_fields.values())))
    return "\n".join(f"{key}: {value}" for key, value in text_fields.items())


def render_schema_block(schema: Mapping[str, Any]) -> str:
    """A compact schema rendering plus an explicit JSON-only contract."""
    compact = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "Respond with a single JSON object matching exactly this schema "
        "(no extra commentary, no markdown fences):\n"
        f"{compact}\n\n"
        "Reply with JSON only."
    )


def render_choice_block(labels: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(labels))]
    lines = "\n".join(f"{letter}. {label}" for letter, label in zip(letters, labels))
    return (
        f"Choose exactly one of the following options:\n{lines}\n\n"
        "Reply with only the corresponding letter."
    )


def compose(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block)
