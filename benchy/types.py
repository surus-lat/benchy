"""The semantic type system: schema compilation, value validation, equality.

Paper §2 defines a schema as *field structure + semantic types*. Appendices A.1
(grammar), A.2 (strict object validation), A.6 (JSON representation) and A.7
(exact-match equality) are four tables keyed by the same type vocabulary, so they
live in one module: adding a semantic type means editing one file.

**There is exactly one representation of a schema anywhere in benchy: the IR JSON
node.** No `SchemaNode` class hierarchy exists, and nothing marshals between an
internal form and the IR. `compile_schema` reads a parsed YAML node and emits the
IR node directly; `validate`, `leaves` and `equal` walk that same IR node. The
engine therefore executes from persisted IR without a conversion step, which is
what makes paper A.8's invariant ("never reinterprets source YAML") structural
rather than a rule someone has to remember.

IR node shapes (paper A.9 / spec §17):

    {"type": "string"}                                    primitive
    {"type": "enum", "values": ["a", "b"]}                closed categorical
    {"type": "object", "fields": {"name": <node>, ...}}   nested structure
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from benchy.errors import BenchyError

__all__ = [
    "PRIMITIVES", "ARTIFACTS",
    "compile_schema", "validate", "leaves", "at", "equal",
]

#: Paper §2 type vocabulary. `enum` is excluded: it is a declaration form
#: (`{enum: [...]}`), not a bare token, so it can never appear as a type name.
PRIMITIVES: frozenset[str] = frozenset(
    {"string", "int", "float", "bool", "date", "time", "datetime", "image", "audio", "document"}
)

#: Types whose runtime representation is a filesystem path (paper A.6, Appendix C).
ARTIFACTS: frozenset[str] = frozenset({"image", "audio", "document"})

# Canonical lexical forms (paper A.6). `fromisoformat` alone is too permissive in
# 3.11+ (it accepts "20260913"), so each form is gated by its pattern first.
_DATE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")
_TIME_RE = re.compile(r"\A\d{2}:\d{2}:\d{2}(\.\d+)?\Z")
_DATETIME_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})\Z")


def _parse_date(v: str) -> _dt.date:
    if not _DATE_RE.match(v):
        raise ValueError("expected YYYY-MM-DD")
    return _dt.date.fromisoformat(v)


def _parse_time(v: str) -> _dt.time:
    if not _TIME_RE.match(v):
        raise ValueError("expected HH:MM:SS[.fraction]")
    return _dt.time.fromisoformat(v)


def _parse_datetime(v: str) -> _dt.datetime:
    if not _DATETIME_RE.match(v):
        raise ValueError("expected RFC 3339 with timezone")
    parsed = _dt.datetime.fromisoformat(v.replace("Z", "+00:00").replace("z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("expected RFC 3339 with timezone")
    return parsed


#: Shared by `validate` (does it parse?) and `equal` (do the parsed values match?),
#: so a temporal type is defined in exactly one place.
_PARSE: dict[str, Callable[[str], object]] = {
    "date": _parse_date,
    "time": _parse_time,
    "datetime": _parse_datetime,
}


# ---------------------------------------------------------------------------
# compile: parsed YAML node -> IR node
# ---------------------------------------------------------------------------

def compile_schema(node: object, *, path: tuple[str, ...] = ()) -> dict:
    """Compile one parsed YAML schema node into its IR node.

    Enforces paper A.1: fields are required, `null` is rejected, lists are
    rejected everywhere, objects are non-empty, and a leaf is either a primitive
    token or an `{enum: [...]}` declaration.
    """
    if isinstance(node, str):
        if node not in PRIMITIVES:
            raise BenchyError(
                "compile", "invalid_schema",
                f"unknown semantic type {node!r}; expected one of {', '.join(sorted(PRIMITIVES))}",
                list(path),
            )
        return {"type": node}

    if isinstance(node, Mapping):
        # A mapping whose key set is exactly {"enum"} is the enum declaration;
        # any other mapping is an object schema (paper A.1).
        if set(node) == {"enum"}:
            return _compile_enum(node["enum"], path)
        if not node:
            raise BenchyError("compile", "invalid_schema", "object schema must declare at least one field", list(path))
        fields: dict[str, dict] = {}
        for name, child in node.items():
            if not isinstance(name, str) or not name:
                raise BenchyError("compile", "invalid_schema", f"field name must be a non-empty string, got {name!r}", list(path))
            fields[name] = compile_schema(child, path=path + (name,))
        return {"type": "object", "fields": fields}

    if node is None:
        raise BenchyError("compile", "invalid_schema", "field value must not be null; every field is required", list(path))
    if isinstance(node, (list, tuple)):
        raise BenchyError("compile", "invalid_schema", "lists are not valid schema nodes", list(path))
    raise BenchyError("compile", "invalid_schema", f"invalid schema node of type {type(node).__name__}", list(path))


def _compile_enum(values: object, path: tuple[str, ...]) -> dict:
    if not isinstance(values, list):
        raise BenchyError("compile", "invalid_schema", "enum must declare a list of values", list(path))
    if not values:
        raise BenchyError("compile", "invalid_schema", "enum must declare at least one value", list(path))
    for v in values:
        if not isinstance(v, str) or not v:
            raise BenchyError("compile", "invalid_schema", f"enum values must be non-empty strings, got {v!r}", list(path))
    if len(set(values)) != len(values):
        raise BenchyError("compile", "invalid_schema", "enum values must be distinct", list(path))
    return {"type": "enum", "values": list(values)}


# ---------------------------------------------------------------------------
# leaves: IR node -> scoring dimensions
# ---------------------------------------------------------------------------

def leaves(node: Mapping) -> list[list[str]]:
    """Every leaf path in depth-first source order (paper §2, handoff §4).

    The output schema's leaves *are* the scoring dimensions, so this ordering is
    the ordering of `scoring.dimensions` in the IR and of `field_scores` in a
    result. Dicts preserve insertion order, so source order is preserved.
    """
    if node["type"] != "object":
        return [[]]
    out: list[list[str]] = []
    for name, child in node["fields"].items():
        out.extend([name, *rest] for rest in leaves(child))
    return out


def at(node: Mapping, path: list[str] | tuple[str, ...]) -> Mapping:
    """The schema node at `path`, for callers that hold a dimension path."""
    for key in path:
        node = node["fields"][key]
    return node


# ---------------------------------------------------------------------------
# validate: runtime value against IR node
# ---------------------------------------------------------------------------

def validate(
    value: object,
    node: Mapping,
    *,
    phase: str,
    path: tuple[str, ...] = (),
    resolve: Callable[[str, tuple[str, ...]], str] | None = None,
) -> object:
    """Strictly validate `value` against `node`, returning the value to store.

    Strict means closed (paper A.2): a missing field, an extra field, or a wrong
    semantic type is invalid. Nested objects recurse.

    `resolve` is the artifact hook. When given, every `image`/`audio`/`document`
    leaf is passed through it before the file check, and the returned path is what
    this function returns in that slot — so resolving dataset-relative references
    into workspace-confined absolute paths is the *same single walk* as validating
    them, rather than a second traversal.
    """
    t = node["type"]

    if t == "object":
        if not isinstance(value, Mapping):
            raise BenchyError(phase, "wrong_type", f"expected an object, got {_name(value)}", list(path))
        out = {}
        for name, child in node["fields"].items():
            if name not in value:
                raise BenchyError(phase, "missing_field", "required field is missing", list(path + (name,)))
            out[name] = validate(value[name], child, phase=phase, path=path + (name,), resolve=resolve)
        for extra in sorted(set(value) - set(node["fields"])):
            raise BenchyError(phase, "extra_field", "field is not declared in the schema", list(path + (extra,)))
        return out

    if t == "enum":
        if not isinstance(value, str):
            raise BenchyError(phase, "wrong_type", f"expected a string enum member, got {_name(value)}", list(path))
        if value not in node["values"]:
            raise BenchyError(phase, "invalid_enum", f"{value!r} is not one of {node['values']}", list(path))
        return value

    if t == "string":
        if not isinstance(value, str):
            raise BenchyError(phase, "wrong_type", f"expected a string, got {_name(value)}", list(path))
        return value

    if t == "bool":
        if not isinstance(value, bool):
            raise BenchyError(phase, "wrong_type", f"expected a boolean, got {_name(value)}", list(path))
        return value

    if t == "int":
        # bool is a subclass of int; paper A.6 forbids accepting it as one.
        if not isinstance(value, int) or isinstance(value, bool):
            raise BenchyError(phase, "wrong_type", f"expected an integer, got {_name(value)}", list(path))
        return value

    if t == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise BenchyError(phase, "wrong_type", f"expected a number, got {_name(value)}", list(path))
        if not math.isfinite(value):
            raise BenchyError(phase, "invalid_value", "float must be finite", list(path))
        return value

    if t in _PARSE:
        if not isinstance(value, str):
            raise BenchyError(phase, "wrong_type", f"expected a {t} string, got {_name(value)}", list(path))
        try:
            _PARSE[t](value)
        except ValueError as exc:
            raise BenchyError(phase, "invalid_value", f"invalid {t}: {exc}", list(path)) from None
        return value

    if t in ARTIFACTS:
        if not isinstance(value, str):
            raise BenchyError(phase, "wrong_type", f"expected a {t} path string, got {_name(value)}", list(path))
        resolved = resolve(value, path) if resolve else value
        p = Path(resolved)
        if not p.is_file() or not os.access(p, os.R_OK):
            raise BenchyError(phase, "artifact_not_found", f"{t} is not an existing readable file: {resolved}", list(path))
        return resolved

    raise BenchyError(phase, "invalid_ir", f"unknown schema type {t!r}", list(path))


def _name(value: object) -> str:
    return "null" if value is None else type(value).__name__


# ---------------------------------------------------------------------------
# equal: exact match (paper A.7)
# ---------------------------------------------------------------------------

def equal(a: object, b: object, node: Mapping) -> bool:
    """Exact-match equality of two already-validated values.

    No trimming, case folding, Unicode normalization, numeric tolerance, or
    semantic similarity (paper A.7). Temporal types compare *parsed* values, so
    `12:00:00` equals `12:00:00.000` and two RFC 3339 strings in different offsets
    denoting the same instant compare equal. Artifacts compare file content.
    """
    t = node["type"]
    if t == "object":
        return all(equal(a[k], b[k], child) for k, child in node["fields"].items())
    if t in _PARSE:
        return _PARSE[t](a) == _PARSE[t](b)
    if t in ARTIFACTS:
        return _same_file(str(a), str(b))
    if t == "bool":
        return a is b
    # string, enum, int and float are all plain value equality; bool was excluded at
    # validation, so no `True == 1` confusion can reach here.
    return a == b


def _same_file(a: str, b: str) -> bool:
    """Byte-for-byte content equality, without loading either file (handoff §9)."""
    pa, pb = Path(a), Path(b)
    if pa == pb:
        return True
    if pa.stat().st_size != pb.stat().st_size:
        return False
    with pa.open("rb") as fa, pb.open("rb") as fb:
        return hashlib.file_digest(fa, "sha256").digest() == hashlib.file_digest(fb, "sha256").digest()
