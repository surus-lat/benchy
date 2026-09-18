"""YAML -> validate -> canonical JSON IR. The whole of paper §7.

This module is the *only* place that reads benchmark YAML. Everything downstream
consumes the IR it emits, which is what makes paper A.8's invariant ("the engine
never reinterprets source YAML") structural: there is no other code path from
source to execution.

The strict loader lives here rather than in a `parser` module because its only
consumer is `compile_benchmark`, and "YAML -> IR" is one transformation.

Validation order is fixed by handoff §15 so diagnostics are stable regardless of
how many things are wrong at once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import yaml

from benchy import ontology, types
from benchy.errors import BenchyError

__all__ = ["parse", "compile_scoring", "compile_benchmark", "SPEC_VERSION", "ROOT_KEYS"]

SPEC_VERSION = "1.0"

#: Spec §1: the root mapping contains exactly these keys. No more, no fewer.
ROOT_KEYS = ("version", "ontology_version", "benchmark", "program", "scoring", "data", "ai-system")

_MERGE_TAG = "tag:yaml.org,2002:merge"


# ---------------------------------------------------------------------------
# strict parse (spec §1)
# ---------------------------------------------------------------------------

class _StrictLoader(yaml.SafeLoader):
    """SafeLoader minus the features that create two ways to say one thing.

    `SafeLoader` already rejects custom and language-specific tags, so only
    anchors, aliases, merge keys and duplicate keys need adding.
    """

    def compose_node(self, parent, index):  # type: ignore[no-untyped-def]
        if self.check_event(yaml.events.AliasEvent):
            raise BenchyError("compile", "invalid_yaml", _at("YAML aliases are not supported", self.peek_event()))
        if getattr(self.peek_event(), "anchor", None) is not None:
            raise BenchyError("compile", "invalid_yaml", _at("YAML anchors are not supported", self.peek_event()))
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                raise BenchyError("compile", "invalid_yaml", _at("YAML merge keys are not supported", key_node))
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                raise BenchyError("compile", "invalid_yaml", _at(f"invalid mapping key {key!r}", key_node)) from None
            if duplicate:
                raise BenchyError("compile", "duplicate_key", _at(f"duplicate mapping key {key!r}", key_node))
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _at(message: str, marked: object) -> str:
    mark = getattr(marked, "start_mark", None)
    return f"{message} (line {mark.line + 1}, column {mark.column + 1})" if mark else message


def parse(text: str) -> dict:
    """Parse benchmark YAML strictly into a plain mapping.

    Syntax only — no key-set or semantic validation, which belong to
    `compile_benchmark`.
    """
    try:
        doc = yaml.load(text, Loader=_StrictLoader)
    except yaml.MarkedYAMLError as exc:
        message = str(exc.problem or exc)
        if exc.problem_mark is not None:
            message += f" (line {exc.problem_mark.line + 1}, column {exc.problem_mark.column + 1})"
        raise BenchyError("compile", "invalid_yaml", message) from None
    except yaml.YAMLError as exc:
        raise BenchyError("compile", "invalid_yaml", str(exc)) from None
    if doc is None:
        raise BenchyError("compile", "invalid_yaml", "benchmark document is empty")
    if not isinstance(doc, dict):
        raise BenchyError("compile", "invalid_yaml", f"benchmark root must be a mapping, got {type(doc).__name__}")
    return doc


# ---------------------------------------------------------------------------
# shared: exact key sets on source sections
# ---------------------------------------------------------------------------

def _require_keys(
    section: object,
    keys: tuple[str, ...],
    path: list[str],
    optional: tuple[str, ...] = (),
) -> Mapping:
    """Every source section is closed: exactly `keys`, plus at most `optional`."""
    if not isinstance(section, Mapping):
        raise BenchyError("compile", "invalid_value", f"expected a mapping, got {type(section).__name__}", path)
    for key in keys:
        if key not in section:
            raise BenchyError("compile", "missing_key", f"required key {key!r} is missing", path)
    for extra in sorted(set(section) - set(keys) - set(optional)):
        raise BenchyError(
            "compile", "unknown_key",
            f"unknown key {extra!r}; expected {', '.join(keys + optional)}", path,
        )
    return section


# ---------------------------------------------------------------------------
# scoring (spec §6)
# ---------------------------------------------------------------------------

def compile_scoring(section: object, output_ir: Mapping) -> dict:
    """Compile `scoring` into fully explicit IR dimensions.

    The output schema's leaves are the scoring dimensions, so the weight tree must
    mirror it exactly: one weight per leaf, none missing, none extra, and none on
    an intermediate object. Emitted in output-schema order — not weight-mapping
    order — so the IR and every `field_scores` array are deterministic.
    """
    _require_keys(section, ("weights", "aggregator"), ["scoring"])
    if section["aggregator"] != "weighted_mean":
        raise BenchyError(
            "compile", "invalid_value",
            f"the only instance aggregator in Benchy 1.0 is 'weighted_mean', got {section['aggregator']!r}",
            ["scoring", "aggregator"],
        )
    dimensions = _dimensions(section["weights"], output_ir)
    if sum(d["weight"] for d in dimensions) <= 0:
        raise BenchyError(
            "compile", "invalid_weight",
            "at least one scoring dimension must have positive weight",
            ["scoring", "weights"],
        )
    return {
        "evaluator": "exact_match",
        "dimensions": dimensions,
        "instance_aggregator": "weighted_mean",
        "benchmark_aggregator": "mean",
    }


def _dimensions(weights: object, node: Mapping, path: tuple[str, ...] = ()) -> list[dict]:
    if node["type"] == "object":
        if not isinstance(weights, Mapping):
            raise BenchyError(
                "compile", "invalid_weight",
                "this output field is an object; weights must mirror it rather than carry a value",
                list(path),
            )
        out: list[dict] = []
        for name, child in node["fields"].items():
            if name not in weights:
                raise BenchyError("compile", "missing_weight", "output leaf has no weight", list(path + (name,)))
            out.extend(_dimensions(weights[name], child, path + (name,)))
        for extra in sorted(set(weights) - set(node["fields"])):
            raise BenchyError(
                "compile", "extra_weight", "weight does not correspond to any output field", list(path + (extra,))
            )
        return out

    if isinstance(weights, Mapping):
        raise BenchyError(
            "compile", "invalid_weight",
            "this output field is a leaf; its weight must be a number, not a sub-tree",
            list(path),
        )
    if isinstance(weights, bool) or not isinstance(weights, (int, float)):
        raise BenchyError("compile", "invalid_weight", f"weight must be a number, got {weights!r}", list(path))
    if not math.isfinite(weights):
        raise BenchyError("compile", "invalid_weight", "weight must be finite", list(path))
    if weights < 0:
        raise BenchyError("compile", "invalid_weight", f"weight must be >= 0, got {weights}", list(path))
    return [{"path": list(path), "weight": float(weights)}]


# ---------------------------------------------------------------------------
# the compiler (handoff §15)
# ---------------------------------------------------------------------------

def compile_benchmark(text: str, *, ontology_store: Any = None) -> dict:
    """Compile benchmark YAML into canonical JSON IR.

    Pure: the only file this reads is the ontology registry. `data.path` is carried
    through verbatim and resolved against the benchmark workspace at run time, so
    the IR is portable and compiling does not require the dataset to exist yet.

    Steps are numbered to match handoff §15; the order is what makes diagnostics
    stable when a document is wrong in several ways at once.
    """
    doc = parse(text)                                                               # 1
    _require_keys(doc, ROOT_KEYS, [])                                               # 2
    if doc["version"] != SPEC_VERSION:                                              # 3
        raise BenchyError(
            "compile", "unsupported_version",
            f"this Benchy implements specification {SPEC_VERSION}, got {doc['version']!r}",
            ["version"],
        )
    registry = ontology.load(doc["ontology_version"], store=ontology_store)         # 4
    benchmark = _compile_classification(doc["benchmark"])
    ontology.check_classification(benchmark, registry)                              # 5
    program = _compile_program(doc["program"])                                      # 6
    ontology.check_program(benchmark, program["input"], program["output"], registry) # 7
    scoring = compile_scoring(doc["scoring"], program["output"])                     # 8, 9
    data = _compile_data(doc["data"])                                               # 10
    system = _compile_ai_system(doc["ai-system"])                                    # 11
    return {                                                                        # 12
        "version": SPEC_VERSION,
        "ontology_version": registry["version"],
        "benchmark": benchmark,
        "program": program,
        "scoring": scoring,
        "data": data,
        "ai-system": system,
    }


def _compile_classification(section: object) -> dict:
    _require_keys(section, ("task", "domain", "language"), ["benchmark"])
    language = section["language"]
    return {
        "task": section["task"],
        "domain": section["domain"],
        "language": dict(language) if isinstance(language, Mapping) else language,
    }


def _compile_program(section: object) -> dict:
    _require_keys(section, ("input", "output"), ["program"])
    return {side: _compile_side(section[side], side) for side in ("input", "output")}


def _compile_side(node: object, side: str) -> dict:
    # An input/output is a named-field object. A bare type token or a root enum
    # would be an anonymous root scalar, which paper §10 places outside the language.
    if not isinstance(node, Mapping) or set(node) == {"enum"}:
        raise BenchyError(
            "compile", "invalid_schema",
            f"program.{side} must be a mapping of named fields, not a bare value",
            ["program", side],
        )
    return types.compile_schema(node, path=("program", side))


def _compile_data(section: object) -> dict:
    _require_keys(section, ("path",), ["data"])
    path = section["path"]
    if not isinstance(path, str) or not path:
        raise BenchyError("compile", "invalid_value", f"data.path must be a non-empty string, got {path!r}", ["data", "path"])
    return {"path": path, "format": "jsonl"}


def _compile_ai_system(section: object) -> dict:
    """The AI-system's *semantic definition* — what is being evaluated.

    How this environment invokes it (credentials, transport, SDK construction) is
    runtime policy bound to an adapter, and deliberately absent here.
    """
    if not isinstance(section, Mapping):
        raise BenchyError("compile", "invalid_ai_system", "ai-system must be a mapping", ["ai-system"])
    kind = section.get("type")
    if kind == "external":
        _require_keys(section, ("type", "id"), ["ai-system"])
        if not isinstance(section["id"], str) or not section["id"]:
            raise BenchyError("compile", "invalid_ai_system", "ai-system.id must be a non-empty string", ["ai-system", "id"])
        return {"type": "external", "id": section["id"]}
    if kind == "model":
        _require_keys(section, ("type", "provider", "model"), ["ai-system"], optional=("prompt", "parameters"))
        for key in ("provider", "model"):
            if not isinstance(section[key], str) or not section[key]:
                raise BenchyError("compile", "invalid_ai_system", f"ai-system.{key} must be a non-empty string", ["ai-system", key])
        if "prompt" in section and not isinstance(section["prompt"], str):
            raise BenchyError("compile", "invalid_ai_system", "ai-system.prompt must be a path string", ["ai-system", "prompt"])
        if "parameters" in section and not isinstance(section["parameters"], Mapping):
            raise BenchyError("compile", "invalid_ai_system", "ai-system.parameters must be a mapping", ["ai-system", "parameters"])
        # Only what the author wrote: no hidden defaults are injected.
        return {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in section.items()}
    raise BenchyError(
        "compile", "invalid_ai_system",
        f"ai-system.type must be 'external' or 'model', got {kind!r}",
        ["ai-system", "type"],
    )
