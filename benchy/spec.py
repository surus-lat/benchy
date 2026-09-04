"""benchy.spec — the declarative front-end: one YAML file -> a live exam.

The engine layer is behavior; a spec file is *names*. Every node resolves
through the module registries — ``benchy.task.builtin``, the
``benchy.scoring`` factories, ``benchy.system.load``, ``benchy.data.load``
— the same surfaces the seam tests pin. Compiling a spec produces exactly
the objects a hand-written benchmark would build; nothing here invents
semantics.

One file, three sections (see ``.plans/DECLARATIVE-SPEC.md``)::

    spec_version: 1
    exam:                      # fingerprinted — this IS the benchmark
      task:
        structured_extraction: {ontology: ..., output: {...}}
      data:  {spec: jsonl:train.jsonl, input: {text: text}, expected: expected}
      scoring:
        field_wise: {fields: [vendor], per_field: {exact_match: {}}}
    system: {url: openai:gpt-5-mini, temperature: 0}   # NOT fingerprinted
    run: {max_concurrency: 8}                          # NOT fingerprinted

The one resolution pattern: a node is a scalar (itself) or a one-key
mapping ``{name: kwargs}`` (registry lookup + call). Nesting is
type-directed and only legal where a parameter is Scorer-typed: scoring is
a compositional language; ``task.output`` and ``data`` kwargs are plain
data and never resolve. Behavior is never serialized; behavior is named.

All module imports are lazy (inside functions): ``import benchy.spec``
succeeds in any tree; compiling requires the modules — it runs at the
composition gate and in the merged tree.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from benchy.core import SchemaViolation

if TYPE_CHECKING:
    from benchy.benchmark import Benchmark
    from benchy.core import Data, Scorer, System, Task

__all__ = [
    "compile_exam", "compile_system", "compile_scoring", "compile_task",
    "compile_data", "fingerprint", "describe", "load_doc",
]

SPEC_VERSION = 1


# ---------------------------------------------------------------------------
# document loading
# ---------------------------------------------------------------------------

def load_doc(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Accept a path, a YAML string, or an already-parsed mapping."""
    if isinstance(source, Mapping):
        doc = dict(source)
    else:
        path = _as_path(source)
        text = path.read_text(encoding="utf-8") if path else str(source)
        doc = yaml.safe_load(text) or {}
    if not isinstance(doc, Mapping):
        raise SchemaViolation(f"spec: top level must be a mapping, got {type(doc).__name__}")
    version = doc.get("spec_version", SPEC_VERSION)
    if version != SPEC_VERSION:
        raise SchemaViolation(
            f"spec: unsupported spec_version {version!r} (supported: {SPEC_VERSION})"
        )
    return doc


def _as_path(source: str | Path) -> Path | None:
    s = str(source)
    if "\n" not in s and (s.endswith((".yaml", ".yml")) or Path(s).exists()):
        return Path(s)
    return None


def _exam(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    exam = doc.get("exam")
    if not isinstance(exam, Mapping):
        raise SchemaViolation("spec: missing 'exam' section (task, data, scoring)")
    return exam


# ---------------------------------------------------------------------------
# the one resolution pattern: {name: kwargs} -> lookup + call
# ---------------------------------------------------------------------------

def _call(registry: Mapping[str, Callable], node: Any, ctx: str) -> Any:
    if not (isinstance(node, Mapping) and len(node) == 1):
        raise SchemaViolation(
            f"spec.{ctx}: expected a single-key call {{name: {{kwargs...}}}}, got {node!r}"
        )
    (name, kwargs), = node.items()
    fn = registry.get(name)
    if fn is None:
        raise SchemaViolation(
            f"spec.{ctx}: unknown name {name!r} (known: {', '.join(sorted(registry))})"
        )
    if not isinstance(kwargs, Mapping):
        raise SchemaViolation(f"spec.{ctx}.{name}: kwargs must be a mapping, got {kwargs!r}")
    return fn(**dict(kwargs))


def _builtin_registry() -> dict[str, Callable]:
    from benchy import task as task_mod

    mod = task_mod.builtin.__name__
    return {
        n: o for n, o in vars(task_mod.builtin).items()
        if callable(o) and not n.startswith("_")
        and getattr(o, "__module__", None) == mod
    }


def _scoring_registry() -> dict[str, Callable]:
    from benchy import scoring

    return {
        n: o for n, o in vars(scoring).items()
        if callable(o) and not isinstance(o, type) and not n.startswith("_")
        and str(getattr(o, "__module__", "")).startswith("benchy.scoring")
        and n not in {"parse_scorer", "register_scorer"}
    }


def _scorer_typed(fn: Callable, arg: str) -> bool:
    """True when `fn`'s `arg` parameter is annotated as Scorer-typed."""
    hint = str((getattr(fn, "__annotations__", {}) or {}).get(arg, ""))
    return "Scorer" in hint


def compile_scoring(node: Any, *, registry: Mapping[str, Callable] | None = None) -> Scorer:
    """Compile a scoring call-tree into a live Scorer.

    Type-directed nesting: a kwarg resolves recursively iff its value is a
    one-key call AND the parameter is Scorer-typed. A data-like kwarg
    whose dict happens to collide with a factory name (``weights: {invert:
    1.0}``) must stay plain data — annotation discipline, not name magic.
    """
    reg = registry if registry is not None else _scoring_registry()
    if not (isinstance(node, Mapping) and len(node) == 1):
        raise SchemaViolation(
            f"spec.exam.scoring: expected a single-key call {{name: {{kwargs...}}}}, got {node!r}"
        )
    (name, kwargs), = node.items()
    fn = reg.get(name)
    if fn is None:
        raise SchemaViolation(
            f"spec.exam.scoring: unknown scorer {name!r} (known: {', '.join(sorted(reg))})"
        )
    if not isinstance(kwargs, Mapping):
        raise SchemaViolation(
            f"spec.exam.scoring.{name}: kwargs must be a mapping, got {kwargs!r}"
        )
    resolved: dict[str, Any] = {}
    for k, v in kwargs.items():
        if (
            isinstance(v, Mapping)
            and len(v) == 1
            and next(iter(v)) in reg
            and _scorer_typed(fn, k)
        ):
            resolved[k] = compile_scoring(v, registry=reg)
        else:
            resolved[k] = v
    return fn(**resolved)


def compile_task(node: Any) -> Task:
    return _call(_builtin_registry(), node, "exam.task")


def compile_data(node: Any, *, base_dir: Path | None = None) -> Data:
    """Compile the `data:` section through `benchy.data.load`.

    Local-file source specs (`jsonl:`, `csv:`, `tsv:`, `glob:`, bare paths)
    resolve against `base_dir` — the spec file's directory — so a shared
    benchmark folder runs from any cwd. Remote schemes (`hf:`) and
    absolute paths pass through untouched.
    """
    from benchy.data import load

    if not isinstance(node, Mapping):
        raise SchemaViolation(f"spec.exam.data: expected a mapping, got {node!r}")
    spec_str = node.get("spec")
    if not isinstance(spec_str, str) or not spec_str:
        raise SchemaViolation(
            f"spec.exam.data: 'spec' must be a non-empty source string, got {spec_str!r}"
        )
    return load(_resolve_spec(spec_str, base_dir), **{k: v for k, v in node.items() if k != "spec"})


_LOCAL_SCHEMES = {"jsonl", "csv", "tsv", "glob"}


def _resolve_spec(spec: str, base_dir: Path | None) -> str:
    if base_dir is None:
        return spec
    scheme, sep, rest = spec.partition(":")
    if scheme in _LOCAL_SCHEMES and sep and rest and not Path(rest).is_absolute():
        return f"{scheme}:{base_dir / rest}"
    if not sep and not Path(spec).is_absolute():
        return str(base_dir / spec)
    return spec


def compile_system(doc_or_source: str | Path | Mapping[str, Any]) -> System:
    """Compile the `system:` section (url + kwargs) through `system.load`."""
    node = load_doc(doc_or_source).get("system")
    if not isinstance(node, Mapping) or "url" not in node:
        raise SchemaViolation("spec.system: expected {url: scheme:rest, **kwargs}")
    from benchy.system import load

    return load(node["url"], **{k: v for k, v in node.items() if k != "url"})


def compile_exam(doc_or_source: str | Path | Mapping[str, Any]) -> Benchmark:
    """Compile the `exam:` section into a live, runnable Benchmark."""
    source_path = _as_path(doc_or_source) if not isinstance(doc_or_source, Mapping) else None
    exam = _exam(load_doc(doc_or_source))
    for key in ("task", "data", "scoring"):
        if key not in exam:
            raise SchemaViolation(f"spec.exam.{key}: required section missing")
    from benchy.benchmark import Benchmark

    return Benchmark(
        task=compile_task(exam["task"]),
        data=compile_data(exam["data"], base_dir=source_path.parent if source_path else None),
        scoring=compile_scoring(exam["scoring"]),
    )


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def fingerprint(doc_or_source: str | Path | Mapping[str, Any]) -> str:
    """sha256 over the canonical exam tree — task node, data node, and the
    compiled scorer's repr (which round-trips). Formatting, comments, and
    the `system:`/`run:` sections never affect it: comparability is a
    property of the exam, not of the run."""
    exam = _exam(load_doc(doc_or_source))
    canonical = {
        "task": exam.get("task"),
        "data": exam.get("data"),
        "scoring": repr(compile_scoring(exam["scoring"])),
    }
    blob = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the GUI contract
# ---------------------------------------------------------------------------

def _signature_json(fn: Callable) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {"params": [], "doc": None}
    params = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = None if p.default is p.empty else p.default
        params.append({
            "name": p.name,
            "annotation": str(p.annotation) if p.annotation is not p.empty else None,
            "default": default if isinstance(default, (str, int, float, bool, type(None))) else repr(default),
            "scorer_typed": _scorer_typed(fn, p.name),
        })
    doc = inspect.getdoc(fn) or ""
    return {"params": params, "doc": doc.splitlines()[0] if doc else None}


def describe() -> dict[str, Any]:
    """The editable ontology as JSON-able data. The GUI renders forms from
    this: for every task builtin and scoring factory — name, params,
    defaults, one-line doc; for system and data — the registered schemes.
    The frontend never parses Python; it reads this."""
    from benchy.system import schemes
    from benchy.data.sources import sources as data_sources

    task_reg, scoring_reg = _builtin_registry(), _scoring_registry()
    return {
        "spec_version": SPEC_VERSION,
        "task": {n: _signature_json(f) for n, f in sorted(task_reg.items())},
        "scoring": {n: _signature_json(f) for n, f in sorted(scoring_reg.items())},
        "system": {"schemes": sorted(schemes())},
        "data": {"sources": sorted(data_sources)},
    }