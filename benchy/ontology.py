"""The shared SURUS ontology: /<task>/<domain>/<language>/.

Two responsibilities that the paper deliberately keeps apart:

**The registry** (Appendix B) is versioned external data — identifiers and
descriptions, shared with DataHub, EvalsHub and the rest of SURUS. It contains no
constraint language.

**The task validators** are Benchy code, keyed by ontology version. A task defines
a family of admissible programs, and the compiler checks `P ∈ P_T`. Expressing
that as data would require inventing a constraint DSL, which paper §10 and handoff
§18 both rule out; three small functions say it more precisely and can be read.

A benchmark pins `ontology_version` and never a filepath — the runtime resolves the
requested version from its registry store (amendment §4). `load` therefore accepts
a version token and rejects anything path-shaped.

Ontology 1.0 has three tasks. `transcribe` is absent because exact match cannot
rank transcription systems; see `ontologies/1.0.yaml` and paper Appendix E.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from benchy import types
from benchy.errors import BenchyError

__all__ = ["load", "check_classification", "check_program", "STORE"]

#: The packaged registry store. A deployment may point `load` elsewhere.
STORE = Path(__file__).parent / "ontologies"

_VERSION_RE = re.compile(r"\A\d+\.\d+\Z")


# ---------------------------------------------------------------------------
# registry loading
# ---------------------------------------------------------------------------

def load(version: object, *, store: Path | None = None) -> dict:
    """Resolve an ontology registry by version token (never by path)."""
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise BenchyError(
            "compile", "unsupported_ontology_version",
            f"ontology_version must be a version token such as '1.0', got {version!r}",
            ["ontology_version"],
        )
    if version not in _VALIDATORS:
        raise BenchyError(
            "compile", "unsupported_ontology_version",
            f"this Benchy implements task validators for {', '.join(sorted(_VALIDATORS))}, not {version!r}",
            ["ontology_version"],
        )
    path = (store or STORE) / f"{version}.yaml"
    if not path.is_file():
        raise BenchyError(
            "compile", "unsupported_ontology_version",
            f"no ontology registry {version!r} in {store or STORE}",
            ["ontology_version"],
        )
    registry = yaml.safe_load(path.read_text())
    if not isinstance(registry, Mapping) or not all(
        isinstance(registry.get(k), (str, dict)) for k in ("version", "tasks", "domains", "languages")
    ):
        raise BenchyError(
            "compile", "unsupported_ontology_version",
            f"registry {version!r} must declare version, tasks, domains and languages",
            ["ontology_version"],
        )
    return dict(registry)


# ---------------------------------------------------------------------------
# classification + task-to-program validation
# ---------------------------------------------------------------------------

def check_classification(benchmark: Mapping, registry: Mapping) -> None:
    """Step 5 of handoff §15: task, domain and language against the registry.

    Runs before the program is compiled, so a misspelled domain is reported as a
    misspelled domain rather than being masked by a schema error.
    """
    task = benchmark.get("task")
    if task not in registry["tasks"]:
        raise BenchyError(
            "compile", "unknown_task",
            f"{task!r} is not a task in ontology {registry['version']} "
            f"({', '.join(sorted(registry['tasks']))})",
            ["benchmark", "task"],
        )
    if task not in _VALIDATORS[registry["version"]]:
        raise BenchyError(
            "compile", "unknown_task",
            f"ontology {registry['version']} declares task {task!r} but this Benchy "
            f"implements no structural validator for it",
            ["benchmark", "task"],
        )
    if benchmark.get("domain") not in registry["domains"]:
        raise BenchyError(
            "compile", "unknown_domain",
            f"{benchmark.get('domain')!r} is not a domain in ontology {registry['version']}",
            ["benchmark", "domain"],
        )
    _VALIDATORS[registry["version"]][task][0](benchmark, registry)


def check_program(benchmark: Mapping, input_ir: Mapping, output_ir: Mapping, registry: Mapping) -> None:
    """Step 7 of handoff §15: the task's structural constraint, `P ∈ P_T`."""
    _VALIDATORS[registry["version"]][benchmark["task"]][1](input_ir, output_ir)


# --- language shape (paper §3) ---------------------------------------------

def _one_language(benchmark: Mapping, registry: Mapping) -> None:
    language = benchmark.get("language")
    if not isinstance(language, str) or language not in registry["languages"]:
        raise BenchyError(
            "compile", "unknown_language",
            f"language must be one registered language string, got {language!r}",
            ["benchmark", "language"],
        )


def _source_target(benchmark: Mapping, registry: Mapping) -> None:
    language = benchmark.get("language")
    if not isinstance(language, Mapping) or set(language) != {"source", "target"}:
        raise BenchyError(
            "compile", "task_program_mismatch",
            "translate requires language to be an ordered {source, target} relation",
            ["benchmark", "language"],
        )
    for side in ("source", "target"):
        if language[side] not in registry["languages"]:
            raise BenchyError(
                "compile", "unknown_language",
                f"{language[side]!r} is not a registered language",
                ["benchmark", "language", side],
            )


# --- program structure (spec §5) -------------------------------------------

def _leaf_types(ir: Mapping) -> list[str]:
    return [types.at(ir, path)["type"] for path in types.leaves(ir)]


def _sole_output(output_ir: Mapping, expected: str, task: str) -> None:
    kinds = _leaf_types(output_ir)
    if len(kinds) != 1 or kinds[0] != expected:
        raise BenchyError(
            "compile", "task_program_mismatch",
            f"{task} requires exactly one output leaf of type {expected}, got {kinds}",
            ["program", "output"],
        )


def _extract_program(*_: Mapping) -> None:
    """No task-specific structural constraint beyond the normal program rules."""


def _classify_program(_input_ir: Mapping, output_ir: Mapping) -> None:
    _sole_output(output_ir, "enum", "classify")


def _translate_program(input_ir: Mapping, output_ir: Mapping) -> None:
    if "string" not in _leaf_types(input_ir):
        raise BenchyError(
            "compile", "task_program_mismatch",
            "translate requires at least one string input leaf",
            ["program", "input"],
        )
    _sole_output(output_ir, "string", "translate")


#: Task validators keyed by ontology version, then task: one pair per task, holding
#: its language rule (checked with the classification) and its program rule (checked
#: after the program compiles). Adding ontology 1.1 means adding a registry YAML and
#: one entry here — the compiler does not change.
_VALIDATORS: dict[str, dict[str, tuple]] = {
    "1.0": {
        "extract": (_one_language, _extract_program),
        "classify": (_one_language, _classify_program),
        "translate": (_source_target, _translate_program),
    },
}
