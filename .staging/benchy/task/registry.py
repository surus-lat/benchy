"""Ontology registry: `OntologyPath` -> `Task`, with partial-path resolution.

A Task registers under its full ontology path, e.g.
`transcription/fleurs/pt-BR`. Looking up a shorter prefix -- just
`transcription`, say -- should resolve unambiguously if exactly one
registered Task falls under that prefix, and raise a `LoadError` naming
every candidate otherwise. This mirrors how `OntologyPath.is_prefix_of` is
documented to be used in `benchy.core`.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchy.core import LoadError, OntologyPath

from .base import Task


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[OntologyPath, Task] = {}

    def register(self, task: Task) -> None:
        self._tasks[task.ontology] = task

    def load(self, path: str | OntologyPath) -> Task:
        key = path if isinstance(path, OntologyPath) else OntologyPath.parse(path)
        if key in self._tasks:
            return self._tasks[key]

        candidates = [task for task in self._tasks.values() if key.is_prefix_of(task.ontology)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise LoadError(f"no task registered under ontology {str(key)!r}")
        names = ", ".join(sorted(str(task.ontology) for task in candidates))
        raise LoadError(f"ambiguous ontology lookup {str(key)!r}: candidates are [{names}]")

    def __contains__(self, path: str | OntologyPath) -> bool:
        try:
            self.load(path)
        except LoadError:
            return False
        return True

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def clear(self) -> None:
        self._tasks.clear()


#: The default, process-wide registry. `register`/`load` below delegate to it.
registry = TaskRegistry()


def register(task: Task) -> None:
    registry.register(task)


def load(path: str | OntologyPath) -> Task:
    return registry.load(path)
