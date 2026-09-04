"""Ontology registry: register/load and partial-path resolution."""

from __future__ import annotations

import pytest

from benchy.core import LoadError, OntologyPath
from benchy.task import Task, TaskRegistry


def make_task(ontology: str) -> Task:
    return Task(name=ontology, ontology=ontology, input=None, output={"type": "string"})


class TestRegisterAndLoad:
    def test_exact_lookup_round_trips(self):
        reg = TaskRegistry()
        task = make_task("transcription/fleurs/pt-BR")
        reg.register(task)
        assert reg.load("transcription/fleurs/pt-BR") is task

    def test_load_accepts_an_ontology_path_object(self):
        reg = TaskRegistry()
        task = make_task("transcription/fleurs/pt-BR")
        reg.register(task)
        assert reg.load(OntologyPath.parse("transcription/fleurs/pt-BR")) is task

    def test_missing_ontology_raises_load_error(self):
        reg = TaskRegistry()
        with pytest.raises(LoadError):
            reg.load("nope/nothing/here")

    def test_len_and_iter(self):
        reg = TaskRegistry()
        reg.register(make_task("a/b/c"))
        reg.register(make_task("d/e/f"))
        assert len(reg) == 2
        assert {str(t.ontology) for t in reg} == {"a/b/c", "d/e/f"}

    def test_contains(self):
        reg = TaskRegistry()
        reg.register(make_task("a/b/c"))
        assert "a/b/c" in reg
        assert "x/y/z" not in reg

    def test_clear(self):
        reg = TaskRegistry()
        reg.register(make_task("a/b/c"))
        reg.clear()
        assert len(reg) == 0


class TestPartialResolution:
    def test_unambiguous_prefix_resolves(self):
        reg = TaskRegistry()
        reg.register(make_task("transcription/fleurs/pt-BR"))
        resolved = reg.load("transcription")
        assert str(resolved.ontology) == "transcription/fleurs/pt-BR"

    def test_unambiguous_two_segment_prefix_resolves(self):
        reg = TaskRegistry()
        reg.register(make_task("transcription/fleurs/pt-BR"))
        resolved = reg.load("transcription/fleurs")
        assert str(resolved.ontology) == "transcription/fleurs/pt-BR"

    def test_ambiguous_prefix_raises_naming_candidates(self):
        reg = TaskRegistry()
        reg.register(make_task("transcription/fleurs/pt-BR"))
        reg.register(make_task("transcription/fleurs/es-AR"))
        with pytest.raises(LoadError) as exc_info:
            reg.load("transcription")
        message = str(exc_info.value)
        assert "transcription/fleurs/pt-BR" in message
        assert "transcription/fleurs/es-AR" in message

    def test_sibling_prefix_is_not_a_match(self):
        """`trans` must not fuzzy-match `transcription` (segment-wise, not string-wise)."""
        reg = TaskRegistry()
        reg.register(make_task("transcription/fleurs/pt-BR"))
        with pytest.raises(LoadError):
            reg.load("trans")


class TestModuleLevelDefaultRegistry:
    def test_register_and_load_use_the_shared_default(self):
        import benchy.task as task_mod

        task_mod.registry.clear()
        task = make_task("qa/general/en")
        task_mod.register(task)
        assert task_mod.load("qa/general/en") is task
        task_mod.registry.clear()
