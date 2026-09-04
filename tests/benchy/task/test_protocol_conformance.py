"""Definition-of-done checks called out explicitly in the build brief."""

from __future__ import annotations

from benchy.core import Task as TaskProto
from benchy.task import builtin


def test_freeform_satisfies_the_core_task_protocol():
    assert isinstance(builtin.freeform(ontology="x"), TaskProto)


def test_every_builtin_shape_satisfies_the_core_task_protocol():
    assert isinstance(builtin.transcription(ontology="transcription/x/y"), TaskProto)
    assert isinstance(
        builtin.structured_extraction(output={"type": "object"}, ontology="structured_extraction/x/y"),
        TaskProto,
    )
    assert isinstance(builtin.classification(["a", "b"], ontology="classification/x/y"), TaskProto)
    assert isinstance(builtin.freeform(ontology="freeform/x/y"), TaskProto)
