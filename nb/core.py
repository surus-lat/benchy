"""nb.core — the contracts.

Benchy has four pillars: TASK, SCORING, DATA, SYSTEM.  This file is the
contract layer: the records that flow between the pillars and the one
runtime-checkable protocol an exam-taker must satisfy.  No behavior lives
here — implementations are trivial one-liners elsewhere.
"""
from typing import Protocol, TypedDict, runtime_checkable


class Case(TypedDict):
    """Pillar DATA — one exam page: the input and the expected answer."""

    input: object
    expected: object


@runtime_checkable
class System(Protocol):
    """Pillar SYSTEM — any AI program: model, node, workflow, agent, stub.

    The whole contract is one method: given an input, predict.  Conformance
    is structural (duck): no inheritance, no registration.
    """

    def invoke(self, x: object) -> object: ...


