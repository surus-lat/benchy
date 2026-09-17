"""P10 — every conformance case in the build plan is exercised by a named test.

The matrix is the contract, so coverage of it should not be able to erode quietly.
This scans test function names for their `cNN` tokens and fails if one goes missing.

C07 and C08 are absent by design: both are `transcribe` cases, and `transcribe` was
withdrawn from ontology 1.0 (see paper/v10-transcribe-removal-brief.md) because exact
match cannot rank transcription systems.
"""

from __future__ import annotations

import re
from pathlib import Path

WITHDRAWN = {"c07", "c08"}
EXPECTED = {f"c{n:02d}" for n in range(1, 34)} - WITHDRAWN

_TEST = re.compile(r"^\s*(?:async\s+)?def (test_\w+)", re.MULTILINE)
_CASE = re.compile(r"c\d{2}")


def covered() -> set[str]:
    here = Path(__file__).parent
    names = [m for path in here.glob("test_*.py") for m in _TEST.findall(path.read_text())]
    return {case for name in names for case in _CASE.findall(name)}


def test_every_conformance_case_has_a_named_test():
    assert not EXPECTED - covered(), f"uncovered conformance cases: {sorted(EXPECTED - covered())}"


def test_no_test_claims_a_withdrawn_case():
    assert not WITHDRAWN & covered()
