"""the exam — a benchmark as an exam, in data.

An exam lives in a directory named by its ontology path (/sentiment -> /sentiment).
It is three data files a non-engineer can read:

    question.json    — what the taker must produce (in/out shape + instructions)
    cases.json       — the pages: each page gives the prompt and the expected answer
    answer_key.json  — how each page is graded, and how points combine

Nothing else. A benchmark is data, never required Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ── the exam words ────────────────────────────────────────────────────
# Exam (question + pages + answer key) — plus Taker/ReportCard in sit.py.
# The question, the pages and the key are kept AS WRITTEN: the engine never
# needs to interpret the question, so a wrapper class per data file would be
# a mirror of json.loads with a nicer name. Pages and the key stay raw dicts.
# The grading seam is Exam.grade_page — when a real exam needs a rule beyond
# exact, that seam is where it goes (cycle 9 deleted the empty EXAM_RULES
# registry: speculative machinery, zero users, zero honest exam word).


# ── the exam itself ────────────────────────────────────────────────────

@dataclass
class Exam:
    """A benchmark: its pages and answer key. Nothing else.

    The question is kept as written on disk (question.json) — it is for
    the taker (and the human author) to read; the engine never
    interprets it, so it is not carried.
    """
    path: str                        # ontology path, e.g. "/sentiment"
    pages: list[dict]                # cases.json as written: prompt/expected/points
    answer_key: dict                 # answer_key.json as written: grade (+ rule args)

    @classmethod
    def from_dir(cls, dir_path: Path) -> "Exam":
        """Read an exam from its directory of three data files."""
        d = Path(dir_path)
        q = json.loads((d / "question.json").read_text())
        c = json.loads((d / "cases.json").read_text())
        k = json.loads((d / "answer_key.json").read_text())
        return cls(path=q["path"], pages=c["pages"], answer_key=k)

    def grade_page(self, page: dict, actual) -> float:
        """Grade one page: 1 point if the answer matches the key, else 0.

        The builtin rule is exact match. An answer key naming anything else
        fails loudly — this exam wants a rule the engine does not know.
        When a real exam needs one, this seam is where it goes.
        """
        grade = self.answer_key["grade"]
        if grade == "exact":
            return 1.0 if actual == page.get("expected") else 0.0
        raise ValueError(f"unknown grading rule: {grade!r} — "
                         "this exam wants a rule the engine does not know.")