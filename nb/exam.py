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


# extra grading rules live here once a real exam needs one (the escape hatch)
EXAM_RULES: dict = {}


# ── the exam itself ────────────────────────────────────────────────────

@dataclass
class Exam:
    """A benchmark: a question, its pages, and the answer key. Nothing else.

    The question is kept as written — it is for the taker (and the human
    author) to read; the engine never needs to interpret it.
    """
    path: str                        # ontology path, e.g. "/sentiment"
    question: dict                   # question.json as written
    pages: list[dict]                # cases.json as written: prompt/expected/points
    answer_key: dict                 # answer_key.json as written: grade (+ rule args)

    @classmethod
    def from_dir(cls, dir_path: Path) -> "Exam":
        """Read an exam from its directory of three data files."""
        d = Path(dir_path)
        q = json.loads((d / "question.json").read_text())
        c = json.loads((d / "cases.json").read_text())
        k = json.loads((d / "answer_key.json").read_text())
        return cls(path=q["path"], question=q["question"],
                   pages=c["pages"], answer_key=k)

    def grade_page(self, page: dict, actual) -> float:
        """Grade one page: 1 point if the answer matches the key, else 0.

        The builtin rule is exact match. Other rules named in an answer key
        are the Python escape hatch — import nb.exam and set EXAM_RULES.
        """
        grade = self.answer_key["grade"]
        if grade == "exact":
            return 1.0 if actual == page.get("expected") else 0.0
        rule = EXAM_RULES.get(grade)
        if rule is None:
            raise ValueError(f"unknown grading rule: {grade!r} — "
                             "this exam wants a rule the engine does not know.")
        return float(rule(page.get("expected"), actual,
                          self.answer_key.get("rule", {})))