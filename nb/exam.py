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


# ── the four concepts of a benchmark, as exam words ──────────────────
# Question · Page(case) · AnswerKey · Exam — plus Taker/ReportCard in sit.py


@dataclass
class Question:
    """What the exam asks for: what goes in, what must come out."""
    asks: dict          # the input the taker receives, by name
    answer_shape: dict   # the form an acceptable answer takes
    instructions: str = ""

    @classmethod
    def from_data(cls, d: dict) -> "Question":
        return cls(asks=d["asks"], answer_shape=d["answer_shape"],
                   instructions=d.get("instructions", ""))


@dataclass
class Page:
    """One case: the prompt for one page of the exam, and the expected answer."""
    prompt: dict
    expected: object          # may be None when the exam is unsupervised
    points: float = 1.0       # weight of this page in the final grade

    @classmethod
    def from_data(cls, d: dict) -> "Page":
        return cls(prompt=d["prompt"], expected=d.get("expected"),
                   points=float(d.get("points", 1.0)))


@dataclass
class AnswerKey:
    """How to grade one page: compare expected vs actual, then how points combine."""
    grade: str               # the rule name, e.g. "exact" (a builtin rule)
    rule: dict               # extra arguments the rule needs, if any
    combine: str = "mean"    # how page scores combine into the exam score

    @classmethod
    def from_data(cls, d: dict) -> "AnswerKey":
        return cls(grade=d["grade"], rule=d.get("rule", {}),
                   combine=d.get("combine", "mean"))


# ── grading rules: the builtin answer-key rules ───────────────────────
# A rule is a function (expected, actual, rule_args) -> score in [0, 1].
# New rules are the Python escape hatch, registered by name in the key.


def grade_exact(expected, actual, rule: dict) -> float:
    """1 point for an exact match, else 0."""
    return 1.0 if actual == expected else 0.0


def grade_keyword(expected, actual, rule: dict) -> float:
    """1 point if every keyword in rule['keywords'] appears in the answer."""
    kws = rule.get("keywords", [])
    if not isinstance(actual, str):
        return 0.0
    return 1.0 if all(k in actual for k in kws) else 0.0


GRADES = {"exact": grade_exact, "keyword": grade_keyword}


# ── the exam itself ───────────────────────────────────────────────────

@dataclass
class Exam:
    """A benchmark: a question, its pages, and the answer key. Nothing else."""
    path: str                        # ontology path, e.g. "/sentiment"
    question: Question
    pages: list[Page]
    answer_key: AnswerKey

    @classmethod
    def from_dir(cls, dir_path: Path) -> "Exam":
        """Read an exam from its directory of three data files."""
        d = Path(dir_path)
        q = json.loads((d / "question.json").read_text())
        c = json.loads((d / "cases.json").read_text())
        k = json.loads((d / "answer_key.json").read_text())
        return cls(path=q["path"], question=Question.from_data(q["question"]),
                   pages=[Page.from_data(p) for p in c["pages"]],
                   answer_key=AnswerKey.from_data(k))

    def grade_page(self, page: Page, actual) -> float:
        """Grade one page against the key's rule."""
        rule = GRADES.get(self.answer_key.grade)
        if rule is None:
            raise ValueError(f"unknown grading rule: {self.answer_key.grade!r} — "
                             "this exam wants a rule the engine does not know.")
        return float(rule(page.expected, actual, self.answer_key.rule))

    def combine(self, page_scores: list[float]) -> float:
        """Turn per-page points into the exam score. Weighted by page points."""
        if not page_scores:
            return 0.0
        pts = [p.points for p in self.pages]
        return sum(pt * s for pt, s in zip(pts, page_scores)) / sum(pts)