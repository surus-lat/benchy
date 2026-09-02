"""sitting the exam — taker, graded run, resume, report card.

Exam words for the runtime half:

    Taker       — anyone who can answer a question: answer(prompt) -> answer
                  (a model, a node, a workflow, an agent — all the same here)
    sit()       — the taker takes the exam, page by page
    ReportCard  — the graded artifact: per-page scores + the exam score

The ReportCard IS the loss: score = how well you did, loss = 1 - score.
Resume is not a second verb: sit() again over a workbox keeps what was
already scribbled — retaking an exam IS sitting it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .exam import Exam


@dataclass
class Taker:
    """A system under evaluation: one who answers. A name plus a callable."""
    name: str
    answer: callable          # (prompt: dict) -> answer


@dataclass
class ReportCard:
    """The graded artifact: per-page results + the exam score. This is the JSON.

    Each row carries only what the exam's cases.json cannot: which page
    (by index), what was answered, and the fraction of its points earned.
    The card never re-states the page — the exam is the single source of it.
    """
    exam: str                     # ontology path of the exam
    taker: str                    # who sat the exam
    pages: list                   # rows: {page, answered, earned}
    score: float                  # exam score, weighted mean of page scores
    loss: float                   # 1 - score
    taken_at: str                 # ISO timestamp

    def write(self, out_dir: Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"report_card_{self.taker}.json"
        path.write_text(json.dumps(self.__dict__, indent=2))
        return path


def sit(exam: Exam, taker: Taker, limit: int | None = None,
        workbox: Path | None = None) -> ReportCard:
    """The taker takes the exam: every page, graded page by page.

    workbox: a directory to scribble answers into as we go. The scribbles
    are one honest answers.json (page index -> answer), rewritten after
    every page — so an interrupted exam is resumed by sitting again:
    pages already answered are kept, only the rest are asked.
    """
    pages = exam.pages if limit is None else exam.pages[:limit]
    wb = Path(workbox) / "answers.json" if workbox is not None else None
    answers: dict = json.loads(wb.read_text()) if wb and wb.exists() else {}
    done: list[dict] = []
    for i, page in enumerate(pages):
        if str(i) in answers:            # already answered: keep it
            answered = answers[str(i)]
        else:
            answered = taker.answer(page["prompt"])
            if wb is not None:
                answers[str(i)] = answered
                wb.parent.mkdir(parents=True, exist_ok=True)
                wb.write_text(json.dumps(answers, indent=2))
        done.append({"page": i, "answered": answered,
                     "earned": exam.grade_page(page, answered)})
    total = sum(page.get("points", 1.0) for page in pages)
    score = (sum(page.get("points", 1.0) * d["earned"]
                 for page, d in zip(pages, done)) / total) if total else 0.0
    return ReportCard(exam=exam.path, taker=taker.name,
                      pages=done, score=score, loss=1.0 - score,
                      taken_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def as_loss(exam: Exam, taker: Taker, limit: int = None) -> float:
    """The exam as a loss function over takers: sit, then 1 - score."""
    return sit(exam, taker, limit=limit).loss