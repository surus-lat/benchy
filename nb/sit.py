"""sitting the exam — taker, graded run, retake (resume), report card.

Exam words for the runtime half:

    Taker       — anyone who can answer a question: answer(prompt) -> answer
                  (a model, a node, a workflow, an agent — all the same here)
    sit()       — the taker takes the exam, page by page
    retake()    — sit() again, skipping pages already answered (resume)
    ReportCard  — the graded artifact: per-page scores + the exam score

The ReportCard IS the loss: score = how well you did, loss = 1 - score.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .exam import Exam


@dataclass
class Taker:
    """A system under evaluation: one who answers. A name plus a callable."""
    name: str
    answer: object            # callable: (prompt: dict) -> answer

    def sit(self, prompt: dict):
        return self.answer(prompt)


@dataclass
class PageResult:
    """One graded page: what was asked, what was answered, points earned."""
    page: int          # index of the page in the exam
    prompt: dict
    expected: object
    answered: object
    points: float      # what the page was worth
    earned: float      # fraction of the points earned, 0..1


@dataclass
class ReportCard:
    """The graded artifact: per-page results + the exam score. This is the JSON."""
    exam: str                     # ontology path of the exam
    taker: str                    # who sat the exam
    pages: list                   # per-page results (PageResult as dicts)
    score: float                  # exam score, weighted mean of page scores
    loss: float                   # 1 - score
    taken_at: str                 # ISO timestamp

    def write(self, out_dir: Path, filename: str = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = filename or f"report_card_{self.taker}.json"
        path = out / name
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def sit(exam: Exam, taker: Taker, limit: int | None = None,
        workbox: Path | None = None) -> ReportCard:
    """The taker takes the exam: every page, graded page by page.

    workbox: a directory to scribble answers into as we go (enables retake).
    When given, each page's answer is saved as soon as it is produced, so an
    interrupted exam can be retaken without re-answering what is already done.
    """
    pages = exam.pages if limit is None else exam.pages[:limit]
    done: list[PageResult] = []
    for i, page in enumerate(pages):
        if workbox is not None and _read_scribble(workbox, i) is not _UNANSWERED:
            answered = _read_scribble(workbox, i)   # already answered: keep it
        else:
            answered = taker.sit(page["prompt"])
            if workbox is not None:
                _scribble(workbox, i, answered)
        earned = exam.grade_page(page, answered)
        done.append(PageResult(page=i, prompt=page["prompt"],
                               expected=page.get("expected"),
                               answered=answered,
                               points=page.get("points", 1.0), earned=earned))
    total = sum(d.points for d in done)
    score = (sum(d.points * d.earned for d in done) / total) if total else 0.0
    return ReportCard(exam=exam.path, taker=taker.name,
                      pages=[asdict(d) for d in done],
                      score=score, loss=1.0 - score,
                      taken_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def retake(exam: Exam, taker: Taker, workbox: Path,
           limit: int | None = None) -> ReportCard:
    """Sit the exam again, keeping answers already in the workbox (resume)."""
    return sit(exam, taker, limit=limit, workbox=workbox)


def as_loss(exam: Exam, taker: Taker, limit: int = None) -> float:
    """The exam as a loss function over takers: sit, then 1 - score."""
    return sit(exam, taker, limit=limit).loss


_UNANSWERED = object()


def _scribble(workbox: Path, i: int, answered) -> None:
    wb = Path(workbox)
    wb.mkdir(parents=True, exist_ok=True)
    (wb / f"page_{i:04d}.json").write_text(json.dumps({"answered": answered}))


def _read_scribble(workbox: Path, i: int):
    p = Path(workbox) / f"page_{i:04d}.json"
    if not p.exists():
        return _UNANSWERED
    return json.loads(p.read_text())["answered"]