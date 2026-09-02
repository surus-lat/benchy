"""the exam hall — how you run an exam from the shell.

One command, two ways to say who is sitting:

    python -m nb.hall bench/hello --stub good     sit the exam with a stub
    python -m nb.hall bench/hello --retake DIR    finish an interrupted exam

The stubs are demo takers: keyword-tally (counts good/bad words) and
always-pos. Real takers are the Python escape hatch (anything with
answer(prompt) -> answer); stubs exist so the hall works offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exam import Exam
from .sit import Taker, sit, retake

GOOD_WORDS = ("great", "excelente", "loved")
BAD_WORDS = ("broken", "porqueria", "never")


def keyword_tally(prompt: dict) -> str:
    """Count good and bad words in the text; more good words -> pos, else neg."""
    text = prompt["text"].lower()
    good = sum(text.count(w) for w in GOOD_WORDS)
    bad = sum(text.count(w) for w in BAD_WORDS)
    return "pos" if good >= bad else "neg"


def always_pos(prompt: dict) -> str:
    return "pos"


STUBS = {"good": keyword_tally, "dumb": always_pos}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hall")
    p.add_argument("exam_dir", help="directory holding question/cases/answer_key")
    p.add_argument("--stub", choices=sorted(STUBS), help="which stub taker sits")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="where to put the report card")
    p.add_argument("--retake", default=None,
                   help="workbox dir with answers to keep (resume)")
    args = p.parse_args(argv)

    exam = Exam.from_dir(Path(args.exam_dir))
    if args.stub is None and args.retake is None:
        p.error("say who sits: --stub good|dumb, or --retake DIR")

    if args.stub is not None:
        taker = Taker(name=args.stub, answer=STUBS[args.stub])
        workbox = Path(args.retake) if args.retake else None
        card = sit(exam, taker, limit=args.limit, workbox=workbox)
        name = taker.name
    else:
        if args.retake is None:
            p.error("--retake needs DIR")
        taker = Taker(name="retake", answer=always_pos)
        card = retake(exam, taker, workbox=Path(args.retake), limit=args.limit)
        name = taker.name

    out = Path(args.out) if args.out else Path.cwd()
    path = card.write(out, filename=f"report_card_{name}.json")
    print(json.dumps({k: v for k, v in card.__dict__.items()}, indent=2)[:400])
    print(f"report card: {path}")
    print(f"{taker.name} scored {card.score} on {exam.path} (loss {card.loss})")
    return 0


if __name__ == "__main__":
    sys.exit(main())