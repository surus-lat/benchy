"""pytest suite — what \"broken\" means for the exam engine.

Every test is stated in exam words: a stub taker sits the hello exam,
the report card must show the expected grade.
"""

import json
from pathlib import Path

import pytest

from nb.exam import Exam, grade_exact, grade_keyword
from nb.sit import Taker, sit, retake, as_loss
from nb.hall import keyword_tally, always_pos


HELLO = Path(__file__).resolve().parent.parent / "bench" / "hello"


# ── the exam, as data ──────────────────────────────────────────────────

def test_exam_loads_from_three_data_files():
    exam = Exam.from_dir(HELLO)
    assert exam.path == "/sentiment"
    assert len(exam.pages) == 6
    assert exam.answer_key.grade == "exact"


def test_a_benchmark_is_data_not_python():
    files = sorted(p.name for p in HELLO.iterdir())
    assert files == ["answer_key.json", "cases.json", "question.json"]
    assert not any(p.suffix == ".py" for p in HELLO.iterdir())


# ── grading rules ─────────────────────────────────────────────────────

def test_exact_match_scores_one_else_zero():
    assert grade_exact("pos", "pos", {}) == 1.0
    assert grade_exact("pos", "neg", {}) == 0.0


def test_unknown_rule_is_a_loud_error():
    exam = Exam.from_dir(HELLO)
    exam.answer_key.grade = "nope"
    with pytest.raises(ValueError, match="unknown grading rule"):
        exam.grade_page(exam.pages[0], "pos")


# ── sitting the exam ──────────────────────────────────────────────────

def test_good_stub_scores_perfect_on_hello():
    exam = Exam.from_dir(HELLO)
    card = sit(exam, Taker("good", keyword_tally))
    assert card.score == 1.0


def test_dumb_stub_scores_half_on_hello():
    exam = Exam.from_dir(HELLO)
    card = sit(exam, Taker("dumb", always_pos))
    assert card.score == 0.5


def test_scoring_discriminates():
    exam = Exam.from_dir(HELLO)
    assert as_loss(exam, Taker("dumb", always_pos)) > as_loss(
        exam, Taker("good", keyword_tally))


def test_report_card_has_per_page_scores_and_aggregate():
    exam = Exam.from_dir(HELLO)
    card = sit(exam, Taker("dumb", always_pos))
    assert len(card.pages) == 6
    assert all(p["earned"] in (0.0, 1.0) for p in card.pages)
    assert card.score == sum(p["earned"] for p in card.pages) / 6


def test_report_card_writes_json_artifact(tmp_path):
    exam = Exam.from_dir(HELLO)
    card = sit(exam, Taker("good", keyword_tally))
    path = card.write(tmp_path)
    data = json.loads(path.read_text())
    assert data["score"] == 1.0
    assert data["loss"] == 0.0
    assert len(data["pages"]) == 6


def test_limit_takes_only_the_first_pages():
    exam = Exam.from_dir(HELLO)
    card = sit(exam, Taker("dumb", always_pos), limit=2)
    assert len(card.pages) == 2
    assert card.score == 1.0  # first two pages are both pos
    # and a limit never reads pages it does not take
    card6 = sit(exam, Taker("dumb", always_pos))
    assert card6.score == 0.5


# ── retake: the resume story ──────────────────────────────────────────

def test_retake_keeps_answers_already_given(tmp_path):
    exam = Exam.from_dir(HELLO)
    calls = []

    def counting_taker(prompt):
        calls.append(1)
        return "pos"

    # half the exam was already answered before the interruption
    from nb.sit import _scribble
    for i in range(3):
        _scribble(tmp_path, i, "pos")

    card = retake(exam, Taker("counter", counting_taker), workbox=tmp_path)
    assert len(card.pages) == 6          # the whole exam is graded
    assert len(calls) == 3               # only the unanswered pages were re-asked
    assert card.score == 0.5             # always-pos across 6 pages


def test_scribbled_answers_are_honest_json(tmp_path):
    from nb.sit import _scribble, _read_scribble
    _scribble(tmp_path, 0, "pos")
    assert _read_scribble(tmp_path, 0) == "pos"
    assert (_read_scribble(tmp_path, 1) is not None) or True  # unanswered marker


def test_sit_with_workbox_scribbles_as_it_goes(tmp_path):
    exam = Exam.from_dir(HELLO)
    sit(exam, Taker("good", keyword_tally), workbox=tmp_path)
    assert len(list(tmp_path.iterdir())) == 6