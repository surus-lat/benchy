"""nb — the exam engine.

Bare metal of a benchmark, in plain exam words:

    an Exam  is question.json + cases.json + answer_key.json —
             three data files: what must be produced, the pages,
             and how each page is graded.
    a taker  is anyone who can answer: a name + answer(prompt) -> answer.
    To sit the exam is to take it; grading produces a ReportCard.
    The ReportCard IS the loss: score = how well you did; loss = 1 - score.
"""