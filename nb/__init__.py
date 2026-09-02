"""nb — the exam engine.

Bare metal of a benchmark, in plain exam words:

    an Exam is  a Question (what the taker must produce)
              + Cases      (the pages of the exam)
              + an AnswerKey (how each page is graded)
    a Taker  is anyone who can sit the exam: answer(question, case) -> answer.
    To sit the exam is to take it; grading produces a ReportCard.
    The ReportCard IS the loss: score = how well you did; loss = 1 - score.
"""