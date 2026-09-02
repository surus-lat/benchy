# SUMMARY — s02 "exam" (the metaphor IS the model)

## final metrics (golem report, verbatim)

{
  "cycles": 15,
  "verdicts": {
    "HARD_PUSH": 10,
    "NOISE_REMOVED": 4,
    "BARE_METAL": 1
  },
  "files": 4,
  "loc": 162,
  "deps": 0,
  "concepts": 7
}

## final shapes

    bench/hello/            the exam, as data (zero Python)
      question.json         path + asks / answer_shape / instructions
      cases.json            pages: prompt, expected, points
      answer_key.json       grade rule: {"grade": "exact"}
    nb/
      exam.py               Exam, from_dir, grade_page (the grading seam)
      sit.py                sit(exam, name, answer), ReportCard, as_loss
      hall.py               the shell door: sit from CLI, stub takers
    nb_tests/              13 tests: what "broken" means
    golem.py               the guard

    card = sit(exam, name, answer, workbox=None)   # run + resume, one verb
    loss = as_loss(exam, name, answer)            # 1 - score
    exam = Exam.from_dir("bench/hello")           # bare metal (cycle 15)

7 concepts: Exam, from_dir, grade_page, sit, ReportCard, as_loss,
hall.main (+ taker as a plain string/callable pair, stubs as fixtures).

## best discovery

The metaphor is a DELETION ORACLE. "Does this concept have an honest
exam word?" killed limit, --out, taken_at, PageResult, retake, and the
Taker class — six deletions a metaphor-free search would probably have
kept. And the oracle runs in both directions: from_dir survived because
deleting it made running-a-benchmark user Python again — the first and
only BARE_METAL verdict of the 15 cycles, earned on the vision
invariant, not on convenience.

## most expensive mistake

Cycles 1-6 built wrapper classes (Question, Page, PageResult,
AnswerKey, Taker) around what json.loads already returned. Five of the
six died as "mirrors of json.loads". If we had started with raw dicts +
two functions, six cycles would have been free for harder questions.
Lesson: NEVER wrap data the engine does not interpret.

## what I would tell the other nine searchers

1. Wrapper classes around parsed data are the loudest first noise.
   Delete them in cycle 1, not cycle 8.
2. The vision invariants are your ONLY source of BARE_METAL verdicts.
   Tests alone never break small deletions — the vision does. Attempt
   from_dir's deletion to feel the difference between "tests pass" and
   "the product still exists".
3. Resume is one dict (page index → answer), rewritten whole per page,
   and resume is the SAME verb as run. Do not build two paths.
4. The grading seam is ONE method, no registry. Registries for zero
   users are speculative machinery — the seam is the escape hatch.
5. Timestamps, output paths, limits: runner/shell territory. If your
   angle has no honest word for it, it is s09's or the shell's, not
   yours. Give it away early.
6. A benchmark directory with three JSON files a non-engineer can read
   IS the whole DATA pillar. The loader is load-bearing; the engine
   reading the question file is NOT (the taker reads it).