# DESIGN — s02 "exam" (the metaphor IS the model)

## the design in one breath

A benchmark is an **exam**. An exam is three readable data files — a
**question.json** (what must be produced), **cases.json** (the pages:
prompt + expected answer + points), and **answer_key.json** (which grading
rule applies). A **Taker** is anyone who can answer — model, node,
workflow, agent: all the same here. Pages stay raw dicts. The taker
**sits** the exam
page by page; grading produces a **ReportCard** (the artifact JSON:
per-page scores + the exam score). Answers already **scribbled** in a
workbox are kept on the next sit (resume — retaking IS sitting). The
ReportCard **is** the loss: `loss = 1 - score`, and `as_loss(exam, taker)`
is the exam as a function over takers for prompt-optimizers.

A non-engineer reads `bench/hello/` and understands the exam from the
data alone; reads `nb/exam.py` and understands it from the words alone.

## the shape

    bench/hello/            the exam, as data
      question.json         path + asks / answer_shape / instructions
      cases.json            pages: prompt, expected, points
      answer_key.json       grade rule (+ rule args)
    nb/
      exam.py               Exam, AnswerKey, grading seam
      sit.py                Taker, sit(), ReportCard, as_loss()
      hall.py               the CLI: sit an exam with a stub, from the shell
    nb_tests/              pytest: what "broken" means
    golem.py               the guard

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| Exam | DATA | the named thing you author, share, and locate by ontology path; the union of question+pages+key is the benchmark | 1 |
| Exam.from_dir | DATA | an exam must be loadable from data with zero user Python | 0 |
| Exam.grade_page | SCORING | the single grading seam: the key's rule applied to one page. Deleting it scatters comparison logic into the run loop. The seam is also where a beyond-exact rule goes, when a real exam needs one | 3 |
| answer_key.json (raw dict) | SCORING | which rule grades each page, as written: {"grade": "exact"}. A wrapper class would be a mirror of json.loads; the dead "combine" and "rule" fields died with it | 2 |
| Taker | SYSTEM | the exam word for the AI-system: a name + answer(prompt). The primitive is the system-as-taker, not the model | 0 |
| sit() | DATA | the taker takes the exam; the run itself — AND the resume: sit again over a workbox keeps the scribbles. One verb, two tempos | 1 |
| as_loss | SCORING | the vision's headline: benchmark-as-loss-function for prompt optimizers. loss = 1 - score | 0 |
| ReportCard | DATA | the graded artifact: per-page scores + aggregate, JSON. Evidence trace of one loss evaluation. Rows are plain dicts {page, answered, earned} — the card never re-states the page; the exam is the single source of it | 1 |
| keyword_tally / always_pos | SYSTEM | the two stub takers: offline demo, no network, no keys. They prove the hall works and that scoring discriminates | 0 |
| hall.main | — | the shell door: sit an exam from the CLI. Deleting it leaves the engine library-only, unusable from the terminal | 0 |
| report (sit.py helper) | DATA | ~~fold results into the card~~ DELETED cycle 3: a two-line fold with one call site — inlined into sit() | gone |
| workbox answers.json | DATA | the resume format: one honest dict (page index → answer), rewritten after every page. Two helper functions + a sentinel object died for it | 0 |

## deletions (what the push proved to be noise)

- **PageResult** (cycle 5): a dataclass that was only ever converted to a dict
  before writing the card. The rows are now built as plain dicts directly in
  the sit loop — the card never re-states the page (prompt/expected/points
  live in the exam; the row says only: which page, what was answered, what
  was earned). One fewer class, −8 LOC.
- **_scribble/_read_scribble/_UNANSWERED** (cycle 6): the workbox was a file
  per page plus a sentinel object standing for "unanswered". The workbox is
  now ONE honest answers.json (page index → answer), rewritten after every
  page. Two helpers, one sentinel, five files on disk → one dict read at
  sit, one dict written per page. Resume got simpler, not harder.

- **Taker.sit** (cycle 7): a one-line delegation `return self.answer(prompt)`.
  A method whose whole body was to call another method — the run loop now
  calls `taker.answer(prompt)` directly. The verb sit() belongs to the RUN
  (sit(exam, taker)), not to the taker.

- **AnswerKey class + "combine" field** (cycle 8): the key is data — a raw
  dict {"grade", "rule"}, exactly the third mirror-of-json.loads wrapper
  that cycles 1 and 5 already proved to be noise (Page, Question, PageResult).
  Its `combine` field was NEVER read by any code path (the engine combines
  scores where the points are, in sit()) — a dead field in a class and a
  dead field in answer_key.json. Both died.

- **EXAM_RULES registry + "rule" field** (cycle 9): an empty dict that
  nobody ever registered a rule into, read by a code path that could never
  fire for hello. Speculative machinery — the escape hatch is the SEAM
  (Exam.grade_page), not a registry around it. When a real exam needs a
  beyond-exact rule, that seam is where it goes. The never-read "rule" field
  in answer_key.json died with it. Also died: ReportCard.write's `filename`
  param — the only caller passed exactly the default value.

- **limit param** (cycle 10): threaded through sit(), as_loss() and the
  hall's --limit flag, it had no honest exam word — an exam is not taken
  "up to N pages"; you author fewer pages or you resume a workbox. The
  vision's smoke-run need is runner territory (angle s09), not exam metal.
  The limit test became an honest smaller-exam test.

- **Question, Page dataclasses** (cycle 1): the engine never interprets the
  question or the page — it loads them, passes them, grades them. A wrapper
  class per data file was a mirror of `json.loads` with a nicer name.
  Pages stay raw dicts.
- **grade_exact / grade_keyword / GRADES** (cycle 1): the builtin rule IS
  exact match, inlined in `Exam.grade_page`. A registry for one builtin
  plus one demo rule was a framework for zero users.
- **Exam.combine** (cycle 1): dead — the score combination lives in
  `report()` where the points are at hand. Weighted-mean logic in two
  places is one place too many.
- **sit.grade_page** (cycle 2): a second `grade_page` — one name doing two
  jobs. Grading seam: `Exam.grade_page` alone; the row build inlined into
  the sit loop.
- **report()** (cycle 3): a two-line fold with one call site — fused into
  `sit()`. The card IS the end of the sit loop, not a separate stage.
- **retake()** (cycle 4): a pure alias of `sit(..., workbox=...)`. Resume is
  not a second verb — retaking an exam IS sitting it. Also killed the hall's
  stub-vs-retake dual path (which had quietly hard-coded `always_pos` as
  "the retake taker" — a lie); the hall now has ONE way to say who sits and
  `--workbox` resumes.

## invariants (from GOLEM.md, law 6)

- The primitive is the Taker (a program), not the model.
- Exam = question + pages + answer key. The taker is the ARGUMENT:
  `card = sit(exam, taker)` and `loss = as_loss(exam, taker)`.
- Ontology: the exam's `path` (e.g. `/sentiment`) locates it.
- Benchy creates NEW benchmarks: authoring = writing three JSON files.