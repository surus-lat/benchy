# DESIGN — s02 "exam" (the metaphor IS the model)

## the design in one breath

A benchmark is an **exam**. An exam is three readable data files — a
**question.json** (what must be produced), **cases.json** (the pages:
prompt + expected answer + points), and **answer_key.json** (which grading
rule applies). A **taker** is anyone who can answer — model, node,
workflow, agent: all the same here, and all the same *shape*: a name
plus an `answer(prompt)` callable. Pages stay raw dicts. The taker
**sits** the exam
page by page; grading produces a **ReportCard** (the artifact JSON:
per-page scores + the exam score). Answers already **scribbled** in a
workbox are kept on the next sit (resume — retaking IS sitting). The
ReportCard **is** the loss: `loss = 1 - score`, and `as_loss(exam, name, answer)`
is the exam as a function over takers for prompt-optimizers.

A non-engineer reads `bench/hello/` and understands the exam from the
data alone; reads `nb/exam.py` and understands it from the words alone.

## the shape

    bench/hello/            the exam, as data
      question.json         path + asks / answer_shape / instructions
      cases.json            pages: prompt, expected, points
      answer_key.json       grade rule
    nb/
      exam.py               Exam, from_dir, grade_page (the grading seam)
      sit.py                sit(exam, name, answer), ReportCard, as_loss
      hall.py               the hall: sit from the shell, with a stub
    nb_tests/              pytest: what "broken" means
    golem.py               the guard

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| Exam | DATA | the named thing you author, share, and locate by ontology path; the union of question+pages+key is the benchmark. The question stays on disk (engine never reads it); Exam carries pages+key | 1 |
| Exam.from_dir | DATA | **BARE METAL (cycle 15)**: deleted and inlined into its callers, 13/13 tests still passed — but the exam-directory FORMAT (three file names + the q["path"]/c["pages"] surgery) leaked into every caller's Python. Running a benchmark would again require user Python; a runner or a prompt-optimizer importing as_loss would each re-author the loader. The loader IS the "benchmark = data, zero user Python" invariant's one load-bearing brick. Restored byte-identical | 1 |
| Exam.grade_page | SCORING | the single grading seam: the key's rule applied to one page. Deleting it scatters comparison logic into the run loop. The seam is also where a beyond-exact rule goes, when a real exam needs one | 3 |
| answer_key.json (raw dict) | SCORING | which rule grades each page, as written: {"grade": "exact"}. A wrapper class would be a mirror of json.loads; the dead "combine" and "rule" fields died with it | 2 |
| taker (name + answer callable) | SYSTEM | the exam word for the AI-system: who sits, and the `answer(prompt)` callable they sit with. The primitive is the system-as-taker, not the model. The word survives as `taker: str` on the card and a `name` arg; the class was noise — a name field grouping two args (cycle 14) | 1 |
| sit() | DATA | the taker takes the exam; the run itself — AND the resume: sit again over a workbox keeps the scribbles. One verb, two tempos | 1 |
| as_loss | SCORING | the vision's headline: benchmark-as-loss-function for prompt optimizers. loss = 1 - score | 0 |
| ReportCard | DATA | the graded artifact: per-page scores + aggregate, JSON. Evidence trace of one loss evaluation. Rows are plain dicts {page, answered, earned} — the card never re-states the page; the exam is the single source of it | 2 |
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
  "up to N pages"; you author fewer pages or you resume a workbox.
  The vision's smoke-run need is runner territory (angle s09), not exam metal.
  The limit test became an honest smaller-exam test.

- **ReportCard.taken_at** (cycle 11): a timestamp on the card. When the
  exam was sat is RUNNER metadata (which s09 owns: run_outcome.json
  already carries started_at/ended_at), not exam metal — the card says
  what was answered and what it earned, and that is the whole honest
  content of a graded page. No test read it, nothing in the acceptance
  bar needs a clock, and the `time` import died with it. −3 LOC.

- **hall --out flag** (cycle 12): the card goes to the working directory
  — where the shell already is. "Where the artifact lands" is shell
  territory (`cd`, `mv`), same argument that killed --limit; a flag whose
  only caller path was the default is a flag with no honest exam word.
  −2 LOC.

- **Exam.question field** (cycle 13): the engine never reads the question
  — it is read by the taker and the human author, as data, from
  question.json where it is already written. Carrying it as a field meant
  the Exam object duplicated a file no code path touched; the exam now
  LOCATES the question (from_dir reads it for `path`), it does not carry
  it. The three data files on disk are unchanged — an exam is still
  authored as question+cases+answer_key.

- **Taker class** (cycle 14): a dataclass whose fields were `name` and
  `answer` — a name for a pair of arguments. Its only behavior,
  `Taker.sit`, had already died (cycle 7); what was left was a two-field
  wrapper every caller had to construct before doing anything. `sit(exam,
  name, answer)` and `as_loss(exam, name, answer)` take the pair directly.
  The WORD survives — `taker: str` on the card, `name` on the sit — the
  class was noise. Fifth mirror-of-json.loads death in this search
  (Question, Page, PageResult, AnswerKey, now Taker).

- **Exam.from_dir — survived (cycle 15, BARE_METAL)**: the honest
  deletion was executed: from_dir gutted, the 4-line load inlined into
  hall.main, a `load_hello()` helper written in the tests, all 12 call
  sites migrated. 13/13 passed. But the pass was the leak: the format
  knowledge (which three files, and how their dicts become an Exam) now
  lived in TWO places outside the engine, and every future caller —
  a runner, a prompt-optimizer importing as_loss — would have to copy
  those 4 lines again. "A benchmark is data, never required Python"
  (GOLEM.md law 5) breaks at the load boundary: if the loader is not IN
  the engine, running the exam IS user Python. Restored byte-identical.
  The one concept this search proved bare-metal by breakage — and what
  it guards is the vision, not a convenience.

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

- The primitive is the taker (a program), not the model.
- Exam = question + pages + answer key. The taker is the ARGUMENT:
  `card = sit(exam, name, answer)` and `loss = as_loss(exam, name, answer)`.
- Ontology: the exam's `path` (e.g. `/sentiment`) locates it.
- Benchy creates NEW benchmarks: authoring = writing three JSON files.