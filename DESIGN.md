# DESIGN — s02 "exam" (the metaphor IS the model)

## the design in one breath

A benchmark is an **exam**. An exam is three readable data files — a
**question.json** (what must be produced), **cases.json** (the pages:
prompt + expected answer + points), and **answer_key.json** (which grading
rule applies). A **Taker** is anyone who can answer — model, node,
Pages stay raw dicts. The taker **sits** the exam
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
| Exam.grade_page | SCORING | the single grading seam: the key's rule applied to one page. Deleting it scatters comparison logic into the run loop | 1 |
| EXAM_RULES | SCORING | the Python escape hatch for grading rules beyond exact — a benchmark stays pure data; only new RULES need Python | 0 |
| AnswerKey | SCORING | which rule grades each page. Deleting it merges grading policy into pages, hiding what "good" means | 0 |
| Taker | SYSTEM | the exam word for the AI-system: a name + answer(prompt). The primitive is the system-as-taker, not the model | 0 |
| sit() | DATA | the taker takes the exam; the run itself — AND the resume: sit again over a workbox keeps the scribbles. One verb, two tempos | 1 |
| as_loss | SCORING | the vision's headline: benchmark-as-loss-function for prompt optimizers. loss = 1 - score | 0 |
| ReportCard | DATA | the graded artifact: per-page scores + aggregate, JSON. Evidence trace of one loss evaluation | 0 |
| PageResult | DATA | one row of the report card: what was asked, answered, earned | 0 |
| keyword_tally / always_pos | SYSTEM | the two stub takers: offline demo, no network, no keys. They prove the hall works and that scoring discriminates | 0 |
| hall.main | — | the shell door: sit an exam from the CLI. Deleting it leaves the engine library-only, unusable from the terminal | 0 |
| report (sit.py helper) | DATA | ~~fold results into the card~~ DELETED cycle 3: a two-line fold with one call site — inlined into sit() | gone |
| _scribble/_read_scribble | DATA | workbox I/O: answer per page saved as soon as produced. The honesty of retake | 0 |

## deletions (what the push proved to be noise)

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