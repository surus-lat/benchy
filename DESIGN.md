# DESIGN — s02 "exam" (the metaphor IS the model)

## the design in one breath

A benchmark is an **exam**. An exam is three readable data files — a
**Question** (what must be produced), **Cases** (the pages: prompt +
expected answer + points), and an **AnswerKey** (which grading rule, how
points combine). A **Taker** is anyone who can answer — model, node,
workflow, agent: all the same thing here. The taker **sits** the exam
page by page; grading produces a **ReportCard** (the artifact JSON:
per-page scores + the exam score). **Retake** = resume: answers already
scribbled in the workbox are kept. The ReportCard **is** the loss:
`loss = 1 - score`, and `as_loss(exam, taker)` is the exam as a function
over takers for prompt-optimizers.

A non-engineer reads `bench/hello/` and understands the exam from the
data alone; reads `nb/exam.py` and understands it from the words alone.

## the shape

    bench/hello/            the exam, as data
      question.json         asks / answer_shape / instructions
      cases.json            pages: prompt, expected, points
      answer_key.json       grade rule + combine
    nb/
      exam.py               Exam, Question, Page, AnswerKey, grading rules
      sit.py                Taker, sit(), retake(), ReportCard, as_loss()
      hall.py               the CLI: sit an exam with a stub, from the shell
    nb_tests/              pytest: what "broken" means
    golem.py               the guard

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| Exam | DATA | the named thing you author, share, and locate by ontology path; the union of question+pages+key is the benchmark | 0 |
| Question | TASK | what the taker must produce: asks + answer_shape. Without it a page is just a bare dict and the task (the program we search for) has no home | 0 |
| Page | DATA | one case: prompt + expected + points. The exam's substance; n pages = n cases. `Page` is the exam word for a case | 0 |
| AnswerKey | SCORING | which rule grades each page and how points combine. Deleting it merges grading policy into pages, hiding what "good" means | 0 |
| grade_exact | SCORING | the hello bar demands 1 point per exact match | 0 |
| grade_keyword | SCORING | proves rules are data-addressable by name; the escape hatch for when exact is too strict | 0 |
| GRADES | SCORING | the registry that makes rules addressable by name from answer_key.json — a benchmark stays pure data | 0 |
| Exam.from_dir | DATA | an exam must be loadable from data with zero user Python | 0 |
| Exam.grade_page | SCORING | applies the key's rule to one page; the single grading seam | 0 |
| Taker | SYSTEM | the exam word for the AI-system: a name + answer(prompt). The primitive is the system-as-taker, not the model | 0 |
| Taker.sit | SYSTEM | how a taker answers one page. One method is the whole AI-API | 0 |
| sit() | DATA | the taker takes the exam; the run itself. Interruptible via workbox scribbles | 0 |
| retake() | DATA | resume: keep scribbled answers, re-ask the rest. The vision's resume story in one word | 0 |
| as_loss | SCORING | the vision's headline: benchmark-as-loss-function for prompt optimizers. loss = 1 - score | 0 |
| ReportCard | DATA | the graded artifact: per-page scores + aggregate, JSON. Evidence trace of one loss evaluation | 0 |
| PageResult | DATA | one row of the report card: what was asked, answered, earned | 0 |
| keyword_tally / always_pos | SYSTEM | the two stub takers: offline demo, no network, no keys. They prove the hall works and that scoring discriminates | 0 |
| hall.main | — | the shell door: sit an exam from the CLI. Deleting it leaves the engine library-only, unusable from the terminal | 0 |
| grade_page / report (sit.py helpers) | DATA | grade one page / fold results into the card. Fused into sit.py as the grading seams; public only because the golem counts names | 0 |
| _scribble/_read_scribble | DATA | workbox I/O: answer per page saved as soon as produced. The honesty of retake | 0 |

## invariants (from GOLEM.md, law 6)

- The primitive is the Taker (a program), not the model.
- Exam = question + pages + answer key. The taker is the ARGUMENT:
  `card = sit(exam, taker)` and `loss = as_loss(exam, taker)`.
- Ontology: the exam's `path` (e.g. `/sentiment`) locates it.
- Benchy creates NEW benchmarks: authoring = writing three JSON files.