# LEARNINGS — s02 "exam" (the metaphor IS the model)

15 cycles. 214 → 162 LOC, 15 → 7 concepts, 4 files, 0 deps.
Verdicts: HARD_PUSH ×10, NOISE_REMOVED ×4, BARE_METAL ×1.

## the verdict on the angle: SURVIVED

The hypothesis was: name every concept with exam words; if a needed
concept has no honest exam word, suspect it is noise. That is exactly
what happened, 15 times out of 15:

- No exam word ever had to LIE. The words that survived — exam, page,
  question, answer key, taker, sit, report card, workbox, hall — are all
  doing the same work in the code as they do in the metaphor. The one
  "Scheduler"-like smell we caught early (retake as a second verb) died
  precisely because retaking an exam IS sitting it — the metaphor itself
  voted for deletion.
- Every concept with NO honest exam word died: `limit` (an exam is not
  taken "up to N pages"), `--out` (where a card lands is shell
  territory), `taken_at` (when you sat is runner metadata), `PageResult`,
  `Taker` (the class — the word is alive and honest).
- The falsification test was the metaphor stretching. It never did.
  Where it COULD have stretched (proctor? scheduler? registry?) the
  golem killed the concept first.

## pillar by pillar: the bare metal under the metaphor

### TASK — a file nobody's code reads

question.json is authored, shared, and read by takers and humans — and
the engine NEVER reads it. Cycle 13 deleted the Exam.question field:
carrying it duplicated a file no code path touched. The bare metal of
TASK is: one data file that declares the input/output shape, existing
to be read by the thing that will answer, not by the thing that grades.
The engine's only TASK knowledge is `path` (the ontology locator).

### SCORING — a raw dict + one seam

answer_key.json is a raw dict ({"grade": "exact"}) — the AnswerKey class
died in cycle 8 as the third mirror-of-json.loads. The whole scoring
pillar is TWO things: the dict as written, and `Exam.grade_page` — one
seam where the key's rule meets a page. No registry (the empty
EXAM_RULES died in cycle 9): when a real exam needs a rule beyond
exact, the seam is where it goes, and not one line before.
The hierarchy-of-importance lives in `points` on each page — the
weighted mean is computed where the points are, in sit(). Score IS
loss: `loss = 1 - score`, and as_loss is three lines.

### DATA — three files, one loader, raw dicts

An exam is a directory: question.json + cases.json + answer_key.json.
Pages stay raw dicts forever — Page, Question, PageResult, AnswerKey,
Taker: five wrapper classes died, all the same death (a mirror of
json.loads with a nicer name). One loader (Exam.from_dir) is BARE METAL:
cycle 15 gutted it, inlined it into callers, 13/13 tests passed — and
the vision broke anyway, because the format knowledge leaked out of the
engine and running a benchmark became user Python again. The loader is
the load-bearing brick of "a benchmark is data, never required Python".

### SYSTEM — an argument, not a class

A taker is a name + an answer(prompt) callable. The primitive is the
system-as-taker, not the model — model, node, workflow, agent are all
the same shape here, and that shape is two function arguments:
`card = sit(exam, name, answer)`. The Taker class died (cycle 14) as a
name for a pair of args. The stubs (keyword_tally, always_pos) are
SYSTEM demo fixtures, not engine concepts.

## what the metaphor bought, what it cost

BOUGHT:
- a deletion oracle: "does this have an honest exam word?" killed
  limit, --out, taken_at, PageResult, retake, and the Taker class —
  six deletions a spec-only search would likely have kept.
- a self-explaining codebase: sit/score/report-card need no docs.
- resume for free: "sit again over a workbox" needed no new concept.

COST:
- two words did double duty: "page" is both a case and a card row
  (row["page"] is an index — we say "page 3" for both the exam page and
  its row; harmless at this scale, worth renaming at larger scale).
- "hall" is a stretch: it is just main(). An honest name might be
  "cli.py", but a CLI has no exam word and we kept the metaphor. Cheap.

## the pattern (three deaths, one law)

1. Mirror-of-json.loads classes die: Question, Page, PageResult,
   AnswerKey, Taker — every wrapper around what json.loads already
   returned. If your class has no behavior the dict lacks, it is noise.
2. No-honest-exam-word concepts die: limit, --out, taken_at, retake.
   When the metaphor cannot name it, the design does not need it.
3. Synonyms die: report vs sit (the card is the END of the sit loop),
   retake vs sit, Taker.sit vs answer. One verb, one noun, per act.

## advice for the runner angle (s09)

Our workbox is the minimal honest resume shape: ONE answers.json
(page index → answer), rewritten WHOLE after every page. Not a file per
page, not a sentinel for unanswered, not a status enum — a dict, read
at sit start, written per page. Lessons:
- Resume is not a verb of its own: retake() died because sitting
  again over the workbox IS resuming. Keep ONE entry path.
- The s09 artifact contract (run_outcome.json's spirit) should stay a
  projection of the sit loop, not a peer concept — timestamps and
  exit codes belong to the runner because WHEN you sat and HOW MANY
  pages survived a kill are runner facts, not exam facts (that is why
  taken_at died here).
- The dumb stub scoring 0.5 is the cheapest possible scoring test:
  keep one taker that cannot fail and one that cannot pass, and the
  scoring discriminates by construction.
- from_dir is load-bearing for you too: your runner will call it. Do
  not "simplify" the load boundary — cycle 15 proved it bare metal.