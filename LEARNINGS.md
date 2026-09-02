# s08 LEARNINGS — the bare metal, pillar by pillar

The CLI-first angle asked: is the product three commands?  The search's
answer, earned by deletion: **two verbs + one flag absorbed the whole vision
loop** (create → run → grade → export as loss), and the engine underneath
shrank to 89 LOC / 4 files / 4 concepts / 0 deps, stdlib only.  The
three-command hypothesis is falsified — `report` was noise wearing a product
word.

## TASK (the program: input → output)

- The task lives in **benchmark.json as taker-facing data** (`task: {}`), and
  the ENGINE never touches it.  Cycle 1 deleted the engine's own task
  plumbing (Exam.task-param): it was a lens — everything it carried was
  derivable from the benchmark's address.  The cloud compiler (steering
  addendum) reads task to build prompts; that is the SYSTEM pillar's
  business, not the engine's.
- Presence is enforced, semantics stay FREE.  The scaffold teaches a blank
  `{}` because any typed example teaches a type the engine does not check —
  the scaffold task-lie (c9, law (b)).

## SCORING

- **Scoring derives from the output schema** (IDEAS.md, confirmed c11): exact
  match ← the output enum; 1 point per case; exam score = mean.  The
  configurable-scorer constructor param died — no caller ever passed a
  different scorer; the seam was speculative.  When a real exam needs
  weights, the DATA format grows a scoring key LOUDLY (the exact-set check
  will demand it), never a silent Python seam.
- **The loss is composed ONCE, by grading** (c13).  The CLI once computed
  `1 - score` privately for its ack line while as_loss() computed it again
  and the report test a third time — three addresses of one formula.  Now
  grading owns the formula, the artifact carries the loss, and every reader
  (as_loss, stdout, any external program reading runs/*.json) READS it.
- `as_loss()` is a projection, not an implementation: it reads the graded
  artifact's loss field.  Lower is better; `loss(dumb) > loss(good)` pins
  the ranking.

## DATA (the exam)

- **benchmark.json is EXACTLY `{task, cases}`** — enforced by an exact-set
  check at load (c9).  One message names missing AND junk keys.
- **The DIRECTORY is the ontology address** (c6): the vision's
  `/<task?>/<domain?>/<language?>` is literally the filesystem under bench/.
  The `path` data field died — a second address; locate's rglob walk existed
  only to reconcile it with the directory, and a sibling benchmark could
  break an unrelated lookup.
- **The per-case contract is enforced at LOAD** (c14): a case is at least
  `{input, expected}`; extra keys (context, id) are the taker's data and pass
  through to the artifact.  Before this, a malformed case died MID-EXAM —
  cloud money already spent — in a cryptic `KeyError: 'expected'` traceback,
  while the scaffold's ack taught the exact shape the engine never checked.
  Data errors refuse at load, in words; taker errors still crash honestly.
- Zero cases refuse at load too (loud, before anything runs).
- `new` scaffolds the whole thing as pure data (GOLEM law 5): one file,
  immediately runnable (zero cases → loud refusal), refuses overwrite
  (test-pinned, c14).  CREATING benchmarks is the product's first step
  (VISION p.2) — c15 proved the verb itself metal: deleting it broke the
  two tests pinning the create-loop.

## SYSTEM (the exam-taker)

- **A system is any invoked program** (GOLEM law 4): `invoke(input) ->
  prediction`.  systems.py sits NEXT TO the benchmark but is NOT part of it —
  stubs today; the cloud taker (steering addendum) joins here as a spec +
  compiler, long-term work, not engine code.
- **The binding seam is one runpy line** (c12): importlib's 3-line ceremony
  (spec + module + exec) bought a module OBJECT whose only use was getattr.
  The taker is data addressed by name, not a Python module identity.
- The system is the ARGUMENT: `result = benchmark.run(system)`,
  `loss = benchmark.as_loss()(system)` — GOLEM law 6's own spelling, which is
  why the Exam class survived dissolution (c5 BARE_METAL: the 3-tuple leaked
  the exam's internal shape to every caller).

## the CLI (the fifth pillar, this angle's own)

- **Metal verbs**: `run` (the product itself — take the exam, write evidence)
  and `new` (data-scaffold; CREATING benchmarks is the focus per VISION).
  **Noise verb**: `report` — deleted c4; the graded artifact IS the report
  (JSON, interprets alone, pinned by its own test), and the vision loop has
  no report word in it.
- **Metal flag**: `--limit N` — the smoke valve against cloud spend (the old
  benchy's entire smoke workflow, AGENTS.md, reduced to one flag; a money
  justification, not convenience).  c10 made it honest: the artifact carries
  `total`, so a smoke run can never masquerade as a full run, and `--limit 0`
  grades nothing and says so (`benchy: division by zero`) instead of
  silently running the whole exam — the old falsy-check's lie.
- **The three-command hypothesis is falsified**.  Every other shape the
  vision demands arrived as a word, not a flag.  The whole UX fits in a
  tweet: `benchy new <name>` · `benchy run <bench> <system> [--limit N]`.
- **main() dissolved** (c8): dispatch is module code in __main__.py; the
  if/elif over two verbs IS the verb table; tests drive the real process via
  subprocess, so the argv testability seam was dead weight.
- **The graded artifact is all evidence** (c15, per-field deletion): `loss`
  (8 tests read it), `total` (scope honesty), `system`/`benchmark`
  (interprets-alone identity), per-case `{input, expected, prediction,
  score}`.  No write-only fields.  stdout carries nothing the artifact
  doesn't.  `expected` is the right name — one name, one address: enforced
  at load, read at scoring, taught by the scaffold's ack.

## the two laws (this tree's own, for the unify phase)

**Law (a) — every deletion that broke revealed a SECOND ADDRESS.**  Five
instances:
1. artifact `dir.name` (c3): the ontology path was already the address.
2. benchmark.json's `path` field (c6): the directory already was the address;
   the walk existed only to reconcile the two.
3. filename-as-identity (c7): a JSON that leans on its filename does not
   interpret alone — identity belongs IN the artifact.
4. the loss formula ×3 (c13): grading's composition, the CLI's private
   `1 - score`, and the report test's re-derivation — three addresses of one
   formula; grading owns the one.
5. command docstrings duplicating the module docstring (c14): the module
   docstring IS the usage text __main__ prints; repeating it per function
   taught the same words twice.

**Law (b) — every probe that lied was a claim the engine never enforced.
Test-first exposed all four:**
1. scaffold shape (c9): "an exam is {task, cases}" was a docstring claim
   until a test-first invariant failed against the engine → exact-set check.
2. limit honesty (c10): `total` was claimed-implicit, absent from the
   artifact — KeyError, test-first → a smoke run can no longer masquerade.
3. per-case contract (c14): speak-words was claimed but a case missing
   `expected` died mid-exam in a traceback → load-time enforcement (cloud
   money).
4. overwrite refusal (c14): existed in code, NO test pinned it — an
   unenforced claim until pinned.

## Exam-as-concept-compressor (s06 replication)

The Exam class survived full dissolution (c5 BARE_METAL) because it is the
concept-compressor: it carries the vision invariant's own syntax —
`benchmark.run(system)`, `benchmark.as_loss()` (GOLEM law 6) — and hides the
exam's internal shape (a 3-tuple leaked that shape to every caller:
destructure + re-wrap + alias collision).  Its whole state is ONE field,
`cases` (c11) — one field, one concept.  Grading composes loss; identity is
the caller's (the write, c7).

## the shape that landed

```
benchy new <name>          -> bench/<name>/benchmark.json {task:{}, cases:[]}
benchy run <bench> <sys> [--limit N]
                           -> runs/<path>-<sys>.json
                              {system, benchmark, total, cases:[{input,
                              expected, prediction, score}], score, loss}
```

89 LOC, 4 files, 4 concepts (Exam, locate, new, run), 0 deps, stdlib only,
16 tests.  `python -m nb` until packaging earns the console script.