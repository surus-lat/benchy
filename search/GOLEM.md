# THE BARE-METAL GOLEM — law of the search

This file is the constitution of the redesign search. VISION.md and IDEAS.md
are the WHY. This file is the HOW. If they ever conflict, the vision wins.

## what "bare metal" means

Not the smallest code — the minimum set of concepts the vision requires, with
zero noise. A concept earns the name bare metal ONLY by surviving a serious
attempt to delete it. Everything else is noise.

Benchy has exactly four pillars (IDEAS.md):

1. TASK — the description of the program we are searching for: the
   input/output schema. "takes a pdf, returns extracted_fields[json]".
2. SCORING — the grading function. What "good" means, including the
   hierarchy of importance between fields. It is also the loss.
3. DATA — the exam: n cases. The system takes the exam; we get graded
   evidence. Everything is a distribution; we report point estimates.
4. SYSTEM — the exam taker: an AI-API / compiler for learned programs.
   Model, node, workflow, agent: all the same thing here — it is invoked,
   it produces a prediction.

Any line of code not serving one of these four pillars is noise.

## the law

1. Every push cycle must attempt to break something. If nothing broke, the
   push was TOO_SOFT — attempt a BIGGER deletion in the same cycle. Fear of
   breaking is the enemy; a redesign that breaks nothing proved nothing.
2. Metrics must not grow: files, lines, external deps, public concepts.
   Growth needs an explicit `python3 golem.py check --allow-growth "<why>"`.
3. Every public concept must appear in DESIGN.md's concept table with: the
   pillar it serves, one line on why it cannot be deleted, and how many
   deletion attempts it survived. No entry = noise = delete it or justify it.
4. The engine is stdlib-only. Tests may use pytest. Nothing else, ever.
5. A benchmark is DATA (yaml/json/…), never required Python. Python is the
   escape hatch, not the interface.
6. Vision invariants, unbreakable:
   - The primitive is the AI-system (a program), not the model.
   - Benchmark = task + data + scoring. The system is the ARGUMENT:
     `result = benchmark.run(system)` and `loss = benchmark.as_loss()`.
   - Ontology: `/<task?>/<domain?>/<language?>`.
   - Benchy is for CREATING new benchmarks, not for serving existing ones.
7. The old benchy (benchy/, src/, tests/ inside this worktree) is OFF
   LIMITS. From zero means from zero. (Sole exception: angle s10, final
   cycles, reading the main checkout only — see ANGLES.md.)

## the push cycle (one iteration)

1. Pick the loudest thing: the most complex concept, the biggest file, the
   most suspicious abstraction, the flag you cannot explain.
2. Attempt to delete it or fuse it into another concept.
3. Run the tests, then `python3 golem.py check`.
4. Verdict:
   - broke → you fixed forward = the tests guarded noise → NOISE_REMOVED
   - broke → you restored = the concept is bare metal → BARE_METAL
     (write in DESIGN.md exactly WHY it is essential)
   - nothing broke → TOO_SOFT: do NOT stop. Attempt a bigger deletion in
     this same cycle until something breaks or the thing is gone entirely.
     If the escalation ended in a deletion or a break, the cycle's verdict
     is HARD_PUSH (a too-soft opening recovered by a harder push).
5. Log ONE line in ITERATIONS.md:
   `<n> | pushed=<target> | broke=yes|no | verdict=<NOISE_REMOVED|BARE_METAL|HARD_PUSH|TOO_SOFT> | loc=<n> | concepts=<n>`
6. Update DESIGN.md (concept table, design text if the shape changed).

A cycle where nothing broke AND nothing was deleted does not count.

## the hello benchmark — acceptance bar (identical exam for all 10)

Your engine must run this offline, end to end, before cycle 10:

  task: sentiment classification
  input: one short text (string)
  output: exactly one of {pos, neg}
  cases (exactly these 6):
    ("this works great", pos) ("excelente servicio", pos) ("loved it", pos)
    ("broken on arrival", neg) ("una porqueria", neg) ("never again", neg)
  scoring: 1 point per exact match, exam score = mean
  systems: two stubs, no network, no keys:
    - good stub: keyword heuristic (great|excelente|loved → pos) → scores 1.0
    - dumb stub: always pos → scores 0.5
  must-haves:
    - the benchmark is defined as data, locatable by its ontology path
      /sentiment (the format itself is your angle's to design)
    - a graded artifact: JSON with per-case scores + the aggregate
    - as_loss() exists and ranks the stubs: loss(dumb) > loss(good)
    - the dumb stub scoring 0.5 proves the scoring discriminates

## deliverables (worktree root)

  DESIGN.md       design + concept table: concept | pillar | why undeletable | survived N
  ITERATIONS.md   the golem log, one line per cycle
  LEARNINGS.md    what the bare metal actually is, pillar by pillar
  SUMMARY.md      ≤ 50 lines for the unify phase: final shapes + metrics,
                  best discovery, most expensive mistake, what you would
                  tell the other nine searchers
  nb/             the engine (this exact dir name, so unify can compare)
  nb_tests/       the minimal pytest suite that defines "broken"
  bench/hello/    the hello benchmark in YOUR format, runnable offline
  golem.py        your copy of the guard; you may sharpen it, never weaken it
  .golem_state.json

Commit to your branch every ~3 cycles. Tags: [ADD] [MOD] [REM] [DOC].
The golem is watching you.