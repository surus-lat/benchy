# LEARNINGS — s07 datacentric (15 cycles, golem --final PASS)

## The falsification that shaped the tree (c1)
The angle's maximal claim — derive the answer space AND the scoring from the
samples — broke in cycle 1 and the break was the lesson: **inference is a
cleverness engine**. An inferred enum absorbs `expected` typos as new classes
(it hides corruption), shrinks when a class goes unrepresented (the answer
space silently loses a legal output), and moves "what good means" from the
exam paper into code (scoring stops being data). The metal is the opposite:
**DECLARED beats INFERRED** — task lens and scoring policy live in exam.json,
loud-validated at load.

## Pillar by pillar
- **task**: a bare list of legal outputs. Not a dict with metadata, not an
  object — the wrapper was noise. Every sample's `expected` is validated
  against it at load (c1: drift dies at the door).
- **scoring**: the declared policy `{"match": "exact"}` is the metal. c13
  inlined the SCORE_KEYS indirection: the one policy IS the literal; any
  deviation is "not the policy" — the NAME was the indirection, the check is
  the metal. c14 proved reading ≠ checking: five literal `data["key"]` reads
  never see an UNKNOWN key; the schema check is what makes drift loud.
- **data (exam)**: ONE data file — `exam.json` = {path, task, scoring,
  samples, systems}. This is s07's evidence on the data-pillar divergence:
  s05 passed the same bar with a directory of JSONs (task.json / scoring.json
  / cases.jsonl / systems/). Both are honest; one file wins on "the exam is
  one artifact you can read in a sitting", a directory wins on diffs and
  per-file ownership. The UNIFY should not force one — the invariants are:
  declared lenses, loud checks, benchmark = data never required Python.
- **system (compiler)**: systems-as-data — ONE spec kind survived
  (keyword with `any=[]` subsumes constants, c7). invoke's dict/kind gate is
  BARE_METAL (c12): load guards DATA entry, invoke guards ARGUMENT entry — a
  spec handed straight to as_loss/run (a prompt-optimizer's candidate) never
  passes load. Two gates, two duties. Cloud specs land here as data.

## BARE_METAL badges (proven by deletion probes)
- **as_loss** (c5): the loss-export is the vision contract itself (law 6),
  not derivable noise — the headline feature for prompt-optimizers.
- **CLI** (c3): an engine only reachable via pytest is archaeology — the bar
  is offline end-to-end runnable by a person.
- **probe ≠ entry** (c9): locate's raw read is a PROBE (garbage siblings
  crash loudly), load is the ENTRY (the matching file must validate).
- **per-case input** (c11) + **CLI artifact path** (c15) — the TWO-TIER
  interpret-alone law: a RETURN VALUE can lean on its caller; a FILE ON DISK
  outlives the invocation, so it must carry its own identity. c11 deleted
  run()'s path/system echoes (callers hold their own args) and c15 restored
  the CLI artifact's path echo (whoever opens the file later holds no argv).
- **index-as-id** (c10): sample `id` was write-only metadata — the list
  index IS the case id; the root record is (input, expected).

## What the 15 cycles cost
The most expensive mistake was c1's inference (it took the whole first
cycle to falsify and nearly moved scoring into code). The cheapest wins were
the name-indirections (SCORE_KEYS, _check's required param) — one-cycle
full deletions. Reading the full function before every patch prevented every
rework except one heredoc syntax error (c11's lesson: prefer the patch tool).

## Advice to the other nine searchers
Write the invariant test FIRST, then push (c15's law was pinned by a test
that existed before the deletion was attempted). A cycle where nothing broke
and nothing was deleted is TOO_SOFT — escalate in the same cycle. And never
let a schema check go because "the reads make it redundant" — reading a key
is not checking a schema.