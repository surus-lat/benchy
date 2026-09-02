# LEARNINGS — what the bare metal actually is (s10, salvage)

20 push cycles. Final shape: 2 engine files, 95 loc, 5 concepts, 0 deps,
stdlib only, 26 tests. Every concept survived ≥1 deletion probe. The
four pillars are DATA; the engine is four pure functions + a CLI.

## TASK pillar — the declaration is load-bearing

A task is two lines of data: `input` (text) and `output` (declared
choices). The lesson that cost a cycle (c8): a declaration the engine
never reads is a LIE, not a contract. The task block existed for 7
cycles as decoration; the probe broke only its own presence guard,
proving the engine never consulted it. Fixed forward: grade now refuses
any exam key outside `task.output.choices`. The task pillar is not a
schema library — it is one enforcement point inside the seam. Ontology
path (`/sentiment`) is the task's identity, and it must be coherent
(c18): the file's declared path must equal the requested path, or the
exam is a broken install, refused.

## SCORING pillar — declared == implemented, exactly

Scoring is data (`rule: match, aggregate: mean`) and the engine enforces
LITERAL equality (c11): declare anything else — unknown rule, unknown
aggregate, extra keys nobody reads — and grade refuses. Two same-kind
refusals fused into one check. Scoring is a lookup, not a framework;
when a new rule is implemented, the literal changes, and that is the
whole ceremony. The aggregate is the point estimate of a distribution
(GOLEM's law) — and it is the ONE scalar: `score = mean(per-case)` and
the loss consumes it directly.

## DATA pillar — the exam is cases; one fact, one address

n cases, each (input, expected). Position IS the id (c9 deleted the
explicit index — derivable). Per-case rows carry input/expected/
prediction/score/error; nothing derivable is stored twice (c19 REJECTED
the old counts block for exactly this: passed/failed/error counts are a
second address for len(cases) + the rows). An empty exam is a broken
exam, refused upfront (c17) — without the refusal as_loss crashes the
optimizer with ZeroDivisionError; a crash is not a refusal.

## SYSTEM pillar — the spec is data; the compiler is the only growth point

`compile(spec) -> invoke(text) -> prediction`. One kind today (keyword +
default; a constant system is the degenerate keyword, c14 — the choice
lives in the SPEC, as data, because a hardcoded default is score-blind
on a balanced exam). Matching is LITERAL (c13 deleted silent case
folding — the spec carries case; explicit beats implicit). Cloud kinds
join here as new spec kinds, per the steering addendum: the exam-taker
is cloud-first, serving is long-term, the compiler is the ONLY place
the engine may grow. And the seam accepts ANY callable (c3): real APIs,
workflows, cached runs bypass compile and grade directly.

## SYSTEM as argument — the vision invariant

`run(benchmark, system)` — the system is the ARGUMENT, never a
constructor field (c7 tried to inline it; 6 vision tests broke).
`as_loss(benchmark) -> loss(system) -> float` is the headline export
(c15): the import IS the pin — removing it breaks the test collection
itself; the optimizer consumes loss(system) directly. Engine purity
(c4): values in, values out; files are persistence, the CLI is the file
layer. A CLI for people with no Python knowledge (s07 c3): the system is
NAMED always (c10 — refusal beats surprise), ONE ack line on stdout
(c12 — which exam, which system, what score; the artifact file is for
programs, the ack is for people).

## SALVAGE pillar — the archaeology verdict (this angle's own)

**Hypothesis: confirmed, NARROWLY.** The old benchy knew things a
from-zero search misses — but not its machinery. Its machinery (counts
blocks, exit policies, resume/status machinery, spec registries) is
either derivable inside a smaller design or another angle's territory.
What survived were three SMALL ideas, and every one was half-anticipated
by this tree's own laws — the audit mostly told us WHICH refusal we
were missing, not what shape it should take.

**Donations ACCEPTED (each survived its probe):**
- c16 failures-are-evidence (old run-loop contract,
  .staging/benchy/benchmark.py): a system failure on ONE case is
  evidence, never an abort. grade catches Exception per case;
  prediction=None + error="Type: msg" in the row; scores 0 and stays IN
  (/6 not /5). DELIBERATE DIVERGENCE: the old system excluded errored
  samples from the aggregate — ours includes them, so reliability
  lands in the one scalar the optimizer consumes. Probe-delete broke
  14 tests.
- c17 empty-exam-refused (old status vocabulary, src/outcome.py
  no_samples): grade refuses zero cases. Probed live: without it
  as_loss crashes the optimizer with ZeroDivisionError. A crash is not
  a refusal.
- c18 path-coherence (old spine OntologyPath — registry key == on-disk
  layout): the CLI refuses a benchmark whose declared path != requested
  path; a broken install would silently grade under the WRONG identity.
  One literal check in the file layer, no engine change.

**Donations REJECTED, with reasons:**
- c19 counts block (AGENTS.md run_outcome.json: passed/failed/error/
  pending/no_samples/skipped): donated, pinned, probe-deleted — only
  the donation's own pin broke. Derivable: len(cases) + the rows are
  the same facts at a second address. The old system needed counts
  BECAUSE its aggregate excluded errors; ours includes them.
- exit policies (relaxed/smoke/strict): the old system encoded
  "how bad is this run" as process exit-code policy over status
  vocabularies. Ours has one scalar and two honest refusals (exam
  defects, system failures); a caller reads the artifact. s09's
  territory if anyone's.
- resume/status machinery (task_status.json, skip-completed): the
  runner's problem (s09), not the benchmark's. A pure grade has no
  state to resume.
- spec registries: `compile` dispatches on spec["kind"] literally.
  A registry is a name for an if-statement.

**The honest-isolation protocol (what made the audit trustworthy):**
from-zero FIRST — the artifact was fully built, pinned and green
before the old system was read; archaeology LAST — the audit read the
main checkout read-only only after cycle 12; ideas-not-code — every
donation was re-derived in this tree's own shape and language, then
probe-deleted like any from-zero concept (a donation that cannot
survive the same deletion attempt as a native concept is not metal).
A rejected donation was logged as a real product of the angle, not a
failure of it. Four probes → three survived, one rejected: the
falsification (donations flooding in) did not happen — from-zero was
not a fantasy.

## the from-zero laws (transferable)

1. Grade seam metal: ONE place where any callable meets any exam.
2. Bundle purity: benchmark + systems in one self-contained directory.
3. Engine purity: values in, values out; files live in the CLI layer.
4. Declared == implemented scoring, LITERALLY — refuse the rest.
5. Refusal beats surprise: silent defaults, wrong identities, empty
   exams, undeclared scoring — all refused, exit 2.
6. Ack line for people + artifact for programs: one stdout line.
7. Import-as-pin: the test suite's import of as_loss IS the export.
8. Failures-are-evidence: a failing system scores 0 and stays in.
9. Empty-refused: broken exam data is refused, never crashed on.
10. Path-coherence: declared identity == requested identity.
11. One fact, one address: never store what a consumer can count.
12. Two same-kind refusals are ONE check (fuse them).