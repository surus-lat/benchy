# LEARNINGS — s04 (onefile): what the bare metal actually is

## TASK pillar — the description of the program we search for

The task is DATA, full stop: `spec["task"] = {ont, in, out}` in bench.json.
Zero lines of engine code serve it. Every attempt to give it code died as
noise (Task class c1, spec/task attributes c8). The output schema's enum
looked like it demanded a validation gate (ok(), c4) — it does not: an
out-of-enum prediction cannot match any want, so grading already scores it
0. The enum survives as the DECLARED OUTPUT SPACE for optimizers reading
the task, not as grader logic. The task pillar is a schema written by a
human and read by a machine — the metal is the json keys, not code.

## SCORING pillar — what good means, and the loss

IDEAS.md line 1 — "el scoring function se puede derivar del output
schema!" — turned out to be literally true (c14): grade() dispatches on
WANT's shape. want scalar → one part, whole equality. want dict → weights
over its fields. The scoring spec is a plain `{field: weight}` map,
absent = binary. No 'exact' sentinel, no {"fields": ...} wrapper, no
string special-case: those were three encodings of one idea. Grading
subsumes enum gates (an impossible value simply mismatches). The loss IS
`1 - mean(score)` — as_loss() is a lambda over run(), three lines, and
it ranks the stubs exactly as the vision demands.

## DATA pillar — the exam

The graded artifact dict IS the exam: {ont, score, loss, cases}. No Exam
class, no writer, no formatter — the dict serializes unchanged and is
re-read unchanged by resume. The artifact path IS the system label (c6):
no name derivation, no "system" field. The mid-run incremental write is
BARE METAL (c10): kill-safety is the resume contract — a final-write-only
engine loses ALL graded work to a kill. The O(n²) rewrite-per-row is the
honest price of durability at benchmark scale. Resume reads the artifact
and re-takes only missing rows: the artifact is the protocol between runs.

## SYSTEM pillar — the AI-API / compiler

The deepest cut of the search (c15): the AI-API is a callable SHAPE,
`f(in, ctx) -> out`, not a function. invoke() — which survived 13 cycles
as "the one protocol" — deleted wholesale: run() IS the compiler. It
binds every data shape (name → systems[name], {"py": "file:func"} →
compiled module fn, {"rule": ..., "default": ...} → keyword closure,
callable → itself) to the protocol ONCE per exam, never per case.
Per-case dynamic import was a REAL correctness bug (c9: module state
reset every case) — compile-py-once-per-exam is metal, not style. A
default-only rule IS a constant system; there is no const shape. What
IDEAS.md called "the compiler, the exam taker" is 15 lines at the top of
run(), not a subsystem.

## Ontology — the address IS the registry

`/<task?>/<domain?>/<language?>` needs zero registry code: each bench
declares `task.ont`, and Benchmark.load("/sentiment") WALKS bench/ for
it (c13, BARE_METAL — the acceptance bar demands /sentiment resolve; the
walk IS the filesystem-as-registry made addressable). No Registry, no
index, no manifest — rglob + one equality check.

## did the angle survive?

Partially, and the failure is the finding. The hypothesis — "fits in ONE
file under 400 lines, else the design is wrong" — was never tested by the
ceiling: the engine landed at 99 LOC, 4x under the bar. The 400-line
ceiling NEVER BOUND. What does that mean for the other angles: ENGINE
SIZE WAS NEVER THE CONSTRAINT. The golem's real enforcement — concept
count must not grow, LOC must not grow, every concept must justify
itself — did all the work. Any of the ten angles would fit in one file;
the discriminant is how many CONCEPTS survive deletion attempts, not how
many lines they occupy. The onefile angle's true contribution is not the
file but the forcing function: a single file makes fusion visible (there
is nowhere to hide a helper), so _write and invoke() died by the same
logic — helpers with one caller are not concepts.

## discoveries that transfer

1. Grading subsumes enum gates — an out-of-enum prediction cannot match
   any want; the grader scores it 0. Never validate output shape twice.
2. Artifact-path-as-label — the artifact's path names the run; no label
   derivation, no system field. Evidence needs no metadata about itself.
3. Compile-py-once-per-exam — per-case dynamic import resets module
   state: a correctness bug wearing a performance costume.
4. Kill-safety needs the mid-run write — O(n²) rewrites are the price of
   a resume contract; the artifact is the protocol between runs.
5. Ontology walk = the address IS the registry — rglob beats any index.
6. Shape-tests over string-special-cases — dispatch on the data's shape
   (dict/scalar) deletes sentinel strings and wrapper keys wholesale.
7. The protocol is a SHAPE, not a function — f(in,ctx) is the interface;
   the compiler binds data specs to it per exam. invoke() was noise that
   13 cycles of reverence kept alive.

## the meta-lesson

Claimed-but-untested invariants are noise's favorite hiding place. Every
concept that died (invoke, _write, the 'exact' sentinel, the enum gate,
the Exam class) had a story about why it was essential — and none of
them had a TEST that broke on deletion until the deletion was attempted.
Conversely, the two BARE_METAL verdicts (mid-run write, ontology walk)
became metal exactly when a test was WRITTEN to claim their contract
first, then deletion was attempted. The discipline that works: encode the
claim as a test, then try to delete the code. The test breaks or the code
was noise. Nothing else distinguishes metal from story.