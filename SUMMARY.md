# s10 — salvage (the archaeologist): final summary

## final shapes
- ENGINE nb/engine.py — four pure functions: compile (spec -> callable),
  grade (benchmark + callable -> artifact, the seam), run (system is the
  ARGUMENT), as_loss (benchmark -> loss(system) -> float). Stdlib only.
- CLI nb/__main__.py — python -m nb <root> <path> <artifact> <system>:
  named system required, path-coherence check, one ack line, exit 2 on
  any refusal, artifact JSON for programs.
- DATA bench/hello/bundle/ — sentiment.json (task+scoring+cases, one
  file, locatable by /sentiment) + systems.json (specs as data). Pure
  data, git-tracked, offline.
- 26 tests define "broken". Good stub 1.0, dumb stub 0.5, loss ranks.

## final metrics (golem report, verbatim)
{"cycles": 20, "verdicts": {"HARD_PUSH": 6, "BARE_METAL": 9,
 "NOISE_REMOVED": 5}, "files": 2, "loc": 95, "deps": 0, "concepts": 5}

## the archaeology verdict (3 lines)
Hypothesis confirmed, narrowly: the old benchy knew things from-zero
missed — but not its machinery; three small ideas survived, and each
was half-anticipated by this tree's own laws (the audit said WHICH
refusal was missing, never what shape it should take).
ACCEPTED: c16 failures-are-evidence (divergence: /6 not /5), c17
empty-exam-refused (crash != refusal), c18 path-coherence.
REJECTED: c19 counts (second address; old aggregate excluded errors),
exit policies, resume/status, spec registries — derivable or s09's.

## best discovery
Declarations are load-bearing only when enforced: grade refusing exam
keys outside task.output.choices (c8) and literal declared==implemented
scoring (c11) turned the data schema into the whole framework — the
pillar IS the refusal, and everything else is lookup.

## most expensive mistake
c5's bundle move broke 13 tests because the artifact's shape was pinned
in tests before the concept was probed (pins guard noise too). The fix
became law: probe before pinning presence, pin only behavior and
vision shape; fixed-forward noise deletions are NOISE_REMOVED, not
HARD_PUSH (logged that correction inside c5 itself).

## what I would tell the other nine searchers
Pin behavior, not presence — a presence test guards whatever exists,
including noise; deletion probes must break only what you actually
believe in. Fuse same-kind refusals into one literal check (c11, c20).
Never store what a consumer can count (c9, c19). When a probe breaks
NOTHING, delete the whole target and end HARD_PUSH — softness ends
cycles, deletions earn them. Run your own archaeology last, ideas-not-
code, and log the rejections: they are your angle's real product.