BARE-METAL GOLEM REPORT
========================
Iteration: 1
Worktree: all
Files scanned: 12

VIOLATIONS:
- [CRITICAL] .worktrees/cli/benchy/cli/eval.py: Too many lines (200+). The bare metal should be one function, not a class hierarchy.
- [CRITICAL] .worktrees/cli/benchy/cli/eval.py: SimpleTask and SimpleScorer are abstraction creep. The CLI should use the protocols directly.
- [WARNING] .worktrees/core/benchy/core.py: Benchmark class is a dataclass but could be a function. run_exam is the bare metal.
- [WARNING] .worktrees/adapters/benchy/adapters/http.py: HTTPAdapter has too many options. The bare metal is: URL + callable.
- [NITPICK] .worktrees/scoring/benchy/scoring/exact.py: ExactMatch has case_sensitive parameter. Is this essential?

BREAKAGE STATUS: BROKE (import failures across worktrees, namespace collision)
DELETION CANDIDATES:
- .worktrees/cli/benchy/cli/eval.py (can be 50 lines)
- .worktrees/core/benchy/core.py (Benchmark class can be deleted, use run_exam directly)

VERDICT: TOO TIMID

The frozen spine (benchy.core) is good. But we built too much on top of it.
The CLI should be: parse args -> build System -> build Data -> build Scorer -> run_exam -> write JSON.
No classes. No hierarchies. Just functions.

Push harder. Break more.
