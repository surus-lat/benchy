BARE-METAL GOLEM REPORT
========================
Iteration: 6 (FINAL)
Worktree: all
Files scanned: 8

VIOLATIONS: none

BREAKAGE STATUS: BROKE (all tests pass, end-to-end works)
DELETION CANDIDATES: none

VERDICT: PASS

The new benchy engine is bare metal:
- core: frozen spine with protocols (Task, Scorer, System, Data)
- data: functions that build Datasets (jsonl_dataset, local_dataset)
- scoring: functions that return (score, aggregate) pairs
- adapters: functions that build Systems (http_system, openai_system)
- cli: bare metal eval command (parse args -> build -> run -> write JSON)

The end-to-end test passes with a mock system.
All 5 worktrees integrate correctly.

The old benchy had handlers, interfaces, registries, metadata.yaml, provider configs,
task groups, capability flags, and 200-line files to add a task.
The new benchy has: protocols, functions, and one eval command.

This is the bare metal. Push no further.
