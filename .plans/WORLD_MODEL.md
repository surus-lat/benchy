# Benchy World Model

Benchy is a benchmarking engine for AI systems, powering the LatamBoard leaderboard. The fundamental design is a task/interface decoupling:

- Tasks (what to evaluate) inherit from 4 format handlers: MultipleChoice, Structured, Freeform, MultimodalStructured
- Interfaces (how to talk to providers) return a unified {output, raw, error, error_type} contract
- Providers split into model providers (all OpenAI-compatible) and custom HTTP system providers

The 4 agent skills in `.agent/skills/` are the authoritative operational guide:
1. `evaluate` — canonical two-stage smoke+full workflow
2. `add-task` — 30–50 lines to add a new task
3. `add-provider` — config-only path (OpenAI-compat) or code path (custom HTTP)
4. `interpret-run` — diagnose run_outcome.json failures

Key conventions to always respect: smoke before full, --run-id for resumability, run_outcome.json as source of truth, and `.agent/skills/<skill>/SKILL.md` before doing any of those four operations.

A second layer is being designed on top of the engine (see `docs/SECOND_LAYER_DESIGN.md`). It exposes benchy to non-developer users via a `benchmark.yaml` spec and a `benchy create` wizard. A benchmark has four things: task (inputs/outputs), scoring function (binary | per_field | semantic), data (local | generated | HuggingFace), and target (api | model | local). The compiler maps `benchmark.yaml` → handler instance → existing BenchmarkRunner, zero engine changes.
