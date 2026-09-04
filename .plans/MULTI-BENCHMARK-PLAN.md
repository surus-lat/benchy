# Plan: Multi-benchmark support

## Context

The second layer was designed with a "one `benchmark.yaml` per project" convention. The code already supports multiple benchmarks — every function takes a path argument, `--benchmark <path>` accepts any path, and the `name:` field inside the YAML controls data output directories. The problem is entirely in conventions, defaults, and docs. This plan removes that constraint with minimal code changes.

---

## What changes

### 1. `src/benchy_cli.py` — 4 changes

**a.** Add shared discovery helper `_discover_benchmark_candidates() -> list[str]` — scans `benchmark.yaml` at root and `benchmarks/*.yaml`.

**b.** Add `benchy benchmarks` list command — scans for specs, reads `name:` and `description:` from each.

**c.** Update `benchy create --output` default from `"benchmark.yaml"` to `None`; resolve to `benchmarks/<name>.yaml` after the wizard collects the name.

**d.** Update `benchy validate --benchmark` default from `"benchmark.yaml"` to `None`; use `_discover_benchmark_candidates()` to resolve.

### 2. `src/benchy_create.py` — 1 change

Accept `output_path=None`; resolve to `benchmarks/<name>.yaml` after name is collected. Add `out.parent.mkdir(parents=True, exist_ok=True)` before writing.

### 3. `src/benchy_cli_eval.py` — 2 changes

**a.** Add `_resolve_benchmark_path(benchmark_arg) -> str`: explicit path → use as-is; `benchmark.yaml` at root → use; `benchmarks/*.yaml` with one match → use; multiple found → SystemExit listing candidates; none → SystemExit with hint.

**b.** Replace line 1301 fallback (`"benchmark.yaml"`) with `_resolve_benchmark_path(args.benchmark)`.

**c.** Update the auto-trigger guard at line ~1396 to also fire when a spec can be auto-discovered (not only when `--benchmark` was explicitly passed).

### 4. `coREADME.md` — 3 edits

- Replace "One per project, lives at the root" with `benchmarks/` convention
- Update CLI examples to show explicit paths
- Add `benchy benchmarks` to the authoring surface section

### 5. 7 SKILL.md files — mechanical edits

Replace hardcoded `benchmark.yaml` references with generic path language.

| File | Lines |
|------|-------|
| `define-task/SKILL.md` | ~line 121 |
| `define-scoring/SKILL.md` | ~line 76 |
| `configure-model/SKILL.md` | ~line 108 |
| `setup-data/SKILL.md` | ~lines 80, 84 |
| `synthesize-data/SKILL.md` | ~lines 26, 33, 39, 42 |
| `run-benchmark/SKILL.md` | ~lines 7, 18, 33, 52, 62, 96 |
| `read-results/SKILL.md` | ~line 74 |

---

## What does NOT change

- `src/benchmark_compiler.py` — already path-agnostic
- `src/data_generator.py` — already path-agnostic
- Any engine file
- Existing `benchmark.yaml` at root — found first by auto-discovery, fully backward compatible

---

## Verification

```bash
# Backward compat
benchy eval --benchmark benchmark.yaml --limit 2
benchy eval --limit 2                              # auto-discover single spec

# Multi-benchmark
benchy benchmarks                                  # lists all
benchy validate --benchmark benchmarks/a.yaml
benchy eval --benchmark benchmarks/b.yaml --limit 2

# Create nudges to benchmarks/ dir
benchy create                                      # output → benchmarks/<name>.yaml

# Ambiguous: must error and list candidates
benchy eval --limit 2                              # when both benchmark.yaml and benchmarks/*.yaml exist
```
