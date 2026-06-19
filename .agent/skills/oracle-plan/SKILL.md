---
name: oracle-plan
description: >
  Bidirectional algorithm for converting between a full working implementation
  and a battle-tested design plan. Use when: (1) you have a working implementation
  on a branch and need to produce a high-fidelity plan/design doc from it
  (Implementation → Plan), or (2) you have an oracle plan and need to execute it
  to a working implementation with zero wrong turns (Plan → Implementation).
  Triggers on: "extract the plan", "write the design", "implement from the plan",
  "oracle plan", "impl to plan", "document this branch".
---

# Oracle Plan — Bidirectional Algorithm

## What this skill is

A **plan** is a lossy compression of an implementation that retains exactly enough
information to reconstruct it from scratch. This skill defines how to produce that
compression (Implementation → Plan) and how to decompress it (Plan → Implementation).

The key property that makes the plan "oracle-quality": it was derived from a **working
implementation**, so every architecture decision reflects what actually worked, and
every pitfall documents a real bug that was hit and fixed — not speculation.

---

## Schemas

The function signature, typed:

```
f:   ImplementationSnapshot → OraclePlan    (Direction A)
f⁻¹: OraclePlan → ImplementationSnapshot   (Direction B)
```

Full type definitions in `schema.ts` (canonical) and `schema.py` (Pydantic v2).
Below is the structural summary — see those files for field-level comments.

```typescript
// INPUT to Direction A / OUTPUT of Direction B
interface ImplementationSnapshot {
  branch: string
  base_branch: string
  changed_files: ChangedFile[]       // path, action, diff, insertions, deletions
  test_results: TestSuiteResult[]    // file, count, passed, cases[]
  discovery: DiscoveryState          // tasks?, providers?, models?, other?
}

// OUTPUT of Direction A / INPUT to Direction B
interface OraclePlan {
  title: string                      // "Design: <Feature> (<branch>)"
  branch: string
  repo: string
  context: string                    // prose motivation, 2-4 sentences
  architecture_decisions: ArchitectureDecision[]  // decision, choice, why, rejected?
  file_map: FileMapEntry[]           // action, file, notes — dependency-ordered
  steps: ImplementationStep[]        // number, file, label, insertion_point, code, test
  known_pitfalls: Pitfall[]          // pitfall, fix — the crown jewel
  verification: Verification         // test_suite, discovery[], smoke?
}
```

### Quality validator

`isOracleQuality(plan: OraclePlan): ValidationResult` (TypeScript) /
`is_oracle_quality(plan: OraclePlan) -> ValidationResult` (Python) checks:

- Every step has an exact `insertion_point` (no "somewhere")
- Every step has a non-trivial `code` block (no pseudocode)
- Every step has `test.count > 0` and `test.expected_failure`
- `known_pitfalls.length / steps.length ≥ 0.33`
- At least one `architecture_decision` has a `rejected` field
- Every `discovery` verification has `expected_output`
- `verification.test_suite.expected_total > 0`
- `context` is at least 10 words

---

## Direction A: Implementation → Plan

> Input: `ImplementationSnapshot` (working branch + passing tests)
> Output: `OraclePlan` (design document that could regenerate the implementation)

### Phase 1 — Read the diff

```bash
git diff main...<branch> --stat          # file map
git diff main...<branch>                 # exact changes
```

Walk through every changed file. For each, note:
- **Action**: `modify` or `create`
- **What changed**: which class/function/block, not the whole file
- **Why it changed**: the capability or requirement it enables

### Phase 2 — Extract architecture decisions

For each non-trivial design choice, record:
```
| Decision | Choice | Why |
| What problem | What was picked | What makes this right here |
```

Include what was **rejected** if it would have been a plausible alternative.
Choices with only one obvious answer don't need a row.

**Instance from `feat/transcription-support`:**
| Decision | Choice | Why |
|----------|--------|-----|
| Base handler | `FreeformHandler` | Output is free-form text; reuses scoring scaffold |
| Interface | `OpenAIAudioInterface(OpenAIInterface)` | Inherits client, semaphore, retries; overrides 3 methods |
| New capability dim | `supports_audio` / `requires_audio` | Follows exact pattern of `supports_multimodal` etc. |
| Audio on-disk | WAV files in `.data/<locale>/` | Whisper needs a file path, not a numpy array; skip-if-exists caching |
| WER/CER | `jiwer` via optional dep group | Same pattern as `[translation]` → `sacrebleu` |

### Phase 3 — Build the file map

One row per file, ordered by dependency (modify before create, infrastructure before tasks):

```
| Action | File | Notes (one-liner) |
```

Every file that appears in `git diff --stat` gets a row. Notes explain the **role**,
not the change.

### Phase 4 — Write step-by-step instructions

One step per logical unit of change (usually one file, sometimes two tightly coupled files).
Each step must contain:

1. **The exact code** — not pseudocode, not paraphrase. Copy from the diff. The reader
   should be able to implement the step by pasting, not by interpreting.
2. **The insertion point** — `after line X`, `before the final else:`, `inside the return
   InterfaceCapabilities(...)` call. Never say "add this somewhere".
3. **The test** — test file name, count, and a brief list of what each test covers.
   The tests are written BEFORE the implementation step (TDD).

### Phase 5 — Extract the pitfalls

This is the crown jewel of the oracle plan. During implementation, you hit bugs.
Document each one as:

```
| Pitfall | Fix |
| What went wrong / would have gone wrong | The specific fix, not "be careful" |
```

**Instance from `feat/transcription-support`:**
| Pitfall | Fix |
|---------|-----|
| `_samples = []` as class attribute | Init in `__init__` — class-level lists are shared across instances |
| `aggregate_metrics` KeyError when all samples fail | Pre-initialize `overall[metric.name] = None` before the loop |
| Mock path in FLEURS tests | Patch `src.tasks.transcription.fleurs_X.save_audio_array`, not the definition site |
| `run_with_retries` callback signature | `attempt_fn(_attempt: int)` — underscore prefix, int param, always present |
| `response_format` on Whisper API | Must be `"text"` — returns plain string, not a ChatCompletion object |

A pitfall is only worth recording if it is **non-obvious** — something a competent
engineer would reasonably get wrong on first pass. Don't document things the type
checker would catch.

### Phase 6 — Write the test table

```
| File | Covers (N tests) |
```

Every test file, count, and one-line summary. Include the exact command to run them
and the expected output (e.g. `34 passed`).

### Phase 7 — Write verification

Two kinds:
1. **Discovery** — CLI commands that confirm the feature is wired up:
   ```bash
   benchy tasks | grep <name>
   benchy providers | grep <name>
   ```
2. **Smoke** — a minimal live run that exercises the real path:
   ```bash
   benchy eval --provider X --model-name Y --task Z --limit 5 --run-id smoke
   ```
   Include what to check in the output file and what values prove success.

### Phase 8 — Assemble the plan document

Structure:
```markdown
# Design: <Feature Name> (<branch>)

## Context
<Why this exists. The LatamBoard / product motivation. 2–4 sentences.>

## Architecture Decisions
<table>

## File Map
<table>

## Step-by-Step Implementation
### Step N — <file>: <what>
<exact code blocks + insertion points>

## Known Pitfalls
<table>

## Tests
<table + run command + expected count>

## Discovery Verification
<commands>

## Smoke Test
<command + what to check>
```

The title is **Design:**, not Plan: or Implementation:. It is a design document —
it describes what the implementation is and why, not a to-do list.

### Phase 9 — Publish as a single GitHub issue

```bash
gh issue create \
  --repo <org>/<repo> \
  --title "Design: <Feature Name> (<branch>)" \
  --body "$(cat <<'EOF'
<full plan markdown>
EOF
)"
```

One issue, not N. The design doc is the issue body. This is searchable,
linkable, and reviewable without needing to read code.

---

## Direction B: Plan → Implementation

> Input: `OraclePlan` (produced by Direction A, or written to that standard)
> Output: `ImplementationSnapshot` (working code, passing tests, discoverable feature)

### Execution rules

1. **Follow the file map top-to-bottom.** Dependencies are already ordered.
   Don't skip ahead.

2. **Write the test first for each step.** The plan includes the test file name and
   cases. Create it, run it (expect failure), then implement.

3. **Pre-empt every pitfall.** Read the pitfalls table before implementing each step.
   The fix is already documented — apply it proactively, not reactively.

4. **Use exact code blocks as-is.** The plan contains real code that was verified
   in a working implementation. Don't paraphrase or simplify. Paste and adjust
   only for context (imports, surrounding code).

5. **Verify each step before proceeding.** Run the step's tests. If they fail,
   diagnose before moving to the next step.

6. **Do not commit until the full feature passes verification.** Run the complete
   test suite and the discovery/smoke verification commands before any commit.

### Step template

For each step in the plan:
```
[ ] Write failing test for step N
[ ] Run test → confirm it fails (expected error type)
[ ] Implement step N (exact code from plan)
[ ] Run test → confirm it passes
[ ] Move to step N+1
```

After all steps:
```
[ ] Run full test suite (uv run pytest <all test files> -v)
[ ] Run discovery verification (benchy tasks / providers / models)
[ ] Run smoke test if API key available
[ ] Commit
```

---

## Plan quality checklist

A plan is oracle-quality when:
- [ ] Every step has an exact insertion point (not "add this somewhere")
- [ ] Every step has a real code block (not pseudocode or paraphrase)
- [ ] Every step names its test file and lists what each test covers
- [ ] The pitfalls table has at least one entry per ~3 steps (if zero, you didn't look hard enough)
- [ ] Architecture decisions explain the **rejection** of plausible alternatives, not just the choice
- [ ] Verification commands include expected output, not just "it should work"
- [ ] The total test count is stated and matches reality

---

## Anti-patterns

**In plan extraction:**
- ❌ Pseudocode instead of real code — forces the reader to re-derive, defeats the oracle
- ❌ Vague insertion points ("somewhere in this function") — ambiguous, causes bugs
- ❌ Empty pitfalls table — means you didn't look hard enough at what you got wrong
- ❌ Breaking into N separate issues — a design is one thing; task decomposition is separate
- ❌ Documenting obvious pitfalls ("don't forget to import") — noise, dilutes real warnings

**In plan execution:**
- ❌ Implementing before writing the test — skips TDD, loses the failing-test signal
- ❌ Paraphrasing code blocks instead of using them verbatim — reintroduces bugs the oracle already fixed
- ❌ Skipping the pitfalls table — you will hit the same bugs the oracle already documented
- ❌ Committing before full verification — partial implementations in version history
- ❌ Implementing out of file-map order — breaks dependency assumptions

---

## Reference: `feat/transcription-support` as a canonical instance

This skill was derived from a concrete implementation session. The full oracle plan is
at `.claude/plans/please-deeply-understand-benchy-scalable-journal.md` and the
corresponding GitHub issue is `surus-lat/benchy#31`.

That session established:
- **13 files** touched (4 modified, 9 created)
- **34 tests** across 6 test files
- **5 pitfalls** documented (all real bugs hit during implementation)
- **6 architecture decisions** with explicit rationale
- **1 GitHub issue** as the design doc

Use this as the calibration baseline for what "oracle quality" looks like in benchy.
