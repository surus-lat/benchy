---
name: best-part-is-no-part
description: >
  Use when proposing or reviewing a design that adds parts — classes,
  methods, abstractions, files, YAML fields, registries, lifecycle
  hooks. Especially when justifying additions as "for symmetry,"
  "for flexibility," "we might need it," or "for cleanliness."
  Triggers on phrases like "let's add a base class," "I'll introduce
  a Capabilities dataclass," "we need a registry," "this should
  declare load/unload." Symptoms: every part is reasonable in isolation
  but the count keeps climbing; defenses cite "best practice" rather
  than a current concrete consumer; arguing a capability "wasn't really
  needed" to justify removing a part.
---

# Best Part Is No Part

## The principle (stated correctly)

> **The best part is no part.** Apply ruthlessly. Each part you add is
> a part to test, document, evolve, version, debug, and remove.

This is **not** about removing features. It is about delivering the
same features with fewer parts.

The trap: agents (and humans) hear "best part is no part" and start
trading capability for fewer parts, then argue the cut capability
"wasn't really needed." That isn't this principle. That is feature
removal disguised as simplification.

The correct framing:

```
For each proposed part, find what would carry its weight
if the part didn't exist. If something already in the system —
Python the language, an existing pattern, stdlib, the filesystem,
structural duck typing — can carry that weight, the part is gold
plating. Cut it. Keep the capability.
```

## When to apply

- A design proposal lists more parts than the problem requires.
- Someone is "adding for symmetry" — a part whose only justification
  is that another part exists.
- A new dataclass / protocol / base class is being introduced and you
  can name what it adds beyond what stdlib or existing project types
  already provide.
- A registry, manifest, or central list is being created with one or
  two entries.
- A method's default implementation is `pass`, `return self`, or
  the identity function.
- "Lifecycle methods" (`load`/`unload`/`init`/`destroy`) are being
  added to a protocol when the language already manages the lifecycle.

## When NOT to apply

- Pure feature removal ("we don't need transcription at all"). That is
  a product decision, not a simplification.
- Discipline-enforcement contracts (TDD, "always run tests before
  commit"). Those parts exist precisely because the system won't
  enforce them without explicit framing.
- Concrete past-bug defenses. If a `try/except` exists because of an
  actual production incident, the historical incident is the
  justification — keep it, document the why.

## The four lenses

For each part you are tempted to add, ask each question in order. If
any answer is yes, the part is suspect.

### 1. Does Python the language already provide this?

Common cases where the answer is yes:

| Proposed part | Python already provides |
|---|---|
| `load()` / `unload()` lifecycle methods | Constructor + reference counting + `__del__` + `with` context managers |
| Capability flags as a custom dataclass | Class attributes, dataclasses, `typing.Protocol` |
| Identity-default abstract method | The default implementation IS the identity — write it as a normal method, not abstract |
| "Factory function" wrapping `Class(arg)` | `Class(arg)` is already the factory |
| Result-shape wrapper class | A `dict` or `TypedDict`; the structural type IS the shape |
| Singleton boilerplate | Module-level value; modules are already singletons |

### 2. Does an existing pattern in this codebase already deliver this?

Walk the codebase before adding. Examples that recur:

- A new "Capabilities" dataclass when one already exists for a sibling
  concept (`AdapterCapabilities` next to `InterfaceCapabilities`).
- A new "Runner" / "Manager" / "Coordinator" when the existing one
  could absorb the new responsibility without growing unwieldy.
- A new "config block name" in YAML when the existing pattern is
  `provider: foo` + `foo: {…}` and the new thing follows the same
  shape — adopt the pattern, don't invent.

### 3. Does stdlib or the filesystem already deliver this?

| Proposed part | Stdlib / filesystem alternative |
|---|---|
| `REGISTRY: dict[str, Cls]` for plugin discovery | `pkgutil.iter_modules(pkg.__path__)` + `importlib.import_module` |
| Custom config-file format | `tomllib`, `json`, `pathlib.Path.read_text` |
| Custom "module path" string parser | `importlib.import_module("a.b.c")` |
| Custom "find files matching pattern" | `pathlib.Path.glob`, `pathlib.Path.rglob` |
| "Plugin manifest" tracking | The directory listing IS the manifest |

### 4. Is this accomplished by structural duck typing?

If two classes expose the same method signature with the same return
shape, they ARE substitutable for that purpose. You do not need a
shared base class, a Protocol declaration, or a wrapper that "unifies"
them — Python's duck typing is the unification.

Inheritance and wrappers are for **shared implementation**, not for
**shared shape**. Reach for them only when there is code to share.

## Worked example: the adapter layer

A real brainstorm from this repo. Original proposal (large) vs
final design (small) — same capability delivered by both.

### What both designs deliver (identical capability)

- Per-model-family Adapter abstraction
- Lifecycle: lazy load, resource cleanup
- Discoverability: enumerate available adapters
- Unified Task-side access (Task doesn't care if the inference came from
  an Adapter or an Interface)
- Request introspection for retry and logging
- Pluggable capability declarations
- Backward compatibility with all existing model YAMLs

### Original proposal (~13 new files, 3 new abstractions)

| Part | Capability it delivers |
|---|---|
| `BaseAdapter` ABC with `load`, `unload`, `prepare_requests`, `generate_batch` | Lifecycle + introspection + inference |
| `AdapterCapabilities` dataclass | Capability declaration |
| `REGISTRY: dict[str, type[BaseAdapter]]` in `__init__.py` | Discoverability |
| `WhisperPipelineAdapter` wrapping `TransformersAudioInterface` | Unified Task access |
| `OpenAIChatAdapter` wrapping `OpenAIInterface` | Unified Task access |
| `OpenAIWhisperCloudAdapter` wrapping `OpenAIAudioInterface` | Unified Task access |
| `VoxtralChatAdapter` | The actual unblocking |
| `Qwen3ASRChatAdapter` | The actual unblocking |
| `adapter_config:` YAML field | Adapter-specific config |
| `adapter:` YAML field | Naming |

### Final design (~5 new files, 1 new abstraction)

Every capability above is delivered, with these reframings:

| Original part | Replaced by | Capability preserved? |
|---|---|---|
| `load` / `unload` methods | Lazy init in `generate_batch` + Python GC | Yes — `del adapter` releases resources |
| `AdapterCapabilities` dataclass | `InterfaceCapabilities` class attribute (existing type) | Yes — reuses live code |
| `REGISTRY` dict | `pkgutil.iter_modules` + `importlib.import_module` | Yes — filesystem listing IS the registry; `list_adapters()` is 3 lines |
| Three wrapper adapters | Structural duck typing: `BaseAdapter.generate_batch` and `BaseInterface.generate_batch` have the same shape | Yes — Task code is already polymorphic over both |
| Separate `prepare_requests` protocol method | Default identity implementation on `BaseAdapter`; subclasses override only when needed | Yes — BenchmarkRunner's existing flow works unchanged |
| `adapter_config:` field | Adapter-name block: `adapter: voxtral_chat` + `voxtral_chat: {…}` | Yes — mirrors the existing `provider:` pattern, no new shape |

What survives: `BaseAdapter` (one abstract method + one class attribute + identity default), `voxtral_chat.py`, `qwen3_asr_chat.py`, the 12-line discovery file, the 6-line connection branch.

What got cut: the `AdapterCapabilities` type, three lifecycle methods, the REGISTRY dict, three wrapper adapters, one YAML field.

What got kept: every capability the original promised.

This is the principle correctly applied.

## Rationalization table

| Rationalization | Why it is wrong |
|---|---|
| "It would be cleaner to add a wrapper for symmetry." | Duck typing IS the symmetry. A wrapper that only translates names adds a part without adding behavior. |
| "We might need this someday." | Defer until you do. The cost of adding the part when actually needed is almost always lower than the cost of carrying it speculatively. |
| "A registry is more explicit than importlib." | The filesystem is the source of truth either way. The registry is a second source you have to keep in sync with the filesystem. |
| "Lifecycle methods make resource management obvious." | Python's reference counting and GC already manage resources. An explicit `unload()` is meaningful only if you genuinely deviate from GC — which is rare. |
| "Inheritance enforces the contract." | Duck typing enforces the contract just as well, without coupling subclasses to a base class's evolution. Reach for inheritance for shared *implementation*, not shared *shape*. |
| "I'm following best practices." | Best practices are heuristics. The principle here is not a heuristic — it's a direct cost-benefit analysis on every part. Cite the concrete consumer, not the practice. |
| "Tests are easier with this abstraction." | Untested code is the bigger problem. If the existing structure is hard to test, write the test against the existing structure and only abstract when a real test case demands it. |
| "It's defensive programming." | Defensive against what? Name the failure mode. If you can't name it concretely, the defense is speculative and the part is gold plating. |
| "Premature optimization is bad but this isn't optimization." | Premature *abstraction* is the same disease. Both add complexity for speculative future benefit. |
| "The cut capability wasn't really needed." | Stop. This is the trap. The principle says keep capabilities AND remove parts. If you find yourself arguing this, you have left the principle and are doing feature removal. Go back. |

## Red flags

Self-check signals that you are about to add a part you should not:

- You cannot name a current concrete consumer of the new part — only
  "for cleanliness," "for symmetry," or "for the future."
- The default implementation of an abstract method is `pass`,
  `return self`, or the identity function.
- A registry / manifest / index file has exactly one or two entries.
- A factory function does nothing but call a constructor.
- A wrapper class has no methods of its own — only delegations.
- You are about to define `AdapterX`, `AdapterXBase`, and
  `AdapterXFactory` at the same time.
- You are using inheritance to share a shape that two classes already
  have independently.
- You catch yourself listing capabilities the new part "could
  provide" rather than capabilities it must provide today.

When any red flag fires, pause and walk the four lenses above before
proceeding.

## The audit, condensed

For each part in your proposal, fill in one row:

| Part | Capability it delivers | Could Python / existing pattern / stdlib+filesystem / duck typing deliver it? | Decision |
|---|---|---|---|

If the third column is yes, the part is suspect. Move it to "cut"
unless you can name the concrete consumer that exposes the gap.

If you end the audit with the same parts you started with, you did not
audit. Run it again.

## Common mistakes

- **Cutting capability to justify cutting parts.** The point is to keep
  every capability. If you find yourself arguing a capability wasn't
  needed, you have left the principle.
- **Stopping after one pass.** The first audit removes obvious parts;
  the second often removes parts whose justifications relied on the
  parts you just removed. Run until stable.
- **Adding "future-proof" abstractions.** Future you will thank
  present you for the parts that exist because of concrete current
  consumers, not the parts that exist because someone imagined a
  future use.
- **Confusing "no part" with "no thinking."** This principle asks for
  more design work upfront, not less. The reward is fewer parts to
  maintain forever after.
- **Skipping the audit because "this design is already simple."**
  Simple-looking designs hide the most parts. Audit them anyway.

## How to invoke this skill in a brainstorm

When in the middle of a design session, the user (or you, mid-thought)
can call: "audit this under best-part-is-no-part." Then:

1. List every proposed part as a row in the audit table.
2. For each row, walk the four lenses in order. Stop at the first yes.
3. Re-check the rationalizations table for any cut you are about to
   reverse.
4. Re-check the red flags before committing.
5. Restate every capability the original proposal claimed. Confirm the
   reduced design still delivers each one — naming the alternative
   delivery mechanism explicitly.
