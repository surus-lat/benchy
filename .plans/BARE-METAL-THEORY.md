# The Bare-Metal Theory of Benchy

This is the theory the push cycles are descending along. It exists so that
"push to bare metal" is not a vibe but a gradient: every cycle is a step, the
loss is complexity, and the stopping rule is exhaustion — when every
remaining surface of the core provably satisfies the Cutting Theorem below,
we are at a *local* minimum, and the only way to move again is for the vision
to move.

It formalizes the three pillars from the Sept-01 rant:

1. **The program space** — what is being searched/tested (the task).
2. **The exam** — how the program is tested and scored (data + scoring).
3. **The compiler** — what makes any AI able to sit the exam (the system
   boundary, a.k.a. the AI API).

---

## 1. The program space

Everything benchy does is an instance of one search problem:

> Find (or rank) a **program** `p` that performs a **task** `T`, where a task
> is a pair of spaces `(I, O)` — inputs and outputs — plus the semantics that
> make an element of `O` a *correct* answer for an element of `I`.

* `I` and `O` are given as **schemas** (JSON Schema). This is the "purest
  signal" description of a program: `in: doc[pdf]`, `out: extracted_fields[json]`.
* The program space `P(T)` is every callable from `I` to `O`: a raw model, a
  model+prompt node, a workflow of many models, a tool-using agent, a
  1998-era SVM. Benchy deliberately does not care how the program is
  implemented — written or learned, explicit or implicit. The whole space is
  one protocol: `System.invoke`.
* The ontology `/<task>/<domain>/<language>` is the coordinate system of the
  program space: it names regions of it. Nothing more.

**The program is the free variable.** This is why `System` is the *argument*
of `benchmark.run(system)` and never a field of the benchmark, and why
`as_loss()` is the shape of the exam rather than a bolt-on.

## 2. The exam

A benchmark is an **exam** for a program, and an exam is a triple:

```
E = (T, D, S)
T  task       the program's signature: input schema, output schema, and the
              render/parse coercion rules (the exam format)
D  data       the instances: n cases, each (input, expected)
S  scoring    the rubric: (prediction, expected) -> [0,1], plus how the
              per-case grades collapse into one number
```

`benchmark.run(system)` is the whole pipeline: for each case, render the input
into the exam-taker's wire format, get an answer, coerce it back, grade it.
The exam is *reusable* across any number of candidates — that is its entire
economic function.

### The distribution and the point estimate

A run produces not a number but a **distribution of per-case grades**.
Everything is a distribution; a benchmark's job is to collapse it into a
**point estimate** `fitness = g(grade-distribution)`. The estimator `g` is
part of the rubric and must be explicit:

* `mean` — default (`BaseScorer.aggregate`). "How good on average."
* `min` / quantile — "worst-case quality," the right estimator when one
  critical field must never fail. Expressible today by overriding
  `aggregate` on any scorer; no core surface needed, because the estimator
  lives in the scoring module, not the spine.
* `binary`, `weighted`, `per-field` — the business hierarchy ("this field is
  critical, these are nice-to-have") is encoded in the scorer *composition*
  (`field_wise_weighted`, `weighted_sum`, `binary`), and the scorer's `repr`
  round-trips so the rubric is a persisted, inspectable symbolic expression.

The rule: **higher is always better, always in [0,1]** (error metrics invert
internally), so any optimizer consuming `fitness` never needs to know the
family of the primitives underneath.

### The exam is discrete gradient descent

The user's framing, made exact: a benchmark is the discrete analogue of a
loss surface. `S` is the loss; `D` is the finite set of points where the loss
is probed; `run(system)` is one evaluation of the loss at one point `p` of the
program space; an optimizer (DSPy, TextGrad, a prompt optimizer, RL) is the
descent algorithm that consumes `as_loss()`. Benchy's own push cycles are the
same pattern turned inward: each cycle is a step in *complexity* space, and
"if nothing broke, the step was too small" is the learning-rate rule.

## 3. The compiler (the third pillar)

The rant's hardest question: what is the thing that *runs* the program so it
can sit the exam? There is no universal compiler for learned programs —
vLLM serves some architectures, llama.cpp others, PyTorch/Frida/exotic
old-school ML each need their own runtime, and a workflow is a composition
of runtimes. Mojo may or may not build the general one. Benchy must not wait
for it.

The bare-metal move is: **benchy does not build the compiler; benchy defines
its boundary.** The boundary is three contracts in the spine:

1. **`System`** — the opaque exam-taker. `invoke(Request) -> Response` plus a
   `Capabilities` record. Whatever runtime actually executes the learned
   program — weights on GPU, an HTTP endpoint, a subprocess — is invisible.
   Loader URLs (`openai:`, `endpoint:`, `hf:`, `python:`) are the *registry of
   compilers*, progressively extended, and that universe is deliberately
   deferred.
2. **`Task.render` / `Task.parse`** — the two coercions at the boundary.
   `render(sample, caps) -> Request` turns an exam case into the compiler's
   input format (the rant: "the output scheme of the exam has to match the
   input scheme of the next process"); `parse(response, caps) -> Prediction`
   turns the compiler's output back into the exam's answer format. **All the
   real engineering of the third pillar lives in this seam**, and it is owned
   by the Task because the Task owns the I/O contract.
3. **`Capabilities`** — the negotiation record. `render` branches on it:
   native structured output gets a schema-constrained request; a system
   without it gets the schema in the prompt and repair on the way back.
   Same for audio-in vs transcribe-then-prompt. The author never sees it.

The **AI API** view is the same boundary seen from outside: `Request`/`
Response` is a transport-free wire format (typed content parts + optional
output schema) and any system lowers it into its own transport. That is the
protocol by which information is exchanged with an AI.

### Composition (workflows) without a composition algebra

A workflow — segment, then extract per segment, then merge — is a composition
of learned programs. The spine needs no composition algebra, and adding one
would be abstraction creep: because `System` is opaque, **any composition is
just another System**, and the `python:` loader is the universal constructor —
the host language is the composition algebra. A workflow author writes a
python module that calls the three systems and exposes `invoke`; benchy
grades it identically to a raw model. (What a workflow *node* is inside that
python file is out of scope, by design.)

## 4. The Cutting Theorem

The golem needs a rule, not a mood. This is the rule, and it is checkable:

> A name in `benchy/core.py` is bare metal **iff** at least one holds:
>
> **(a) Branch** — a protocol method branches on it (`Capabilities.accepts`
> branches on the input-modality flags; `render` branches on
> `structured_output`).
> **(b) Consumer** — production code outside the definition site reads it
> (`max_concurrency` is read by the run loop; `Score.scorer` by the report;
> `Scorer.fitness` by `loss.as_metric`).
> **(c) Traffic** — a wire is proven by traffic, not topology: something at
> the far end of the seam reads it *today*. A "minimal information-preserving
> carrier" nobody reads is a souvenir, not wire.
>
> Everything else is cut. The evidence classes rank:
> **production reader > production writer > test-only anything.**
> Test-only consumers lock contracts; they are not life. Writers without
> readers are superstition. Readers imply their writers; writers imply
> nothing.

Applied to the Sept-01 spine, this theorem executes a real cut (see the
golem report for the breakage log):

| Surface | Verdict | Evidence (grep-exhaustive) |
|---|---|---|
| `Capabilities.kind` + `SystemKind` | **CUT** (cycle 1) | 9 re-exports, 4 writers, **0 readers**. Nothing in any protocol branches on the system's shape — opacity is the primitive, so a "kind" tag is decoration. |
| `Capabilities.tools`, `streaming`, `batch` | **CUT** (cycle 1) | **0 readers** anywhere in the spine or the five modules. |
| `Capabilities.video_in` | **CUT** (cycle 1) | **0 readers** and no `VideoPart` even exists. |
| `Capabilities.context_tokens` | **CUT** (cycle 1) | **0 readers**. Length-truncation is the compiler's job, not the exam's. |
| `Data.split` | **CUT** (cycle 1) | **0 callers**; split selection already lives at the *source* level (`jsonl:`/`hf:` sources), so the protocol method is duplicated surface. Partitioning for the optimize/evaluate loop is done by constructing `Data` per partition. |
| `Capabilities` after the cuts | **LOCKED** | Exactly the five fields that satisfy (a)/(b): `text_in`, `image_in`, `audio_in` (branch via `accepts`), `structured_output` (branch via `render`/`parse`), `max_concurrency` (consumer: run loop). Locked by test — adding a sixth field must be a deliberate, test-breaking act. |
| `Response.raw` | **CUT** (cycle 2) | Cycle 1 retained it on a clause-(c) citation that was **false**: 5 writers (system adapters), **0 readers** — the "carrier to custom parse_fns" existed only in the superseded plan. A souvenir, not wire. |
| `System.aclose` | **CUT from protocol** (cycle 2) | **0 production callers** in all 9 trees. Lifecycle belongs to the system's owner; the exam only mandates `invoke`. `BaseSystem.aclose` survives as concrete adapter convenience (its own tests assert it); the protocol must not mandate what no engine calls. |
| `ParseFailure` | **CUT** (cycle 2) | **0 raisers, 0 catchers** — a fossil of the superseded raise-on-parse design. The landed contract is `Prediction.parse_ok=False` (locked by `task/test_parse_never_raises.py`); parse errors flow through `Record.error`, not exceptions. |

The post-cut `Capabilities` is the theorem made flesh: *a capability exists
iff the exam-rendering branches on it.*

## 5. The global minimum, and why we can claim it

A "global minima" cannot be proven the way calculus proves one — the program
space of possible benchy designs is not a smooth manifold. What bare metal
can claim is **minimality by exhaustion**: enumerate every public name in
the frozen spine and show each satisfies (a), (b), or (c) with grep-evidence.
That is what the table above and the golem's consumer-map reports do. The
claim is falsifiable and re-checked every push cycle: if a name survives
that cannot cite a branch, a consumer, or a wire, the claim is false and the
name is cut.

What is deliberately **not** minimized (deferred, not denied):

* The registry of compilers (`benchy.system` loaders). Progressive, open-ended.
* The scoring primitives library. Composable and cheap; each must justify
  itself but the *algebra* is the product.
* The old `src/` tree until the gated nuke (Round 4).

And what moves the minimum: only the vision. The Sept-01 rant moved it once —
the compiler pillar and the estimator clause entered the theory, the dead
surface left the spine, and the golem's "push no further" verdict of
report-006 was vacated because it graded a ghost (the superseded
`AISystem.__call__` primitive, not the landed `System.invoke` + bridge).

## 6. The iteration protocol, formalized

Each push cycle is one step of discrete descent in complexity space:

1. **Move the theory** (or confirm it) — the loss function is the vision.
2. **Push** — delete or merge surface per the Cutting Theorem; edit the spine
   lock tests first, then the code.
3. **Accept the step iff something broke** — breakage is the evidence the
   step was real. Green-on-first-run means the step was too small; take
   another. ("If nothing broke, we're not pushing hard enough.")
4. **Repair** — fix the broken consumers; each repair must cite the theorem.
5. **Prove the new minimum by exhaustion** — consumer-map every name, write
   the golem report, and PASS requires either breakage in the cycle or an
   exhaustion proof. "PASS" without one of those two is vacated.

The stopping rule is not "it feels simple." It is: every remaining name cites
its branch, its consumer, or its wire — and the vision has stopped moving.