# SEED — Roadmap Phase-0 Campaign (verbatim prompt, 2026-09-02)

ok si benchy-semantics representa la capa semantica que se describe en el programa

in: .yaml
out: benchmark run

y benchy engine representa el motor para hacer el programa representado como 

in: .yaml
out: benchmark run

and benchy agent is the program: 

in: human request
out: .yaml


what are we missing then? can we make that roadmap? i guess what's missing the most is the mapping from benchy semantic to benchy engine and make sure that is correct?

store me the last prompt in markdown and then continue

store me the last prompt in markdown and then continue

---

## Verbatim follow-up steering (same session, later turn)

ok deja escrito esto como pendientes, explicame como es la capa semantica y el mapeo de semantica a engine, asegurate que los cambios que propusiste puedan tolerar cambios de diseño en la capa semantica (todavia estoy cerrando el diseño) y luego preparate para arrancar el desarrollo.

para desarrollar esto vamos a usar la master prompt de search and learn, usando 3 modelos kimik3, glm 5.3 y minmax3, cada uno spawnea 3 worktress independientes con 3 subagentes cada uno. entonces 9 subagents y 3 worktrees total por modelo, 27 subagents y 9 worktrees total. 

Esto es recontra importante salga bien. Aloca un budged de 300usd max

(para desarrollar esto vamos a usar la master prompt de search and learn, usando 3 modelos kimik3, glm 5.3 y minmax3, cada uno spawnea 3 worktress independientes con 3 subagentes cada uno. entonces 9 subagents y 3 worktrees total por modelo, 27 subagents y 9 worktrees total.

Esto es recontra importante salga bien. Aloca un budged de 300usd max — re-sent, weight signal)

## Interpretation (orchestrator notes, not verbatim)

- Three named models, ALL on Together AI — our only provider, credits aplenty
  (user re-sent this twice: weight signal). Verified live via chat completions
  on api.together.xyz, 2026-09-03: `moonshotai/Kimi-K3`, `zai-org/GLM-5.3`,
  `MiniMaxAI/MiniMax-M3`. NOTE: the WAF 403-blocks urllib's user-agent
  (error 1010) — use curl for direct probes.
- 9 worktrees total (3 per model), 3 subagent sessions per worktree (27 sessions).
- Budget: $300 max total.
- Mission: the semantics layer (yaml -> live exam) with a **correct-by-construction
  mapping to the engine**, tolerant to design changes in the semantic layer itself
  ("todavía estoy cerrando el diseño" — the user is still closing the design).
- The engine is frozen: `benchy/core.py` spine (27 names, System = invoke-only,
  Response 5 fields) + the 5 worktree modules (task/scoring/system/data/engine),
  all green at the composition gate (72/72), `make test` 431p/43s.
- What the user asked to be WRITTEN as pending (already in ROADMAP.md and the todo
  list): fase 0 loop closure (compile_run + run:-kwargs mapping + round-trip law +
  fingerprint-in-report), fase 1 agent skills, fase 2 GUI, fase 3 data fidelity,
  fase 4 Round-2 merge + CLI unification.
- Correctness of the semantics→engine mapping is THE deliverable of this campaign,
  tested by round-trip laws, not by "seems to work".
- Design-change tolerance requirement: the mapping must survive the user closing
  the semantic-layer design. That means: compile via the registries only (no
  reflection over module internals), pin the mapping with round-trip tests that
  read the registries themselves (parametrized, not hardcoded), and keep an
  adaptation seam (the compiler's resolver is the ONLY file that knows how
  names map to calls). If the user renames/restructures the semantic layer, the
  mapping pins survive if they read `describe()`-style registry data instead of
  hardcoded lists.

---

## Verbatim steering, 2026-09-03 (design conversation — the semantic layer)

algunos comments importantes: 

parece un error definir las tasks en categorias desde el vamos, porque establece limites no muy claros. en realidad la tarea es el programa, el input-output schema, luego ese input output schema se clasifica en una tarea. porque si vos tenes un texto y un json de output, es necesariamente un extract? puede ser un classify en texto? cuales son esas reglas? creo que todavia no lo sabemos, pero el nucleo es el programa, y el programa se define como su input output schema. y luego la scoring function se deriva del input-output schema, con posibilidad de cambiarle los pesos o hacer que sea un test binario de pass si todo esta bien, fail si al menos una cosa esta mal (que tal vez se puede representar con pesos, pero no se me ocurre ahora mismo).

luego el /<domain> se deriva de la data, junto con /<language>
y asi lleva la ontologia de /<task?>/<domain?>/<language?>, pero no se limita desde el vamos.

luego cosas de ingenieria o programacion mas a bajo nivel si usar kwargs, que clases, etc, es secundario. lo principla es definir la ontologia, la forma de pensar indicada en este proyecto para lograr la vision, para transmitir una forma particular de ver el mundo que creemos es la mejor para hacer benchmarks. igual eventualmente vamos a bajar ahi, pero no todavia.

exam.data se ve suficientemente bueno, tal vez a mejorar luego. lo mismo con el system. son piezas de ingenieria que ahora funcionan, luego vemos si las mejoramos o las rediseñamos.

querria pensar en la forma del .yaml, porque se puede diseñar/representar de varias formas y quisiera llega a algo que me guste y me parezca que es pura señal.

## Interpretation (orchestrator notes, not verbatim — 2026-09-03 steering)

- PROGRAM-FIRST: the task IS the program, defined by its input-output
  schema. Task categories (classification/freeform/extraction/transcription)
  are DERIVED classifications, not primary constructors. The classification
  rules are honestly unknown; the classifier is a registry function that
  evolves. The current grammar (task: {freeform: ...}) leads with the category
  — that ordering is the error being corrected.
- SCORING derives from the output schema, with two override knobs: weights,
  and all-or-nothing binary (pass iff everything right, fail if one thing
  wrong). Open question: binary as weights vs as a different aggregate kind.
- ONTOLOGY = /<task?>/<domain?>/<language?> — task from schema classification,
  domain+language from the data; all segments optional, never a gate.
- Engineering level (kwargs, classes) is secondary; the ontology/worldview is
  THE deliverable — transmit a way of seeing benchmarks. exam.data and system:
  good enough, OUT OF SCOPE for this redesign round.
- Campaign impact: round-2 angles become yaml-SHAPE explorations under
  program-first laws, not task-category-first grammars. The target artifact is
  "pura señal" — the yaml form the user likes.