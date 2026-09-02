# UNIFIED.md — benchy bare-metal search, ROUND 1 FINAL (10/10 árboles cerrados, 2026-09-02)

Mandato: rediseñar benchy desde cero bajo VISION.md, guiado por golem-search
de 10 ángulos independientes (~15 ciclos de push/deleción cada uno), luego UNIFY.
Steering (seed.md df70389 + ANGLES.md 6603fbe): la forma más simple de representar
la ontología de benchy a través de los 4 pilares — programa/task, scoring,
exam=data, compiler/ai-endpoint=system. Cloud-first (exam-taker = spec compilado
a un endpoint http). Serving local = largo plazo.

## La campaña — 10/10 CERRADOS

| árbol | ángulo | ciclos | loc | concepts | BM/HP/NR | esencia |
|---|---|---|---|---|---|---|
| s01 | loss-first | 15 | 34 | 1 | 6/5/4 | benchmark is a closure: load(path) returns the loss |
| s02 | exam | 15 | 162 | 7 | 1/10/4 | un exam es tres data files; dos verbos sobre dicts crudos |
| s03 | compiler | 15 | 207 | 10 | 1/8/6 | tres JSON + una free grading function; cero clases funcionó; _backend_http (el único taker cloud) |
| s04 | onefile | 15 | 99 | 3 | 2/5/8 | run() IS the compiler; score = shape-dispatch sobre el want |
| s05 | directory | 15 | 72 | 4 | 3/8/4 | stdout ES el artifact; un cloud endpoint es un spec shape más |
| s06 | contracts | 15 | 46 | 3 | 4/9/2 | the public surface is the spec; unnamed defaults are more honest |
| s07 | datacentric | 15 | 87 | 7 | 8/7/0 | exam = un data file loud-checked; two-tier interpret-alone law |
| s08 | cli | 15 | 89 | 4 | 3/5/7 | dos verbos + un flag; every deletion that broke revealed a second address |
| s09 | runner | 15 | 121 | 7 | 5/8/2 | exam-taking IS the metal; identity = content + system-provenance |
| s10 | salvage | 20 | 95 | 5 | 9/6/5 | arqueología confirmada estrechamente: 3 donaciones, 4 rechazos |

## Invariantes convergentes (sobrevivieron deleción en ≥2 árboles — la evidencia del unify)

1. **as_loss es el concepto más confirmado** — BARE_METAL en 7 árboles (s01,s03,s05,s06,s07,s09,s10). loss = 1−score; lower=better es semántica del optimizer (s01 c14: score-as-loss rompe loss(s) < loss(best)). Export seam: el optimizer consume loss(system) directamente (s10 c15: el import ES el pin).
2. **Loader/locate en el ontology path es metal** — el ladrillo load-bearing de "benchmark = data, never required Python" (s02 c15, s04 c13, s06 c4, s08 c6, s09 c6). **Ley de dos lados del path field**: no declarado → el dir ES el path (s09 c12 lo borró como cargo); declarado sin chequear → lie (s08 c9); declarado Y enforced → coherence metal (s10 c18: instalación rota gradearía bajo identidad equivocada). One name, one address: o no existe, o se enforcea.
3. **CLI offline es metal** — "un engine solo alcanzable vía pytest es arqueología" (s06 c10, s07 c3, s10 c12). El UX entero = **dos verbos**: `new` (scaffold — crear benchmarks es el paso 1 de VISION, s08 c15 lo probó: borrar new rompió el create-loop) y `run`. `report` murió (el artifact graduado ES el report). El único flag metal: `--limit N` (s08 c10 — smoke valve contra cloud spend; el artifact carga `total` para que un smoke no se haga pasar por full).
4. **El artifact interpreta SOLO** — per-case input/expected/prediction/score + aggregate; un file en disco carga su identidad (system, benchmark, total), un return value se apoya en su caller (s07 two-tier law). **Identidad = content + system provenance** (s09 c7+c11): content identity subsume exam-NAME identity (resume a través de un rename funciona); pero el echo `system` es metal — el system es la variable libre del loss; sin el echo, resume mezcla evidencia de dos exam-takers.
5. **Validación loud en LOAD** — keys desconocidas/faltantes levantan en palabras, en LOAD, nunca mid-exam (cloud money — s08 c14). **Declared beats inferred** (s07 c1: derivar answer space de samples hace typos→clases). **Declared == implemented** (s10 c11: el scoring declarado debe ser EXACTAMENTE el implementado — un check literal). Enforced task: grade rechaza keys fuera de task.output.choices (s10 c8), per-case contract enforced at load (s08 c14).
6. **System = el ARGUMENTO** — invoke(input)->pred es un SHAPE, no una función (s04 c15); compile UNA vez por exam en el boundary (s04 c9: per-case import era bug de corrección); **dos gates** (s07 c12: load = data entry, invoke = argument entry — candidates nunca pasan load). Grade seam universal (s10 c3): cualquier invoke(text)->pred es gradable — APIs reales, workflows, cached runs.
7. **Wrapper classes alrededor de data parseada = el noise más loud** — murió en todos los árboles que los tuvieron. PERO: la clase Exam como **concept-compressor** fue restaurada BARE_METAL en 3 árboles (s06 c12, s08 c5, s09) — law 6 escribe el invariant en method syntax `exam.run(system)`; s03/s10 (funciones puras) también pasan el bar. **Decisión UNIFY: keep the Exam class** — la ley es la spec.
8. **Scoring deriva del output schema** — IDEAS.md línea 1 confirmada (s04 c14, s08 c11, s09 c10): dispatch en el SHAPE del want (scalar→exact; dict→weights map). **Weights must RANK** (s09 c10: right-on-critical vence right-on-nice a igual field count, o los weights son una mentira). Grading **subsume enum gates** (s04 c4). Grading compone loss UNA vez; stdout/artifact lo LEEN (s08 c13 — la fórmula tenía tres direcciones).
9. **Runner metal (s09)**: ThreadPoolExecutor (serial falla el bar 1000-case 16x; threads mantienen SYSTEM sync — viral async rechazado); retries con flaky-kind (fails-count como data plana) como único probe honesto; **atomic tmp+rename sirve al LECTOR** (reader-poll: torn 128/2225 polls vs SIGKILL que pasó); **resume por content identity**; total=exam size (partial artifact interpreta solo); errors=projection (computa su único lector, el CLI); **exit 0 iff cero errors — operabilidad, no calidad** (s09 c13: score-gating mutación sobrevivió la suite → discriminator). **Defaults: los VALORES son metal** (s09 c15: borrar defaults rompió 13 tests — run(system) ES la superficie de la visión), sus NOMBRES eran cargo.
10. **Failures are evidence** (s10 c16, donación arqueológica): Exception per case → prediction=None+error en fila, score 0 y QUEDA en el denominador (/6 no /5) — divergencia deliberada con el viejo benchy: la confiabilidad aterriza en el scalar que consume el optimizer. **Empty exam = refusal** (s10 c17: sin él as_loss crashea ZeroDivisionError — crash ≠ refusal). Dos canales: fallo permanente → score 0 en denominador (el optimizer lo ve) Y errors>0 → exit 1 (el operador lo ve).
11. **Meta-laws del método**: claimed-but-untested = escondite del noise — guard test PRIMERO (s01, s08 law-b, s09, s10); **jueces adversariales por construcción** — SIGKILL→reader-poll, score-gate mutation→discriminator, sleep-skip mutant→lower bound (s09 c14: two-sided bounds — upper prueba fan-out, lower certifica trabajo); spec escrita en 3 lugares = 2 de drift; docstrings facturan loc, `#` comments gratis; mutation-test el propio engine antes de confiar en la suite (s09: 4 gaps encontrados así).

## Divergencias resueltas

1. **Clase vs funciones libres** → keep la clase Exam (concept-compressor; law 6 en method syntax; 3 restauraciones independientes). s03/s10 prueban que las funciones puras también pasan — la clase no es más metal que ellas, pero la ley la escribe así.
2. **Un file vs directorio** → no forzar. Directory gana en diffability/ownership; un file en read-in-a-sitting. Las invariantes transfieren igual.
3. **Task enforcement** → enforced (s07/s09/s10 + steering 4-pillar). Complementario con grading-subsumes-enum-gates: s04 scorea la predicción, la enforcement valida el answer key.
4. **Naming del case record** → input/expected (7 árboles + one-name-one-address). in/want queda como evidencia de s01/s04/s09 de que el nombre no es el metal.
5. **stdout** → tres posiciones honestas según bar: sin resume, stdout ES el artifact (s05); con resume contract, artifact file + ack line humana de UNA línea para personas (s10 c12) que no lleva nada que el artifact no tenga (s08 c13).
6. **Cloud spec shape** → tabla backend s03 (stub/http/chain/agent — chain/agent como config pura, CERO cambios core) + argument-entry gate s07 + "cloud endpoint = un spec shape más" (s05). s03 es el único que construyó el taker cloud (openai-compatible, stdlib urllib).

## Arqueología (s10) — el veredicto del ángulo salvage

**Hipótesis confirmada, estrechamente.** El viejo benchy sabía cosas que el search
from-zero perdió — pero no su maquinaria (counts: segunda dirección, rechazado c19;
exit policies, resume/status, spec registries: derivables o territorio de otros
ángulos). Sobrevivieron 3 donaciones, todas BARE_METAL bajo probe-delete, todas
anticipadas a medias por las leyes from-zero del propio árbol: failures-are-evidence
(c16, probe-delete rompió 14 tests), empty-exam-refused (c17), path-coherence (c18).
**El valor del viejo sistema estaba en sus historias de failure, no en su código.**

## Shape unificado recomendado (para el build post-unify)

```
benchmark = DATA: un directorio por exam (dir name IS el ontology path)
  benchmark.json = {task: {input, output.choices}, scoring: {rule, aggregate} | weights,
                    cases: [{input, expected}, ...]}   (exact-key check, unknown keys raise)
  systems/        = system specs como data (kind-keyed: stub/keyword/regex/http/chain/agent)
engine nb/: clase Exam (concept-compressor) — run(system, out=None) / as_loss(system)
  load-time: validación loud (keys exactas, answer space declarado, per-case contract,
             non-empty, declared==implemented, path-coherence)
  compile: spec -> invoke(input)->pred UNA vez por exam, en el boundary; dos gates
  scoring: dispatch en el shape del want; weights que RANKEAN; loss compuesto UNA vez
  runner: ThreadPoolExecutor fan-out, tries con flaky probe, atomic tmp+rename write,
          resume por content identity (+ system echo = provenance), total=exam size
  failures: Exception per case → score 0, error en fila, queda en denominador
  as_loss(): (System) -> float = 1 − score (lower better; export seam al optimizer)
CLI: benchy new <name> + benchy run <bench> <system> [--limit N]
     ack line humana de una línea; runs/<...>.json = artifact para programas
     exit 0 iff cero errors (operabilidad, no calidad)
```

Estimación: ~130–180 loc, 6–8 concepts, stdlib-only, 0 deps — la unión del runner
metal s09 (121 loc) con los cores compactos (s06 46 / s08 89 / s10 95).

## Caveats para round 2

- `.gitignore` raíz arreglado (a6a8cac): negación role-scoped para bench/ y runs/ — el trap `*.json` costó 3 árboles en round 1. Los worktrees `.search/` siguen gitignored por línea 81; su data vive en sus branches.
- Estandarizar golem.py en la variante py3.9-fallback antes del round 2.
- Bar SUMMARY ≤50 líneas con wc -l como juez — s01 (62), s02 (72), s03 (71), s09 (54→49) excedieron en primera pasada; s08 (59→49) también. El bar se aplica DESDE el primer draft en round 2.
- Dos-sided bounds + mutation-testing como práctica de serie para todo claim observable (la lección más cara del round 1: 4 judge gaps en s09 solos).
- s01 caveat resuelto por el task block enforced: cases cargan el task solo por ejemplo → el task declarado+enforzado responde dónde vive la descripción.

## Los 10 SUMMARY.md + LEARNINGS.md son la fuente completa

.search/s01..s10/{SUMMARY,LEARNINGS,ITERATIONS,DESIGN}.md — todos commiteados,
todos los árboles clean en sus branches search/s01..s10.