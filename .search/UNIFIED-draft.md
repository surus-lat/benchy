# UNIFIED.md (BORRADOR v1 — round 1) — benchy bare-metal, 8/10 árboles cerrados

Estado: BORRADOR. Se integra s09 (runner) y s10 (salvage/archaeology) cuando cierren.
Fuente: reporte del analista UNIFY (deleg_cbce5631, 2026-09-02 01:35 -03), verificado
read-only contra los 10 worktrees, ley docs y código final de cada árbol.

## Campaña — 8/10 CERRADOS

| árbol | ángulo | estado | ciclos | loc | concepts | BM/HP/NR | commit |
|---|---|---|---|---|---|---|---|
| s01 | loss-first | CERRADO | 15 | 34 | 1 | 6/5/4 | a2d5b73 |
| s02 | exam | CERRADO | 15 | 162 | 7 | 1/10/4 | d0e84a6 |
| s03 | compiler/AI-API | CERRADO | 15 | 207 | 10 | 1/8/6 | b4b2f9d |
| s04 | onefile | CERRADO | 15 | 99 | 3 | 2/5/8 | a57a38d |
| s05 | directory | CERRADO | 15 | 72 | 4 | 3/8/4 | 00885e8 |
| s06 | contracts-first | CERRADO | 15 | 46 | 3 | 4/9/2 | 28b007c |
| s07 | datacentric | CERRADO | 4/15→15 | 87 | 7 | 8/7/0 | c1eca58 |
| s08 | cli | CERRADO | 15 | 89 | 4 | 3/5/7 | a153a1a |
| s09 | runner | **CERRADO** | 15 | 121 | 7 | 5/8/2 | f413abe |
| s10 | salvage | EN VUELO (c19+deliverables, deleg_09f21079) | 18 | 101 | 5 | 9/4/5 | aa6fd9a |

## Invariantes convergentes (sobrevivieron deleción en ≥2 árboles)

1. **as_loss es el concepto más confirmado** — BARE_METAL en 7 árboles (s01,s03,s05,s06,s07,s09,s10). loss = 1−score; lower=better es semántica del optimizer, no convención (s01 c14: score-as-loss rompe loss(s) < loss(best)).
2. **Loader/locate en el ontology path es metal** — el ladrillo load-bearing de "benchmark = data, never required Python". La divergencia dir-name vs path-field se resolvió a favor de **no-second-address** (s08/s09: un `path` field duplica el filesystem).
3. **CLI offline es metal** — "un engine solo alcanzable vía pytest es arqueología" (s06 c10, s07 c3, s10 c12). El UX entero = **dos verbos**: `new` (scaffold — crear benchmarks es el paso 1 de VISION, s08 c15) y `run` (s08); `report` murió (el artifact graduado ES el report).
4. **El artifact interpreta SOLO** — per-case input/expected/prediction/score + aggregate. **Two-tier law (s07 c15)**: un FILE en disco carga su identidad (system, benchmark, total); un return value puede apoyarse en su caller.
5. **Validación loud en LOAD** — keys desconocidas/faltantes levantan en palabras, en LOAD, nunca mid-exam (cloud money). **Declared beats inferred** (s07 c1: derivar answer space de samples hace typos→clases).
6. **System = el ARGUMENTO, spec compiled a callable una vez por exam** — invoke(input)->pred es un SHAPE, no una función (s04 c15); compile una vez en el boundary (s04 c9: per-case import era bug de corrección); **dos gates** (s07 c12: load = data entry, invoke = argument entry).
7. **Wrapper classes alrededor de data parseada = el noise más loud** — murió en todos los árboles que los tuvieron (s02: 5 clases, s03: 5, s04/s05/s07/s10). Named types/protocols/defaults = annotation-cargo (todo el arco s06).
8. **Scoring deriva del output schema** — IDEAS.md línea 1 confirmada (s04 c14, s08 c11, s09 c10): dispatch en el SHAPE del want (scalar→exact; dict→weights map, ausencia=binario); weights son data loud QUE DEBE RANKEAR (s09 c10). Grading **subsume enum gates** (s04 c4: predicción out-of-enum no matchea ningún want). Complementario (no duplicado) con la enforcement del answer key (s10 c8, s07, s09).
9. **Runner metal (s09)**: ThreadPoolExecutor (serial falla el bar 1000-case 16x; threads mantienen SYSTEM sync — viral async rechazado); retries con flaky-kind como único probe honesto; **atomic tmp+rename sirve al LECTOR** (reader-poll: torn 128/2225); **resume por content identity** (per-record input/want subsume exam-name/fingerprint); total=exam size fijo; errors=projection; **exit 0 iff cero errors (operabilidad, no calidad — mutation-tested c13)**.
10. **--limit N es el único flag metal** (s08 c10: smoke valve contra cloud spend; artifact carga total para que un smoke no se haga pasar por full).
11. **Meta-laws**: claimed-but-untested = escondite del noise — guard test PRIMERO (s01 c11/c15, s08 law-b, s09 judge-sharpening); spec escrita en 3 lugares = 2 de drift (s06 core.py); docstrings facturan loc, `#` comments gratis; **jueces adversariales por construcción** (SIGKILL→reader-poll; score-gate mutation sobrevivió→discriminator); two-sided bounds para claims de performance.

## Divergencias (el UNIFY debe decidir; read del analista)

1. **Clase vs funciones libres** — la más aguda. s03 (cero clases) y s10 (funciones puras) vs s06 c12/s08 c5/s09 (clase restaurada BARE_METAL: law 6 escribe el invariant en syntax de método `benchmark.run(system)`; la clase es concept-compressor que esconde el shape interno). **Read del analista: KEEP la clase** — la ley es la spec, 3 árboles la restauraron bajo deleción independientemente.
2. **Un file vs directorio** para el benchmark: s07/s08/s10 (exam.json único) vs s02/s03/s05 (task/scoring/cases/systems separados). Directory gana en diffability/ownership; un file en read-in-a-sitting. No forzar una sola forma en el UNIFY.
3. **Grado de task enforcement**: pass-through (s01-s06) vs **enforced declaration** (s07/s09/s10 — refuse en load). Steering 4-pillar + evidencia favorece **enforced**. Complementario con grading-subsumes-enum-gates (s04 c4 scorea la predicción; enforcement valida el answer key).
4. **Naming del case record**: input/expected (7 árboles) vs in/want (s01/s04/s09). Mayoría + "one name, one address" → **input/expected**. Index-as-id (s07 c10, s10 c9); s09 necesita per-record identity para resume, pero su propio c7 probó content>name → ids redundantes ahí también.
5. **Write contract**: s09 (atomic+resume+exit codes, metal bajo su bar) vs s05 (stdout ES el artifact sin resume — ambos honestos, **bar relativo al árbol**). Steering cloud-first → el engine unificado adopta el contrato s09.
6. **Cloud spec shape**: s03 _backend_http (openai-compatible, stdlib urllib) es el único que construyó el taker cloud; s05 "un cloud endpoint es un spec shape más". Adoptar la tabla backend s03 (stub/http/chain/agent — chain/agent como config pura, CERO cambios core) + argument-entry gate s07 + escape-hatch s05 c12+c14 (patterns/data son el escape hatch; stdlib es el backend).

## Shape unificado recomendado (para el build del unify)

```
benchmark = DATA: un directorio por exam (dir name IS el ontology path, sin path field)
  benchmark.json = {task: {input, output.choices}, scoring: {rule, aggregate} | weights,
                    cases: [{input, expected}, ...]}   (exact-set check, unknown keys raise)
  systems/        = system specs como data (kind-keyed: stub/keyword/regex/http/cloud/...)
engine nb/: clase Exam (concept-compressor) — run(system) / as_loss()
  load-time: validación loud (keys exactas, answer space declarado, per-case contract, non-empty)
  compile: spec -> invoke(input)->pred UNA vez por exam, en el boundary de run(); dos gates (load/invoke)
  scoring: dispatch en el shape del want; weights = data loud que rankea; grading compone loss UNA vez
  runner: fan-out ThreadPoolExecutor, tries con flaky probe, atomic tmp+rename write,
          resume por content identity, total=exam size, score=sum/total, errors=projection
  as_loss(): (System) -> float, loss = 1 − score (lower better)
CLI: benchy new <name> + benchy run <bench> <system> [--limit N]
     stdout ack = una línea humana (s10 c12); runs/<...>.json = el artifact para programas
```

Estimación: ~130–180 loc, 6–8 concepts, stdlib-only, 0 deps — la unión del runner metal s09 (136 loc) con los cores compactos (s06 46 / s08 89).

## Caveats para el build unify

- s09/s10 moving targets — re-verificar estado final en commit conocido antes de cosechar; el veredicto archaeology de s10 es input declarado del unify.
- s01 caveat: cases cargan el task solo por ejemplo — el task block enforced (s07/s09/s10) lo responde.
- **BUG sistémico**: .gitignore `*.json` untrackea data — el árbol unificado necesita negación role-scoped (`!bench/**/benchmark.json`) desde el commit 1.
- Dos variantes de golem.py — estandarizar en la py3.9-fallback ANTES del round 2.
- SUMMARYs s01 (62), s02 (72), s03 (71) exceden el bar ≤50 — s04-s08 dentro. No corregir retrospectivamente (los árboles ya cerraron), pero el bar se aplica al UNIFIED.md y round 2.

## Notas de divergencia adicionales (post-reporte, orquestador)

- **Identidad del artifact (s09 c11, llegó post-reporte)**: content identity subsume exam-NAME pero NO system identity — el system es la variable libre del loss; el echo `system` es el record de provenance (evita mezcla de evidencia entre exam-takers). Refina el invariante 4: files carry {content identity + system provenance}.
- **Asimetría format s10**: baseline como prosa `#` (golem cycles == push cycles) — los otros 9 lo hacen así; s10 normalizado por el orquestador en 8c9918c.
- **stdout: tres posiciones con evidencia** — s05 (stdout ES el artifact), s08 c13 (stdout solo eco del artifact), s10 c12 (ack line humana + artifact file para programas). El UNIFY debe dirimir por bar: sin resume → stdout-artifact OK; con resume → s09 contract + ack line s10.
- **Arqueología s10 (veredicto: hipótesis confirmada ESTRECHAMENTE)** — la maquinaria vieja rechazada (counts derivables, exit policies/resume/spec-registries = territorio de otros ángulos); sobrevivieron 3 donaciones, todas BARE_METAL bajo probe-delete, todas anticipadas a medias por las leyes from-zero del propio árbol: **c16 failures-are-evidence** (Exception per case → prediction=None+error en fila, score 0, QUEDA en denominador /6 no /5 — divergencia deliberada con el viejo que excluía errores del aggregate: la confiabilidad aterriza en el scalar del optimizer; probe-delete rompió 14 tests), **c17 empty-exam-refused** (sin él as_loss crashea con ZeroDivisionError — crash ≠ refusal), **c18 path-coherence check** (declared path ≠ requested → refuse; instalación rota gradearía bajo identidad equivocada). Lección transversal: **el valor del viejo sistema estaba en sus historias de failure, no en su código**.
- **Divergencia path-field (s10 c18 vs s09 c12) RESUELTA como ley de dos lados**: no declarado → el dir ES el path (s09 c12 borró el field como cargo no leído); declarado sin chequear → lie (s08 c9 unenforced-claim); declarado Y enforced → coherence metal (s10 c18). "One name, one address" aplicado al path: o no existe, o se enforcea.
- **Divergencia failures-in-aggregate (s10 c16 vs s09 errors-projection)**: s10 pone el fallo DENTRO del loss (score 0, queda en denominador — reliability en el scalar); s09 lo proyecta a errors→exit code (operabilidad ≠ calidad). Compatibles bajo steering cloud-first: fallo permanente → score 0 en denominador (el optimizer lo ve) Y errors>0 → exit 1 (el operador lo ve). Ambos honestos; el UNIFY adopta ambos canales.