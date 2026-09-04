# HANDOFF — benchy bare-metal, sesión fresca

> **Para el agente del cluster**: este documento es TODO el contexto de una
> sesión de trabajo de días, destilado. Léelo completo antes de tocar nada.
> Tu misión no es "continuar un transcript" — es una sesión nueva que ya sabe
> todo lo que la sesión anterior sabía, y ejecuta la campaign de search+learn
> que está pendiente. No hay kit de campaign, briefs pre-escritos ni skills
> copiadas en este repo: **tú armas eso**. Este documento te dice qué saber,
> no qué copiar.

- Fecha del handoff: 2026-09-04
- Rama: `REF/new-benchy` (este commit). `main` está congelado atrás en 7c2f4e0.
- Test: `make test` → **991 passed, 43 skipped** (los 43 = system/hf sin red/hardware).
- Python 3.12+ (pyproject `requires-python >= 3.12`); venv vía `make setup-venv-default` o `uv venv`.

## 0. Qué es esto

Benchy, rediseñado a su bare metal. Un benchmark es **examen + sistema**:

- **Task** = schemas (in/out) + semántica de corrección (render/parse).
- **Data** = muestras conformes al schema in.
- **Scoring** = juez determinista (fitness g(distribución)).
- **System** = tomador opaco del examen (`invoke` — su única superficie).
- **Benchmark** = Task + Data + Scoring; el sistema es el argumento de `.run()`.

El motor está **congelado** (27 nombres públicos en `benchy/core.py`, 419 líneas).
Se rediseñó en ciclos de push teórico (cortar superficie muerta, falsificar
retenciones), con la teoría completa en `.plans/BARE-METAL-THEORY.md`.

## 1. Estado del repo

| Qué | Dónde | Estado |
|---|---|---|
| Spine | `benchy/core.py` + `benchy/__init__.py` | congelado, 27 nombres, no tocar |
| Capa declarativa | `benchy/spec.py` + `tests/benchy/integration/test_spec.py` | v1 VERDE, en exploración (grammar abierto) |
| Motor compuesto | `benchy/{benchmark,cli,loss,report,spec}.py` + subpaquetes `data/ scoring/ system/ task/` | completo in-tree, 991p verde |
| Tests | `tests/benchy/` | core contracts + seams + spec + por-módulo (scoring/task/system/data/engine) |
| Ronda 1 search | `search/{seed,UNIFIED,ANGLES,GOLEM}.md` + `search/golem.py` | 10/10 trees cerrados, 11 invariantes |
| Plan | `.plans/` | THEORY, DECLARATIVE-SPEC, ROADMAP, WORLD_MODEL, golem reports 001–008 |
| Engine viejo | `src/` | superseded, coexiste (CLI `benchy`), nuke pendiente de aprobación |
| Evidencia modelos | `.notes/TOGETHER-BENCH-2026-09-02.md` (tracked) | Kimi-K3 97% calidad/lento, GLM-5.3 flaky, MiniMax-M3 mejor costo |
| Precedentes yaml | `proto/bench/*.yaml` | extract (weights), sentiment, extract-together (cloud-first) |

**Identidad del paquete**: `benchy2 = benchy.cli:main` (verbos `new | list |
run | show | export-loss`). La CLI vieja `benchy = src.benchy_cli:main` sigue
andando; la unificación de las dos es fase-4 del roadmap (no hoy).

**Nota sobre `search/seed.md`**: contiene los prompts del usuario VERBATIM
(el original de la campaign, el 3×3×3/$300, y el steering program-first del
2026-09-03). Protocolo de integridad: son la fuente de autoridad del diseño —
léelos antes de decidir cualquier cosa de grammar, y nunca los reescribas de
memoria.

## 2. La teoría en 60 segundos

1. **Task = schemas + corrección**; Benchmark = Task+Data+Scoring; el System
   es el argumento. El "examen" es data + criterio, no el modelo.
2. **`System.invoke` es la única superficie del tomador** — protocolo de 1
   método, cero knobs de inferencia. Todo lo demás es engine.
3. **Registro-nativos**: task builtins, scorers, systems, data sources son
   **registros abiertos**, no jerarquías. `describe()` es el contrato del GUI.
4. **Paths relativos al YAML; two-sided path law** (los inputs se leen, los
   outputs se escriben); fingerprint sobre el núcleo compilado (reprs), no
   sobre el YAML.
5. **Validación loud en LOAD, no en run-time**: un yaml mal formado falla
   AL CARGAR, con error que nombra qué registros existen.
6. **Loss = 1 − score, lower=better** (`as_loss`), convergido 7-tree en ronda 1.
7. **Invariantes UNIFIED ronda 1** (10/10 arms): as_loss; two-sided path law;
   CLI new+run con `--limit`; artifact interpreta solo (identidad = content +
   system provenance); validación loud en load; defaults are values (not metal);
   failures-are-evidence (/6, no /5); two-sided bounds; enforced task.
8. **Teorema de corte** con cláusula (c) TRAFFIC: se corta superficie si
   no tiene lector prod > escritor prod > test-only. Evidence-first con grep
   exhaustivo + causalidad (stash → repro → unstash).

## 3. El grammar abierto (la tarea central del cluster)

El YAML de la capa declarativa. Lo que sabemos y lo que NO:

**Program-first (steering del usuario, verbatim en `search/seed.md`)**:
el autor escribe SOLO el programa (schemas in/out). La categoría es una
clasificación downstream del programa — NO un constructor declarado. La
ontología `/<task?>/<domain?>/<language?>` deriva (task de clasificar el
programa, domain+language de la data); es addressing, nunca gate. El scoring
deriva del out-schema; knobs = weights o binary (pass iff todo bien, fail si
una mal) — binary es un aggregate kind (`min` sobre field scores, cutoff 1.0),
ya representable como `binary: {inner: field_wise, cutoff: 1.0}`. exam.data y
system "suficientemente buenos" — OUT OF SCOPE este round.

Pipeline de compilación: `program → classify(in,out) → task del registro →
scoring deriva del out-schema → Benchmark`. Cero cambios al engine; lo nuevo
= un registro de clasificadores + una función de derivación de scoring. El
classifier es donde vive el conocimiento acumulable (text→enum = classification,
text→object = extraction... hasta reclasificar `{sentiment:[pos,neg]}` como
classify sin tocar un yaml). Ontología = función versionada, no string.

**Formas candidatas** (del diseño program-first, la campaign las explora):
- **A — maximally empty** (favorita, "pura señal"): `exam: name / program: {in: text, out: {vendor: str, total: number}} / scoring: {} / data: {jsonl: ...} / system: {url: ...}` — todo lo derivable NO se escribe; `scoring: {}` deriva del out.
- **B — knobs visibles** (lo que el GUI edita): igual + `scoring: {weights: {...}, aggregate: min}`.
- **C — classify declarando el enum**: `out: {label: [pos, neg, neu]}` → categorical accuracy. Declared beats inferred; JAMÁS inferir clases de las filas (lección s07: typos→clases).

**Principios de pureza** (juzgar las 9 formas): (1) maximally empty + deletion
test (¿borro la línea y el exam se comporta igual? era eco); (2) el programa
es lo que come/produce, el scoring es lo que el autor juzga (weights van en
`scoring:`, no inline); (3) derivar de schemas declarados, nunca de las filas
(domain/language = única excepción legítima); (4) ontología = addressing, no
gate; (5) fingerprint sobre el núcleo compilado, no el yaml.

**Eje abierto A PROPÓSITO**: el micro-lenguaje de los schemas (json-schema
completo vs shorthand python-ish `str/number/[a,b,c]` vs by-example). Parte
de lo que los arms exploran.

## 4. La campaign pendiente (tu trabajo)

Lo que el usuario pidió, verbatim en `search/seed.md`:
**3 modelos × 3 worktrees × 3 subagentes = 27 subagentes / 9 worktrees, budget
$300 max, "Esto es recontra importante salga bien."**

- **Provider**: TODO via Together AI (único). IDs exactos (casing literal):
  `moonshotai/Kimi-K3`, `zai-org/GLM-5.3`, `MiniMaxAI/MiniMax-M3`. Verificados
  vivos en 2026-09-02 (evidencia en `.notes/TOGETHER-BENCH-2026-09-02.md`).
  WAF: api.together.xyz da 403 "error code: 1010" al user-agent de urllib —
  probes HTTP con **curl**, no requests/urllib.
- **Qué explorar**: NO "implementar grammar v1". Explorar FORMAS: 9 arms =
  9 shapes de yaml bajo la misma law program-first, contra el engine congelado
  y el mismo bar (echo + fitness conocidos). Cada arm entrega un yaml que
  CORRE (no prosa).
- **Law + golem**: los de ronda 1 (`search/GOLEM.md` + `search/golem.py`) son
  el precedente; tú armas la law de ronda 2 con la cláusula program-first y
  la cláusula de tolerancia-mutación (mutar el grammar sin romper el
  fingerprint de un exam dado — tolerancia como test ejecutable). Lecciones
  de ronda 1 para la law: SUMMARY ≤50 líneas desde el primer draft,
  two-sided bounds + mutation testing, concept-scan, log en `#`-prose.
- **Despacho**: delegate_task hereda el modelo de la sesión → matriz
  3-modelos mixta requiere **waves separadas por modelo** (wave Kimi-K3,
  wave GLM-5.3, wave MiniMax-M3), 3 worktrees por wave, 3 subagentes por
  worktree. Kimi-K3 es razonador largo: ~50 llamadas por sesión — presupuesta
  accordingly.
- **Learn phase**: después del search, comparar las 9 formas vivas y extraer
  la que sea "pura señal" — ese es el criterio de juicio del usuario.
  Aprende de `search/UNIFIED.md` como formato de harvest (10/10 arms,
  invariantes numerados, divergencias resueltas con veredicto).
- **Bar idéntico entre árboles**: el golem define idéntico bar en todos los
  worktrees (mismos tests, mismo fingerprint). Los 8 worktrees existentes
  (`.worktrees/{scoring,task,system,data,engine,cli,adapters,core}`) son de
  ronda 1; crea los tuyos para ronda 2.
- **Presupuesto**: $300 max. "no super-API de scoring, no DSL de workflows,
  no optimizer surfaces antes del merge, no tocar el spine" (anti-goals del
  roadmap).

## 5. Landmines y reglas del repo

- **No tocar `benchy/core.py`** (spine congelado). Cambios al grammar viven en
  `benchy/spec.py` + registros; el engine no cambia.
- **La CLI vieja `src/` coexiste**: `benchy` (viejo) vs `benchy2` (nuevo).
  `make test` corre AMBAS suites. El nuke de `src/` requiere aprobación.
- **Worktrees con WIP ajeno**: wt/scoring tiene stash@{0} + untracked de
  ronda 1 — no lo pierdas; wt/cli tiene un ghost-import
  (`exact_match_scorer`) de un rewrite en vuelo.
- **Commits**: tags [ADD]/[MOD]/[REM]/[REF]/[HOT]/[FIX]/[MRG]/[DOC] con
  subject en inglés; SIN trailers Co-Authored-By (convención del repo).
- **Cuidado con `.gitignore`**: `.search/` y `.worktrees/` están ignorados;
  todo deliverable va a paths tracked (así llegaron seed/UNIFIED a `search/`).
- **`grep -E` sin lookahead; `\a` es BEL** (no un escape de regex); CPython
  3.12: `list()` llama `__len__` directo (TypeError si len desconocida —
  `sample()` debe pasar len_hint honesto).
- **No imprimir secrets**: TOGETHER_API_KEY vive en `~/.hermes/.env` del
  host, no en el repo; para campaigns, api_key_env en el yaml.
- **Test invocation canónica**: `cd <tree> && /Users/dobleefe/benchy/.venv/bin/python -m pytest tests/benchy -q --no-header -p no:cacheprovider`

  — en el cluster arma tu propio venv (`make setup-venv-default`); el path
  absoluto es de la máquina de Francis.

## 6. Próximos pasos inmediatos

1. `make test` → verificar 991 passed (gate de entrada).
2. Leer en orden: `search/seed.md` (prompts verbatim) →
   `.plans/BARE-METAL-THEORY.md` → `search/UNIFIED.md` →
   `.plans/ROADMAP.md` → este archivo de nuevo.
3. Armar la law de ronda 2 (program-first, bar idéntico, cláusula de
   tolerancia-mutación) + golem r2 + 9 angles (shapes A/B/C × variantes).
4. Crear worktrees de ronda 2 y despachar por waves de modelo (Kimi-K3 →
   GLM-5.3 → MiniMax-M3), 3 subagentes por worktree, $300 cap.
5. Search → learn → `UNIFIED-R2.md`: comparar 9 formas, extraer la que sea
   "pura señal", cerrar el grammar con evidencia.
6. After-search: fase 0 del roadmap (compile_run + `run:`→kwargs +
   round-trip law + fingerprint en report.meta).

## 7. Cómo correr cosas

```bash
make test                          # gate: 991 passed, 43 skipped
benchy2 list                       # walk benchmarks/, ontology tree
benchy2 new <ontology>             # scaffold benchmarks/<t>/<d>/<l>/
benchy2 run <ref> --system echo:   # run contra echo, imprime report
benchy2 show <ref>                 # task/scoring/data summary
benchy2 export-loss <ref> --to X   # exportar como loss function
```

La capa declarativa por código:

```python
from benchy.spec import compile_exam
bench = compile_exam("bench.yaml")  # yaml → Benchmark listo para .run()
```

## 8. Chronology (para orientarte)

1. Ronda 1 search (10 arms) → UNIFIED.md: 11 invariantes, 6 divergencias
   resueltas, veredicto s10 (3 donaciones BARE_METAL, machinery derivable).
2. Spine bare-metal: ciclos 1–2 de corte (cut `SystemKind`, `raw`,
   `aclose`, `ParseFailure`...), propagación a 8 worktrees, teorema TRAFFIC.
3. Capa declarativa v1 (`spec.py`): compile_exam/scoring/system/data,
  describe(), fingerprint sha256, 72/72.
4. [wip] 642e176: .plans/.notes/proto/docs + benchy core+spec+tests (110
   files/15,390 insertions).
5. Steering program-first (2026-09-03): task block muere, classify-then-derive,
   binary = min, formas A/B/C, principios de pureza, campaign = shape search.
6. **Este export** (2026-09-04): motor compuesto in-tree, seed+UNIFIED
   tracked, este handoff. La campaign ronda 2 es tuya.

Good hunting. — sesión origin (Francis's Mac, 2026-09-04)