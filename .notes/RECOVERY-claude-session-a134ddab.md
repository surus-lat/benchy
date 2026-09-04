# Recuperación de sesión Claude Code — a134ddab (cortada por límite 2026-09-01 ~19:24)

Fuente: ~/.claude/projects/-Users-dobleefe-benchy/a134ddab-9d37-48f5-b995-afe6abac9389.jsonl
Último mensaje: "You've hit your session limit · resets 10:40pm (America/Buenos_Aires)"
Último pedido del usuario sin respuesta: "can you at least export what is here?"

## Objetivo de la sesión

Leer VISION.md, re-derivar el diseño desde cero (no heredar el plan de junio),
y construir el nuevo benchy en 5 worktrees paralelos (uno por módulo).

## Las dos correcciones de diseño al spec de junio (CLAVE)

### 1. El system es el argumento, no un campo del constructor

June: `Benchmark(task, scoring, data, system)` — MAL.
La feature estrella de la visión es exportar el benchmark como "loss function"
para prompt-optimizers, y la variable libre de una loss es lo que se optimiza:

```python
Benchmark = Task + Data + Scoring        # el examen
await bench.run(system)                  # calificar un candidato
bench.as_loss()                          # (System) -> float
```

`as_loss` queda en 3 líneas y reutilizar un examen con muchos systems es el path nativo.

### 2. El bridge Task↔System vive en Task (June no lo tenía)

`System.run(sample) -> Prediction` NO puede funcionar: un system genérico
`openai:` no sabe nada de facturas. El prompt-wrap, el "respondé JSON", y el
parseo de la respuesta son conocimiento DEL TASK, no del system.

Candidatos evaluados: System (no, no conoce el task), un 5to módulo adapter
(no, es lo que hizo old benchy y por eso tiene interfaces/ + adapters/ +
god-objects), Task (SÍ — ya es dueño del contrato input/output).

```python
request    = task.render(sample, system.capabilities)   # Sample   -> Request
response   = await system.invoke(request)               # opaco
prediction = task.parse(response, system.capabilities)  # Response -> Prediction
```

`Capabilities` es la negociación: structured_output nativo vs schema-en-prompt
+ repair de JSON; audio_in vs CapabilityError. El autor del benchmark nunca lo ve.

## Estado de los 5 worktrees (verificado con git)

| worktree  | branch      | estado |
|-----------|-------------|--------|
| .worktrees/task    | wt/task    | DONE — commit 0da4d97, 174 tests verdes (27 spine + 147 nuevos). 9 archivos en benchy/task/: __init__, base, builtin, schema, media, repair, choice, render, registry |
| .worktrees/system  | wt/system  | DONE (3 commits) — echo: test double + loader registry; openai: y endpoint: schemes; python: scheme |
| .worktrees/scoring | wt/scoring | DONE (2 commits) — BaseScorer, registry, 12 primitivas atómicas; structural scorers, transforms, public API |
| .worktrees/data    | wt/data    | SIN COMMITS propios (en 018c811 = base) |
| .worktrees/engine  | wt/engine  | SIN COMMITS propios (en 1156f17 = REF/new-benchy HEAD) |

Nada está mergeado a REF/new-benchy todavía. El spine congelado tiene solo
benchy/__init__.py y benchy/core.py (contratos core) + la suite seam como
merge gate (commit 1156f17).

## Firma de Task (del reporte del subagente)

```python
Task(*, name: str, ontology: str | OntologyPath,
     input: BaseModel_subclass | dict | None = None,
     output: BaseModel_subclass | dict | None = None,
     instructions: str = "",
     mode: Literal["schema","text","choice"] = "schema",
     labels: Sequence[str] | None = None,          # requerido si mode="choice"
     template: str | None = None,
     render_fn: Callable[[Sample, Capabilities, Task], Request] | None = None,
     parse_fn: Callable[[Response, Capabilities, Task], Prediction] | None = None)
```

input/output=None → schema permisivo {"type": "object"}. Aceptan pydantic v2
BaseModel o dict JSON-schema.

## Ontología AI de la visión

`/<task?>/<domain?>/<language?>` — un ai system es un programa que hace un task
(raíz), luego dominio, luego lenguaje.

## Side notes

- Se guardó en gbrain la idea `writing-benchmark-is-the-loss-function` (corolario
  de `tweetable-scoring-function-as-loss`, en el arco de Instagram).
- 2 worktrees extra con prefijo agent-a592/a625 (del harness de Claude, en
  .claude/worktrees/) — ambos en 56da52f, probablemente exploratorios.

## Próximos pasos para retomar

1. Verificar qué falta en wt/data y wt/engine (los dos sin commits).
2. Correr la seam suite como merge gate antes de mergear cada worktree.
3. Mergear en orden: task, system, scoring, data, engine.
4. Responder el pedido pendiente: "export what is here" (probablemente exportar
   el estado/diseño — este archivo es eso).
