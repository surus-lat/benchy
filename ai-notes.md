    sí y no. el YAML es la respuesta correcta a
    cómo se especifica un benchmark, pero es un
    front-end, no el substrato. te separo las dos
    cosas, porque ahí vive exactamente el límite
    de ingeniería que preguntás.

    **1. por qué el YAML no puede SER el
    benchmark**

    el YAML captura datos: schemas, paths,
    nombres, configs. pero un benchmark tiene
    comportamiento: parse es una función,
    grade es una función, render es una
    función. comportamiento no se serializa — se
    nombra.

    la solución es un compromiso de dos capas que
    ya está medio construido en benchy sin que lo
    llamáramos así:

    - registros de combinadores cerrados:
    exact_match, field_wise,
    builtin.freeform,
    builtin.structured_extraction, jsonl:,
    hf:, openai:, echo:, python:. todo lo
    que el YAML puede nombrar, alguien lo
    pre-ingenieró primero.
    - escape universal python:: cuando la
    semántica no está en el registro, el YAML
    referencia código. el host language es el
    álgebra de composición — esto lo lockeamos
    este ciclo con el test Seam 1b: un workflow
    de 2 etapas cargado vía python: y graduado
    por el engine idéntico a un modelo.

    o sea: YAML ⊂ python en expresividad, por
    construcción (los nombres del YAML resuelven
    a objetos python). el YAML es *azúcar
    estructurada sobre los registros* que ya
    existen. los strings "openai:gpt-4o",
    "jsonl:datos.jsonl", "qa/general/en" ya
    SON un formato de serialización — el YAML
    solo les da estructura.

    un detalle que el AGENTS.md ya insinúa: en
    realidad son tres documentos, no uno: el
    examen (task+data+scoring), el **sistema
    bajo prueba** (modelo, url, params), y el
    runner (budget, checkpointing, límites).
    mezclarlos en un YAML solo funciona si el
    runner está implícito.

    2. el pipeline duro: del YAML al puntaje

    acá está la ingeniería real, capa por capa —
    con dónde vive hoy:


    yaml → loader (intérprete: yaml → grafo de
    objetos vía registros + python:)
         → Benchmark (el examen vivo)
         → run loop (el único conductor con
    estado)
             por sample: render → invoke → parse
    → grade
             confiabilidad: concurrencia, retry,
    checkpoint, cache
         → aggregate: g(notas) = fitness
         → Report / run_outcome.json


    - loader: trivial mientras todo esté en
    los registros; el trabajo real es *validar
    temprano* (schema violation en carga, no en
    mitad del run).
    - harness de inferencia (adapters): la
    única parte que toca el mundo — red, auth,
    GPUs, encoding de audio, streaming.
    openai_system.py enruta chat vs
    transcriptions según capabilities; los HF
    pipelines cargan modelos. nada de esto es
    declarativo porque es *adaptación a APIs
    ajenas que cambian*.
    - run loop: orquestación async — semáforo
    por max_concurrency, backpressure. ya
    existe (~600 líneas en benchmark.py).
    - confiabilidad: retry con backoff,
    timeouts, y lo más sutil — **identidad
    estable para resume**: cache keys
    deterministas (_derive_key en el worktree
    de data), sample ids, task_status.json
    idempotente (reusar run_id salta lo
    completado). resume sin identidad estable es
    mentira.
    - taxonomía de fallos: error del sistema
    ≠ fallo de parse ≠ violación de schema. cada
    uno fluye distinto (Record.error vs
    parse_ok=False) y tiene distinta semántica de
    resume y costo. esta taxonomía es lo que hace
    un run auditable.
    - medición honesta: usage tokens y
    latency_ms se miden en el adapter y el report
    los confía. el adapter es la frontera de
    confianza de la medición — si miente ahí, el
    benchmark mide cualquier cosa.
    - agregación: g() sobre la distribución
    de notas. hoy media hardcodeada.

    3. quién corre el benchmark — tu gran pregunta

    la respuesta corta: **el engine, dirigido por
    un CLI, con los adapters como única interfaz
    al modelo**. la opacidad es lo que hace esto
    limpio: el run loop no sabe qué es el sistema
    — le habla invoke y nada más. el mismo loop
    corre echo, openai, un pipeline HF local, o
    un workflow compuesto.

    concretamente: `yaml → cli → Benchmark → run
    loop → adapters → puntaje`. benchy ya tiene
    TODO excepto el front-end declarativo — y por
    eso existe el landmine de las dos CLIs que se
    contradicen (engine/cli.py vs
    wt/cli/eval.py): el YAML es la respuesta a
    cuál de las dos debería sobrevivir, porque
    describe lo que ambas hardcodean como flags.

    4. los limitantes que no se diseñan fuera

    - comportamiento ≠ dato (L1 arriba). el
    registro es una deuda: cada cosa que el YAML
    puede nombrar, alguien la mantiene.
    - la semántica de fallo es conductual
    (retry, idempotencia, resume). no puede ser
    declarativa porque es reacción al mundo.
    - identidad y reproducibilidad: mismo
    YAML, otro día = otra versión del modelo,
    otro precio. el YAML especifica el examen;
    run_outcome.json es la evidencia del
    corrido (git, timestamps, modelo). faltaría
    el fingerprint del examen — hash de la
    definición, para que cambiar el YAML invalide
    comparabilidad de forma visible.
    - **g() determina mergeabilidad
    distribuida**: si algún día querés correr
    distribuido, la media es mergeable (sum+n es
    estadístico suficiente), un cuantil NO
    (necesita todos los valores o t-digest). la
    elección de estimador no es solo diseño —
    acota la ingeniería de ejecución. hoy
    media-only nos deja el camino abierto gratis.
    - presupuesto: el runner es dueño de
    plata. faltan spend caps, dry-run, cost
    accounting por run. es ingeniería de
    producto, no de teoría.
    - versionamiento del registro mismo:
    cuando exact_match cambia de semántica,
    scores viejos dejan de ser comparables. el
    registro necesita versionado igual que el
    YAML.

    5. mi lectura

    el YAML correcto es la fase final de Round
    2, no el primer paso: primero tiene que ganar
    la pelea de cuál CLI sobrevive (ciclo 3, ya
    filed), y el registro que el YAML nombra
    tiene que estar congelado por los locks. si
    construís el YAML antes, estás congelando
    superficie que el golem todavía está
    contrayendo — el mismo error que cometimos
    citando "wire" sin tráfico, pero en documento
    en vez de código.

    lo que sí es cierto: del YAML al puntaje, el
    tramo que NO existe aún es solo el primer
    centímetro (declarativo → grafo de objetos) y
    el último (presupuesto, fingerprint). todo el
    medio — el tramo realmente difícil, async +
    confiabilidad + medición — ya está construido
    y verde en el engine. la parte dura de
    ingeniería que intuís ya está hecha; falta la
    cáscara declarativa que la haga portable, y
    esa cáscara es peligrosa exactamente en
    proporción a cuánta superficie le permitas
    nombrar.
