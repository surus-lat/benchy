# Benchy agent skills

Five skills, one per thing that is actually hard. They live at
`.agent/skills/<name>/SKILL.md` and are written as practical how-tos — an agent loads one
via the `Skill` tool; a human reads the file.

| skill | one-liner |
|---|---|
| [author-benchmark](./author-benchmark/SKILL.md) | Turn a business problem into a `benchmark.yaml`. The judgement calls: schema, weights, exam. |
| [run-and-interpret](./run-and-interpret/SKILL.md) | Run it, and read the result correctly — the three statuses mean different things. |
| [write-adapter](./write-adapter/SKILL.md) | Expose any AI-system through the one runtime contract. |
| [add-provider](./add-provider/SKILL.md) | Usually one line; occasionally a new request shape in `llm-client`. Telling them apart. |
| [extend-ontology](./extend-ontology/SKILL.md) | Add a task, and the four documents that must agree. |

## Why only five

The previous set had 21. Most described an architecture that no longer exists, and six of
them — `define-task`, `define-scoring`, `configure-model`, `setup-data`, `add-task`,
`run-benchmark` — collapsed into "edit the YAML" once the engine became declarative. That
collapse was the goal, not an accident, so they are not worth a skill each.

A skill here earns its place by carrying *judgement* that the code and docs cannot: which
fields deserve weight, what an `invalid_output` is really telling you, when a provider
needs a new request shape rather than a table row.
