# Bare-Metal Architecture for New Benchy Engine

## First Principles Reduction

From VISION.md, first principles:

1. **AI system = program that performs a task.** Not a model. Not a workflow. A *program*.
2. **Benchmark = exam for that program.** It has:
   - Input/output schema (what the program does)
   - Scoring function (how we grade it)
   - Data (the exam questions)
   - Target (which program is being examined)
3. **Ontology:** `/<task?>/<domain?>/<language?>` — task is root, domain is sublevel, language is third.

## The Bare Metal

Strip away everything that is not essential. What remains?

### Core Primitive: `AISystem`

An AI system is anything that takes input and produces output. It is a *callable*.

```python
class AISystem(Protocol):
    async def __call__(self, input: Sample) -> Output: ...
```

This is the bare metal. Everything else is plumbing.

### The Exam: `Benchmark`

A benchmark is a triple:
- `TaskSpec` — input/output schema + context
- `ScoringFunction` — how to grade
- `Dataset` — the exam questions

```python
@dataclass
class Benchmark:
    task: TaskSpec
    scoring: ScoringFunction
    data: Dataset

    async def run(self, system: AISystem) -> ExamResult: ...
    def as_loss(self) -> Callable[[AISystem], float]: ...
```

### The TaskSpec

A task is a function signature with semantics:

```yaml
task:
  name: invoice-extraction
  domain: document-understanding
  language: es
  input:
    type: document[pdf]
    description: "Scanned Argentine invoice"
  output:
    type: structured
    fields:
      - name: vendor_name
        type: string
        required: true
      - name: amount
        type: number
        required: true
```

This is the program description. It says: "I need a program that takes a PDF document and returns a JSON object with vendor_name and amount."

### The ScoringFunction

A scoring function maps (expected, actual) → score. It is pure, deterministic, and composable.

```python
class ScoringFunction(Protocol):
    def score(self, expected: Any, actual: Any) -> float: ...
    def aggregate(self, scores: list[float]) -> dict[str, float]: ...
```

### The Dataset

A dataset is a sequence of exam questions:

```python
@dataclass
class Sample:
    id: str
    input: Any          # the exam question
    expected: Any       # the correct answer
    context: dict | None = None  # optional context

class Dataset(Protocol):
    def __iter__(self) -> Iterator[Sample]: ...
    def __len__(self) -> int: ...
```

### The AISystem Adapter

The system under test is adapted to the `AISystem` protocol. Adapters handle:
- HTTP endpoints (any API)
- Local models (transformers, vLLM, llama.cpp)
- Cloud providers (OpenAI, Anthropic, etc.)
- Workflows (composed systems)
- Agents (systems with tools and loops)

The adapter is the *only* place where provider-specific logic lives.

### The Runner

The runner executes the exam:

```python
async def run_exam(benchmark: Benchmark, system: AISystem, config: RunConfig) -> ExamResult:
    results = []
    for sample in benchmark.data:
        output = await system(sample.input)
        score = benchmark.scoring.score(sample.expected, output)
        results.append(ScoredSample(sample, output, score))
    return ExamResult(results, benchmark.scoring.aggregate([r.score for r in results]))
```

## What Gets Thrown Away

From the old benchy, we keep only what serves the bare metal:

**Keep:**
- `run_outcome.json` contract (status, counts, exit codes)
- `GenericAPIInterface` pattern (HTTP endpoint benchmarking)
- `benchmark.yaml` user-facing spec (but simplified)
- Data generation concept (but simplified)
- Smoke → full workflow

**Throw away:**
- Handler class hierarchy (MultipleChoice, Structured, Freeform, MultimodalStructured)
- TaskGroupRunner complexity
- Provider config YAML sprawl
- Interface capability flags
- The entire `src/tasks/` registry pattern
- The `configs/` directory structure
- The `metadata.yaml` task group declarations

## The 5 Worktrees

1. **core** — The bare-metal engine: `AISystem`, `Benchmark`, `TaskSpec`, `ScoringFunction`, `Dataset`, `run_exam`
2. **adapters** — System adapters: HTTP, OpenAI, Anthropic, local (transformers), workflow, agent
3. **scoring** — Scoring functions: exact match, per-field, semantic, fuzzy, custom
4. **data** — Dataset loading, validation, generation, .data/ auto-discovery
5. **cli** — User interface: `benchy eval`, `benchy create`, `benchy validate`, `benchy benchmarks`

## The Bare-Metal Golem

A guardian process that:
1. Runs after every push cycle
2. Checks that we are still at the bare metal (no unnecessary abstractions)
3. Breaks things if they are too abstract
4. Verifies that the core primitive (`AISystem` as callable) is still the center
5. Ensures every line of code serves the vision: "evaluate any AI system as a program"

The golem asks: "Can I delete this file and still run a benchmark?" If yes, the file is not bare metal.

## Iteration Protocol

Each iteration:
1. Push to bare metal in one worktree
2. Golem checks: did we break something? If nothing broke, we didn't push hard enough
3. Fix what broke
4. Integrate across worktrees
5. Repeat until the engine is so simple it cannot be simpler
