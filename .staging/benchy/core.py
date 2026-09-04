"""benchy.core — the frozen spine.

Every other benchy module is written against the contracts in this file and
against nothing else. This file has no dependencies beyond the stdlib, imports
no sibling benchy module, and is the one place where the shape of a benchmark
is decided.

The vision (VISION.md) says two things that this file encodes literally:

1. **The primitive is the AI-system, not the model.** So `System` is an opaque
   protocol with exactly one interesting method, `invoke`, and a
   `Capabilities` record. A raw model, a model+prompt node, a composed
   workflow, and a tool-using agent are all the same thing here.

2. **A benchmark is the first step of AI development, and must export as a
   loss function.** So a `Benchmark` is `Task + Data + Scoring` — the *exam* —
   and the system is the *argument*: `benchmark.run(system)`. The free
   variable of a loss function is the thing being optimized, so
   `as_loss() -> (System) -> float` falls out of the shape rather than being
   bolted on.

The four peer modules from the vision map onto four protocols here:

    Task      what shape does a solution have, and how is a sample turned
              into a request / a response turned into a typed prediction
    Scorer    what does "good" mean, as a composable symbolic program
    System    the AI-program under test, opaque
    Data      the evidence, as a stream of Samples

Nothing else is a first-class concept. Transports, adapters, providers,
runners, pipelines and checkpoints are plumbing that lives below these lines.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    # ontology
    "OntologyPath",
    # content
    "TextPart", "ImagePart", "AudioPart", "Part", "Message", "Role",
    # io
    "Sample", "Request", "Response", "Usage", "Prediction",
    # capability
    "Capabilities", "SystemKind",
    # protocols
    "Task", "Scorer", "System", "Data",
    # results
    "Score", "Record", "Report",
    # types
    "LossFn", "SystemLoader",
    # errors
    "BenchyError", "LoadError", "SchemaViolation", "ParseFailure",
    "SystemFailure", "CapabilityError",
]


# --------------------------------------------------------------------------
# Ontology:  /<task?>/<domain?>/<language?>
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OntologyPath:
    """The whole benchy ontology for AI: /<task>/<domain>/<language>.

    An AI-system is a program that performs a task, so `task` is the root
    level. `domain` and `language` narrow it. Any level may be absent:
    "transcription", "transcription/fleurs" and "transcription/fleurs/pt-BR"
    are all valid, and a missing level means "unspecified", not "empty".

    This is simultaneously the registry key and the on-disk layout:
    `benchmarks/<task>/<domain>/<language>/benchmark.yaml`.
    """

    task: str
    domain: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.task or "/" in self.task:
            raise ValueError(f"task segment must be a non-empty slug, got {self.task!r}")
        for name in ("domain", "language"):
            seg = getattr(self, name)
            if seg is not None and (not seg or "/" in seg):
                raise ValueError(f"{name} segment must be a non-empty slug or None, got {seg!r}")
        if self.domain is None and self.language is not None:
            raise ValueError("cannot specify a language without a domain; use domain='_' to skip")

    @classmethod
    def parse(cls, raw: str) -> OntologyPath:
        parts = [p for p in raw.strip().strip("/").split("/") if p]
        if not parts or len(parts) > 3:
            raise ValueError(f"ontology path must have 1..3 segments, got {raw!r}")
        return cls(*parts)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return "/".join(p for p in (self.task, self.domain, self.language) if p)

    @property
    def segments(self) -> tuple[str, ...]:
        return tuple(p for p in (self.task, self.domain, self.language) if p)

    def is_prefix_of(self, other: OntologyPath) -> bool:
        """`transcription` is a prefix of `transcription/fleurs/pt-BR`."""
        mine, theirs = self.segments, other.segments
        return len(mine) <= len(theirs) and theirs[: len(mine)] == mine


# --------------------------------------------------------------------------
# Content parts — the universal, modality-agnostic request payload
# --------------------------------------------------------------------------

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """Exactly one of data / path / url is set."""

    data: bytes | None = None
    path: str | None = None
    url: str | None = None
    mime: str = "image/png"


@dataclass(frozen=True, slots=True)
class AudioPart:
    """Exactly one of data / path / url is set."""

    data: bytes | None = None
    path: str | None = None
    url: str | None = None
    mime: str = "audio/wav"
    sample_rate: int | None = None


Part = TextPart | ImagePart | AudioPart


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    parts: tuple[Part, ...]

    @classmethod
    def text(cls, role: Role, text: str) -> Message:
        return cls(role=role, parts=(TextPart(text),))


# --------------------------------------------------------------------------
# Sample / Request / Response / Prediction
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Sample:
    """One row of evidence.

    `input` satisfies the Task's input schema. `expected` is the ground truth
    the Scorer grades against — its shape satisfies the Task's output schema.
    `meta` carries anything the benchmark author wants to keep around (source
    file, difficulty band, licence) without polluting the graded contract.
    """

    id: str
    input: Mapping[str, Any]
    expected: Any = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Request:
    """What an AI-system is asked to do. Deliberately transport-free.

    A `Request` says nothing about OpenAI, HTTP, transformers or vLLM. Each
    System implementation lowers it into whatever its transport wants. That
    lowering is the plumbing the vision wants hidden.
    """

    messages: tuple[Message, ...]
    output_schema: Mapping[str, Any] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Response:
    """Raw output from an AI-system, before the Task interprets it.

    `data` is populated only when the system natively returned structured
    output. Otherwise the Task parses `text`.
    """

    text: str | None = None
    data: Any = None
    raw: Any = None
    usage: Usage | None = None
    latency_ms: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class Prediction:
    """A Response interpreted against the Task's output schema."""

    value: Any
    raw_text: str | None = None
    parse_ok: bool = True
    parse_error: str | None = None


# --------------------------------------------------------------------------
# Capabilities — how benchy hides AI-system configuration
# --------------------------------------------------------------------------

SystemKind = Literal["model", "node", "workflow", "agent"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a System can be handed and what it can hand back.

    The Task consults this when rendering a Request: a system with
    `structured_output=True` gets a schema-constrained request; one without
    gets the schema encoded in the prompt and its text repaired on the way
    back. That negotiation is why authoring stays clean.
    """

    kind: SystemKind = "model"
    text_in: bool = True
    image_in: bool = False
    audio_in: bool = False
    video_in: bool = False
    structured_output: bool = False
    tools: bool = False
    streaming: bool = False
    batch: bool = False
    max_concurrency: int = 4
    context_tokens: int | None = None

    def accepts(self, part: Part) -> bool:
        match part:
            case TextPart():
                return self.text_in
            case ImagePart():
                return self.image_in
            case AudioPart():
                return self.audio_in
        return False


# --------------------------------------------------------------------------
# The four peer protocols
# --------------------------------------------------------------------------

@runtime_checkable
class System(Protocol):
    """An AI-program under test: model | node | workflow | agent.

    One method matters. Everything a System implementation does beyond
    `invoke` — loading weights, picking a framework for an architecture,
    holding an HTTP session, spawning a subprocess, running a while-loop with
    tools — is invisible from here. That opacity is the point: benchy grades
    AI-systems, not models.
    """

    url: str
    capabilities: Capabilities

    async def invoke(self, request: Request) -> Response: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class Task(Protocol):
    """The exam contract: what shape does a solution have.

    A Task owns the input/output schema and the two-way bridge between a
    Sample and an opaque System. It owns no data, no scoring, no transport.
    """

    name: str
    ontology: OntologyPath
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]

    def render(self, sample: Sample, caps: Capabilities) -> Request: ...

    def parse(self, response: Response, caps: Capabilities) -> Prediction: ...

    def validate_sample(self, sample: Sample) -> None: ...


@runtime_checkable
class Scorer(Protocol):
    """The rubric, as a composable symbolic program.

    Every scorer is a node in an expression tree. Reading its `repr` tells you
    what the benchmark *means*, and re-evaluating that repr reconstructs it.
    `fitness` is the single scalar an optimizer consumes.
    """

    name: str

    def evaluate(self, prediction: Any, expected: Any, sample: Sample) -> Score: ...

    def fitness(self, prediction: Any, expected: Any, sample: Sample) -> float: ...

    def aggregate(self, scores: Sequence[Score]) -> Mapping[str, Any]: ...

    def __repr__(self) -> str: ...


@runtime_checkable
class Data(Protocol):
    """The evidence: a stream of Samples that satisfy the Task's schema."""

    def __iter__(self) -> Iterator[Sample]: ...

    def __len__(self) -> int: ...

    def take(self, n: int) -> Data: ...

    def split(self, name: str) -> Data: ...


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Score:
    """One sample's grade. `value` is the scalar; `breakdown` is the why."""

    value: float
    breakdown: Mapping[str, Any] = field(default_factory=dict)
    scorer: str = ""


@dataclass(frozen=True, slots=True)
class Record:
    """Everything that happened for one sample, for auditability."""

    sample_id: str
    prediction: Prediction | None
    score: Score | None
    latency_ms: float | None = None
    usage: Usage | None = None
    error: str | None = None
    raw_text: str | None = None


@dataclass(frozen=True, slots=True)
class Report:
    """The result of grading one System against one Benchmark."""

    benchmark: str
    system: str
    scorer: str
    fitness: float
    aggregate: Mapping[str, Any]
    records: tuple[Record, ...] = ()
    n_samples: int = 0
    n_errors: int = 0
    wall_time_s: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Function types
# --------------------------------------------------------------------------

#: What `Benchmark.as_loss()` returns. The system is the free variable — this
#: is the entire reason the vision calls a benchmark "a new loss function".
LossFn = Callable[[System], Awaitable[float]]

#: How `benchy.system` registers a URL scheme: ("openai", loader).
SystemLoader = Callable[..., System]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class BenchyError(Exception):
    """Base for every error benchy raises on purpose."""


class LoadError(BenchyError):
    """A System / Task / Data URL could not be resolved or loaded."""


class SchemaViolation(BenchyError):
    """A Sample or a Prediction does not satisfy the Task's schema."""


class ParseFailure(BenchyError):
    """A Response could not be interpreted against the output schema."""


class SystemFailure(BenchyError):
    """The AI-system errored while producing a Response."""


class CapabilityError(BenchyError):
    """A Task needs a capability the System does not have (e.g. audio in)."""
