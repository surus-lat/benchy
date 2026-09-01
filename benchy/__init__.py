"""benchy — create benchmarks for AI-systems.

    import benchy as b

    task    = b.Task.load("image_extraction/invoices/es-AR")
    data    = b.Data.load("./data/invoices/*.json", schema=task.input_schema)
    scoring = b.field_wise_weighted(fields=FIELDS, weights={"total": 3})

    bench   = b.Benchmark(task=task, data=data, scoring=scoring)

    report  = await bench.run(b.System.load("openai:gpt-5-mini"))
    loss    = bench.as_loss()          # (System) -> float, for an optimizer

The benchmark is the exam: Task + Data + Scoring. The AI-system is the
candidate being graded, and therefore the argument — which is exactly what
makes a benchy benchmark usable as a loss function.
"""

from __future__ import annotations

from benchy.core import (
    AudioPart,
    BenchyError,
    Capabilities,
    CapabilityError,
    Data,
    ImagePart,
    LoadError,
    LossFn,
    Message,
    OntologyPath,
    ParseFailure,
    Part,
    Prediction,
    Record,
    Report,
    Request,
    Response,
    Sample,
    SchemaViolation,
    Score,
    Scorer,
    System,
    SystemFailure,
    SystemKind,
    Task,
    TextPart,
    Usage,
)

__version__ = "0.2.0.dev0"

# Names resolved lazily so that `import benchy` works while the four peer
# modules are still landing in parallel worktrees.
_LAZY: dict[str, str] = {
    "Benchmark": "benchy.benchmark",
    "load_benchmark": "benchy.benchmark",
}


def __getattr__(name: str):  # pragma: no cover - trivial dispatch
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'benchy' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


__all__ = [
    "AudioPart", "BenchyError", "Capabilities", "CapabilityError", "Data",
    "ImagePart", "LoadError", "LossFn", "Message", "OntologyPath",
    "ParseFailure", "Part", "Prediction", "Record", "Report", "Request",
    "Response", "Sample", "SchemaViolation", "Score", "Scorer", "System",
    "SystemFailure", "SystemKind", "Task", "TextPart", "Usage",
    "Benchmark", "load_benchmark", "__version__",
]
