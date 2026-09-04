"""benchy.benchmark — the exam: Task + Data + Scoring, graded against a System.

    bench   = Benchmark(task=task, data=data, scoring=scoring, ontology="...")
    report  = await bench.run(system, limit=200)     # grade one candidate
    loss    = bench.as_loss()                        # (System) -> Awaitable[float]

The AI-system is the argument, not a constructor field -- see VISION.md and
the module docstring in `benchy/loss.py`. `Benchmark` itself only knows about
`benchy.core` protocols (`Task`, `Data`, `Scorer`, `System`); every reference
to a sibling module (`benchy.task`, `benchy.data`, `benchy.scoring`) is a
lazy import confined to `from_yaml`'s YAML-spec resolvers, so `import
benchy.benchmark` succeeds even before those modules exist.

Design decisions worth knowing about (see also the docstring on `run`):

* **Sample validation happens in one upfront, synchronous pass** over the
  (possibly `limit`-ed) sample list before any request is issued. A
  `SchemaViolation` there is a benchmark-authoring defect, not a runtime
  hiccup, so it aborts the whole run loudly instead of being folded into
  `Record.error`. Everything after that point -- render, invoke (with retry
  and a timeout), parse, score -- is caught per-sample; one bad sample can
  never abort the run.
* **Errored samples are excluded from the score list handed to
  `scoring.aggregate()`**, not scored as 0.0. `Score.breakdown` is an opaque,
  scorer-specific shape; synthesizing a `Score` for a failure risks feeding a
  third-party `aggregate()` implementation a shape it never produced itself.
  Errors stay fully visible via `Report.n_errors` / `Report.n_samples` and
  each `Record.error`, so `fitness` answers "how good are the answers we got
  back" while "how reliable is this system" is a separate, always-visible
  number. A caller who wants unreliability to count against fitness can
  combine them explicitly, e.g. `fitness * (1 - n_errors / n_samples)`.
* **Concurrency, retry and timeout** are per-sample: at most `concurrency`
  samples are ever mid-flight (default `system.capabilities.max_concurrency`),
  a failed `system.invoke` (raised exception *or* a returned `Response` with
  `.error` set) is retried up to `retries` times with exponential backoff +
  jitter, and `timeout` (if given) bounds a single attempt. The engine has no
  way to distinguish "transient" from "permanent" failure from the frozen
  `Response.error: str | None` shape alone, so it retries either kind
  uniformly and gives up gracefully after the budget is spent.
* **`run()` never mutates `self` or consumes `self.data`.** It is safe to
  call many times on one `Benchmark` -- which is exactly what an optimizer
  driving `as_loss()` does.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchy.core import (
    Capabilities,
    Data as DataProto,
    LoadError,
    LossFn,
    Record,
    Report,
    Sample,
    Scorer as ScorerProto,
    SystemFailure,
    Task as TaskProto,
)
from benchy.loss import as_loss as _as_loss
from benchy.loss import as_metric as _as_metric
from benchy.report import record_from_json, record_to_json

__all__ = ["Benchmark", "load_benchmark", "RetryPolicy"]


# --------------------------------------------------------------------------
# Retry policy -- slimmed from src/engine/retry.py
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with jitter, capped. `delay(attempt)` is 1-based."""

    max_attempts: int = 3
    base_delay: float = 0.05
    multiplier: float = 2.0
    max_delay: float = 1.0
    jitter: float = 0.05

    def delay(self, attempt: int) -> float:
        raw = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        return raw + random.uniform(0.0, self.jitter)


async def _invoke_with_retry(system, request, *, policy: RetryPolicy, timeout: float | None):
    """Call `system.invoke`, retrying on exception or `Response.error`.

    Returns the last `Response` on give-up-via-error, or re-raises the last
    exception on give-up-via-exception. Either way the caller (`_process_one`)
    turns the outcome into a `Record`.
    """
    last_exc: BaseException | None = None
    last_response = None
    for attempt in range(1, max(1, policy.max_attempts) + 1):
        try:
            if timeout is not None:
                response = await asyncio.wait_for(system.invoke(request), timeout)
            else:
                response = await system.invoke(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
            last_exc = exc
            if attempt >= policy.max_attempts:
                raise
            await asyncio.sleep(policy.delay(attempt))
            continue

        if response.ok:
            return response
        last_response = response
        if attempt >= policy.max_attempts:
            return response
        await asyncio.sleep(policy.delay(attempt))

    if last_exc is not None:  # pragma: no cover - defensive, loop always returns/raises above
        raise last_exc
    return last_response


# --------------------------------------------------------------------------
# Checkpoint -- slimmed from src/engine/checkpoint.py
# --------------------------------------------------------------------------


class _Checkpoint:
    """Per-sample checkpoint file so a long run can resume.

    One JSON object per line, keyed by `sample_id`. Deliberately drops the
    original's config-hash validation (~94 lines -> ~20): the caller passing
    `checkpoint=path` is trusted to reuse it with the same benchmark + data.
    That trades a footgun (stale checkpoint silently reused after an
    unrelated edit) for the simplicity that matches this module's size
    budget; a benchmark author who cares can just use a fresh path per
    benchmark version.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Record]:
        if not self.path.exists():
            return {}
        done: dict[str, Record] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            done[d["sample_id"]] = record_from_json(d)
        return done

    def append(self, record: Record) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


class Benchmark:
    """The exam: `Task + Data + Scoring`. The AI-system is the argument.

    See the module docstring for the run-loop and error-handling contract.
    """

    def __init__(
        self,
        task,
        data,
        scoring,
        *,
        name: str | None = None,
        ontology: str | None = None,
        baselines: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(task, TaskProto):
            raise TypeError(f"task must implement the benchy.core.Task protocol, got {type(task)!r}")
        if not isinstance(data, DataProto):
            raise TypeError(f"data must implement the benchy.core.Data protocol, got {type(data)!r}")
        if not isinstance(scoring, ScorerProto):
            raise TypeError(f"scoring must implement the benchy.core.Scorer protocol, got {type(scoring)!r}")

        self.task = task
        self.data = data
        self.scoring = scoring
        self.name = name
        self.ontology = ontology
        self.baselines: tuple[str, ...] = tuple(baselines)
        self.meta: dict[str, Any] = dict(meta or {})

        # Set by `from_yaml` so `to_yaml` can re-emit the exact task/data/
        # scoring specs it was loaded from, rather than reverse-engineering
        # them from live objects. See `_describe_fallback`.
        self._yaml_spec: dict[str, Any] | None = None
        self._source_path: Path | None = None

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"Benchmark(name={self.label!r}, ontology={self.ontology!r})"

    @property
    def label(self) -> str:
        """The human-facing identity used as `Report.benchmark`."""
        if self.name:
            return self.name
        if self.ontology:
            return str(self.ontology)
        return getattr(self.task, "name", None) or "benchmark"

    # ------------------------------------------------------------------
    # The run loop
    # ------------------------------------------------------------------

    async def run(
        self,
        system,
        *,
        limit: int | None = None,
        concurrency: int | None = None,
        progress: bool = False,
        on_record: Callable[[Record], Any] | None = None,
        timeout: float | None = None,
        retries: int = 3,
        checkpoint: str | Path | None = None,
    ) -> Report:
        data = self.data.take(limit) if limit is not None else self.data
        samples = list(data)

        # Step 1, upfront and synchronous: fail loud on a schema violation.
        for sample in samples:
            self.task.validate_sample(sample)

        n = len(samples)
        concurrency = concurrency or getattr(system.capabilities, "max_concurrency", None) or 4
        policy = RetryPolicy(max_attempts=max(1, retries))
        sem = asyncio.Semaphore(concurrency)

        ckpt = _Checkpoint(checkpoint) if checkpoint is not None else None
        prior = ckpt.load() if ckpt is not None else {}

        records: list[Record | None] = [None] * n
        start = time.monotonic()

        async def worker(index: int, sample: Sample) -> None:
            cached = prior.get(sample.id)
            if cached is not None:
                record = cached
            else:
                async with sem:
                    record = await self._process_one(system, sample, policy=policy, timeout=timeout)
                if ckpt is not None:
                    ckpt.append(record)
            records[index] = record
            if on_record is not None:
                maybe_awaitable = on_record(record)
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            if progress:
                done = sum(1 for r in records if r is not None)
                print(f"[{done}/{n}] {sample.id}", file=sys.stderr)

        if samples:
            await asyncio.gather(*(worker(i, s) for i, s in enumerate(samples)))

        finished: tuple[Record, ...] = tuple(records)  # type: ignore[arg-type]
        n_errors = sum(1 for r in finished if r.error is not None)
        scored = [r.score for r in finished if r.error is None and r.score is not None]
        aggregate = dict(self.scoring.aggregate(scored))
        fitness = float(aggregate.get("fitness", 0.0))
        wall_time_s = time.monotonic() - start

        if ckpt is not None:
            ckpt.clear()

        return Report(
            benchmark=self.label,
            system=getattr(system, "url", repr(system)),
            scorer=repr(self.scoring),
            fitness=fitness,
            aggregate=aggregate,
            records=finished,
            n_samples=n,
            n_errors=n_errors,
            wall_time_s=wall_time_s,
            meta=dict(self.meta),
        )

    async def _process_one(self, system, sample: Sample, *, policy: RetryPolicy, timeout: float | None) -> Record:
        t0 = time.monotonic()
        try:
            request = self.task.render(sample, system.capabilities)
            response = await _invoke_with_retry(system, request, policy=policy, timeout=timeout)
            if not response.ok:
                raise SystemFailure(response.error or "system returned an error response")
            prediction = self.task.parse(response, system.capabilities)
            score = self.scoring.evaluate(prediction.value, sample.expected, sample)
            latency_ms = response.latency_ms if response.latency_ms is not None else (time.monotonic() - t0) * 1000
            return Record(
                sample_id=sample.id,
                prediction=prediction,
                score=score,
                latency_ms=latency_ms,
                usage=response.usage,
                error=None,
                raw_text=prediction.raw_text,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad sample must never abort the run
            latency_ms = (time.monotonic() - t0) * 1000
            return Record(
                sample_id=sample.id,
                prediction=None,
                score=None,
                latency_ms=latency_ms,
                usage=None,
                error=f"{type(exc).__name__}: {exc}",
                raw_text=None,
            )

    def run_sync(self, system, **kw: Any) -> Report:
        """Convenience for scripts/CLI: `asyncio.run(self.run(system, **kw))`."""
        return asyncio.run(self.run(system, **kw))

    async def compare(self, systems: Sequence, **kw: Any) -> list[Report]:
        """Grade several systems against this one exam. Order-preserving.

        Runs sequentially (one system's `run()` fully completes before the
        next starts) so each system gets the full `concurrency` budget and
        results are easy to reason about; a caller who wants systems graded
        in parallel can `asyncio.gather(*(bench.run(s, **kw) for s in systems))`
        directly -- `run()` is safe to call concurrently on the same instance.
        """
        return [await self.run(system, **kw) for system in systems]

    # ------------------------------------------------------------------
    # Loss / metric export
    # ------------------------------------------------------------------

    def as_loss(self) -> LossFn:
        return _as_loss(self)

    def as_metric(self) -> Callable[[Any, Any, Any], float]:
        return _as_metric(self)

    # ------------------------------------------------------------------
    # YAML authoring surface
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Benchmark:
        import yaml

        path = Path(path)
        if path.is_dir():
            path = path / "benchmark.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise LoadError(f"{path}: benchmark.yaml must be a mapping, got {type(raw).__name__}")
        base_dir = path.parent

        for required in ("task", "data", "scoring"):
            if required not in raw:
                raise LoadError(f"{path}: missing required '{required}' field")

        task = _load_task(raw["task"], base_dir)
        data = _load_data(raw["data"], base_dir)
        scoring = _load_scoring(raw["scoring"], base_dir)

        bench = cls(
            task=task,
            data=data,
            scoring=scoring,
            name=raw.get("name"),
            ontology=raw.get("ontology"),
            baselines=tuple(raw.get("baselines", ())),
            meta=raw.get("meta", {}),
        )
        bench._yaml_spec = raw
        bench._source_path = path
        return bench

    def to_yaml(self, path: str | Path | None = None) -> str:
        import yaml

        spec: dict[str, Any] = {}
        if self.ontology is not None:
            spec["ontology"] = str(self.ontology)
        if self.name is not None:
            spec["name"] = self.name

        if self._yaml_spec is not None:
            spec["task"] = self._yaml_spec.get("task")
            spec["data"] = self._yaml_spec.get("data")
            spec["scoring"] = self._yaml_spec.get("scoring")
        else:
            spec["task"] = _describe_task(self.task)
            spec["data"] = _describe_data(self.data)
            spec["scoring"] = repr(self.scoring)

        if self.baselines:
            spec["baselines"] = list(self.baselines)
        if self.meta:
            spec["meta"] = dict(self.meta)

        text = yaml.safe_dump(spec, sort_keys=False, default_flow_style=False, allow_unicode=True)
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        return text


def load_benchmark(ref: str | Path) -> Benchmark:
    """Load a Benchmark by on-disk path (file or directory) or by ontology.

    `ref` is tried, in order, as: an existing file/directory path; then an
    ontology path resolved to `benchmarks/<task>/<domain>/<language>/benchmark.yaml`
    relative to the current working directory.
    """
    path = Path(ref)
    if path.exists():
        return Benchmark.from_yaml(path)

    from benchy.core import OntologyPath

    try:
        onto = OntologyPath.parse(str(ref))
    except ValueError:
        onto = None
    if onto is not None:
        candidate = Path("benchmarks", *onto.segments, "benchmark.yaml")
        if candidate.exists():
            return Benchmark.from_yaml(candidate)

    raise LoadError(f"could not resolve benchmark ref {ref!r} (tried it as a path and as an ontology)")


# --------------------------------------------------------------------------
# YAML spec resolvers -- the only place this module reaches for a sibling
# --------------------------------------------------------------------------

_FILE_REF_RE = re.compile(r"^(?P<path>.+\.py):(?P<attr>\w+)$")
_SOURCE_RE = re.compile(r"^(?P<scheme>[a-zA-Z][\w+.-]*):(?P<rest>.*)$")


def _resolve_ref(ref: str, base_dir: Path) -> Any:
    """Resolve a `"path/to/file.py:Attr"` reference into a live Python object."""
    m = _FILE_REF_RE.match(ref.strip())
    if not m:
        raise LoadError(f"expected a 'path/to/file.py:Attr' reference, got {ref!r}")
    file_path = (base_dir / m.group("path")).resolve()
    if not file_path.exists():
        raise LoadError(f"referenced file not found: {file_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise LoadError(f"could not load {file_path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attr = m.group("attr")
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise LoadError(f"{file_path} has no attribute {attr!r}") from exc


def _resolve_source_path(source: str, base_dir: Path) -> str:
    """Resolve a relative path inside a `"scheme:path"` data source string."""
    m = _SOURCE_RE.match(source)
    if not m:
        return source
    scheme, rest = m.group("scheme"), m.group("rest")
    if scheme in ("http", "https", "hf", "s3"):
        return source
    if not rest.startswith("/"):
        return f"{scheme}:{(base_dir / rest).resolve()}"
    return source


def _load_task(spec: Any, base_dir: Path):
    """Resolve a `task:` YAML field into a `benchy.core.Task`.

    Lazy import: `benchy.task`. Expected API used here:
      - `benchy.task.builtin.<name>(**kwargs) -> Task` when `spec['builtin']` is set.
      - `benchy.task.load(ontology: str) -> Task` when only `spec['ontology']` is set.
      - `benchy.task.Task(name=, ontology=, input=, output=, instructions=) -> Task`
        (the generic constructor) when `spec['input']`/`spec['output']` are set directly
        -- this is also what `Benchmark.to_yaml`'s fallback serializer emits, since
        `input_schema`/`output_schema` are the only Task internals the frozen
        `benchy.core.Task` protocol guarantees are readable back off a live instance.
    """
    if not isinstance(spec, Mapping):
        raise LoadError(f"'task' must be a mapping, got {type(spec).__name__}")
    spec = dict(spec)

    try:
        import benchy.task as task_mod
    except ImportError as exc:
        raise LoadError(
            "benchmark.yaml has a 'task' field but 'benchy.task' is not installed "
            "(it lands from a sibling worktree and is wired in at merge)."
        ) from exc

    if "builtin" in spec:
        kind = spec.pop("builtin")
        factory = getattr(task_mod.builtin, kind, None)
        if factory is None:
            raise LoadError(f"unknown builtin task {kind!r}")
        return factory(**_resolve_task_kwargs(spec, base_dir))

    if "input" in spec or "output" in spec:
        return task_mod.Task(**_resolve_task_kwargs(spec, base_dir))

    if "ontology" in spec:
        return task_mod.load(spec["ontology"])

    raise LoadError(f"task spec must have 'builtin', 'ontology', or 'input'/'output': {spec!r}")


def _resolve_task_kwargs(spec: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in ("name", "ontology", "input", "output", "instructions"):
        if key not in spec:
            continue
        value = spec[key]
        if key in ("input", "output") and isinstance(value, str):
            value = _resolve_ref(value, base_dir)
        kwargs[key] = value
    return kwargs


def _load_data(spec: Any, base_dir: Path):
    """Resolve a `data:` YAML field into a `benchy.core.Data`.

    Lazy import: `benchy.data`. Expected API used here:
      - `benchy.data.load(source: str, **opts) -> Data` when `spec['source']` is set.
      - `benchy.data.Data.from_samples(list[Sample]) -> Data` when `spec['samples']`
        is set directly -- this is also what `Benchmark.to_yaml`'s fallback
        serializer emits (materializing whatever `Data.__iter__` yields, since
        that is the only thing the frozen `Data` protocol guarantees).
    """
    if not isinstance(spec, Mapping):
        raise LoadError(f"'data' must be a mapping, got {type(spec).__name__}")
    spec = dict(spec)

    try:
        import benchy.data as data_mod
    except ImportError as exc:
        raise LoadError(
            "benchmark.yaml has a 'data' field but 'benchy.data' is not installed "
            "(it lands from a sibling worktree and is wired in at merge)."
        ) from exc

    if "samples" in spec:
        rows = spec["samples"]
        samples = [
            Sample(
                id=row["id"],
                input=row.get("input", {}),
                expected=row.get("expected"),
                meta=row.get("meta", {}),
            )
            for row in rows
        ]
        return data_mod.Data.from_samples(samples)

    if "source" not in spec:
        raise LoadError(f"data spec must have 'source' or 'samples': {spec!r}")
    source = _resolve_source_path(spec.pop("source"), base_dir)
    return data_mod.load(source, **spec)


def _load_scoring(raw: Any, base_dir: Path):
    """Resolve a `scoring:` YAML field into a `benchy.core.Scorer`.

    Lazy import: `benchy.scoring`. Expected API: `benchy.scoring.parse_scorer(str) ->
    Scorer`. A `"./scoring.py:default"`-shaped string is instead resolved as a
    custom-rubric file reference, per the spec.
    """
    if not isinstance(raw, str):
        raise LoadError(f"'scoring' must be a string, got {type(raw).__name__}")
    ref = raw.strip()
    if _FILE_REF_RE.match(ref):
        return _resolve_ref(ref, base_dir)

    try:
        from benchy.scoring import parse_scorer
    except ImportError as exc:
        raise LoadError(
            "benchmark.yaml has a 'scoring' field but 'benchy.scoring' is not installed "
            "(it lands from a sibling worktree and is wired in at merge)."
        ) from exc
    return parse_scorer(ref)


# --------------------------------------------------------------------------
# to_yaml's fallback serializers, for a Benchmark not built via from_yaml
# --------------------------------------------------------------------------


def _describe_task(task) -> dict[str, Any]:
    d: dict[str, Any] = {}
    name = getattr(task, "name", None)
    if name:
        d["name"] = name
    ontology = getattr(task, "ontology", None)
    if ontology is not None:
        d["ontology"] = str(ontology)
    input_schema = getattr(task, "input_schema", None)
    if input_schema:
        d["input"] = dict(input_schema)
    output_schema = getattr(task, "output_schema", None)
    if output_schema:
        d["output"] = dict(output_schema)
    return d


def _describe_data(data) -> dict[str, Any]:
    samples = [
        {"id": s.id, "input": dict(s.input), "expected": s.expected, "meta": dict(s.meta)} for s in data
    ]
    return {"samples": samples}
