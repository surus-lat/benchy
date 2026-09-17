"""benchy.report — rendering and persistence for a `Report`.

A `Report` (benchy.core.Report) is the frozen result of grading one System
against one Benchmark. This module owns everything that happens to a Report
*after* it exists: printing it, turning it into JSON, saving/loading it, and
laying several of them side by side as a leaderboard.

The JSON contract (`to_json` / `from_json`) is versioned via `SCHEMA_VERSION`
so a benchmark's history stays readable as the shape evolves. It assumes
`Prediction.value` and `Score.breakdown` are JSON-serializable -- which is
the deal a Task/Scorer implementation makes when it returns them (a
structured-extraction Task should hand back a plain dict, not a live
pydantic instance, for exactly this reason).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchy.core import Prediction, Record, Report, Score, Usage

__all__ = [
    "SCHEMA_VERSION",
    "render_text",
    "render_markdown",
    "to_json",
    "from_json",
    "save",
    "load",
    "compare_table",
    "record_to_json",
    "record_from_json",
]

#: Bumped whenever the shape of `to_json` changes in a way old readers can't
#: shrug off. Consumers should treat an unknown version as "read what you can".
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Record <-> JSON (also used by Benchmark's checkpoint file)
# --------------------------------------------------------------------------


def _prediction_to_json(p: Prediction | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {
        "value": p.value,
        "raw_text": p.raw_text,
        "parse_ok": p.parse_ok,
        "parse_error": p.parse_error,
    }


def _prediction_from_json(d: Mapping[str, Any] | None) -> Prediction | None:
    if d is None:
        return None
    return Prediction(
        value=d.get("value"),
        raw_text=d.get("raw_text"),
        parse_ok=d.get("parse_ok", True),
        parse_error=d.get("parse_error"),
    )


def _score_to_json(s: Score | None) -> dict[str, Any] | None:
    if s is None:
        return None
    return {"value": s.value, "breakdown": dict(s.breakdown), "scorer": s.scorer}


def _score_from_json(d: Mapping[str, Any] | None) -> Score | None:
    if d is None:
        return None
    return Score(value=d["value"], breakdown=d.get("breakdown", {}), scorer=d.get("scorer", ""))


def _usage_to_json(u: Usage | None) -> dict[str, Any] | None:
    if u is None:
        return None
    return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "cost_usd": u.cost_usd}


def _usage_from_json(d: Mapping[str, Any] | None) -> Usage | None:
    if d is None:
        return None
    return Usage(
        input_tokens=d.get("input_tokens"),
        output_tokens=d.get("output_tokens"),
        cost_usd=d.get("cost_usd"),
    )


def record_to_json(record: Record) -> dict[str, Any]:
    """A `Record` as a plain, JSON-safe dict. One line of a checkpoint file."""
    return {
        "sample_id": record.sample_id,
        "prediction": _prediction_to_json(record.prediction),
        "score": _score_to_json(record.score),
        "latency_ms": record.latency_ms,
        "usage": _usage_to_json(record.usage),
        "error": record.error,
        "raw_text": record.raw_text,
    }


def record_from_json(d: Mapping[str, Any]) -> Record:
    """Inverse of `record_to_json`."""
    return Record(
        sample_id=d["sample_id"],
        prediction=_prediction_from_json(d.get("prediction")),
        score=_score_from_json(d.get("score")),
        latency_ms=d.get("latency_ms"),
        usage=_usage_from_json(d.get("usage")),
        error=d.get("error"),
        raw_text=d.get("raw_text"),
    )


# --------------------------------------------------------------------------
# Report <-> JSON
# --------------------------------------------------------------------------


def to_json(report: Report) -> dict[str, Any]:
    """Serialize a `Report` to a plain, JSON-safe dict. Stable, versioned."""
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": report.benchmark,
        "system": report.system,
        "scorer": report.scorer,
        "fitness": report.fitness,
        "aggregate": dict(report.aggregate),
        "n_samples": report.n_samples,
        "n_errors": report.n_errors,
        "wall_time_s": report.wall_time_s,
        "meta": dict(report.meta),
        "records": [record_to_json(r) for r in report.records],
    }


def from_json(d: Mapping[str, Any]) -> Report:
    """Inverse of `to_json`. Unknown/missing optional keys default sanely."""
    return Report(
        benchmark=d["benchmark"],
        system=d["system"],
        scorer=d.get("scorer", ""),
        fitness=d["fitness"],
        aggregate=d.get("aggregate", {}),
        records=tuple(record_from_json(r) for r in d.get("records", ())),
        n_samples=d.get("n_samples", 0),
        n_errors=d.get("n_errors", 0),
        wall_time_s=d.get("wall_time_s", 0.0),
        meta=d.get("meta", {}),
    )


def save(report: Report, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(report), indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def load(path: str | Path) -> Report:
    return from_json(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_text(report: Report) -> str:
    """A short terminal summary -- what `benchy run` prints by default."""
    lines = [
        f"Benchmark: {report.benchmark}",
        f"System:    {report.system}",
        f"Scorer:    {report.scorer}",
        f"Fitness:   {_fmt(report.fitness)}",
        f"Samples:   {report.n_samples}  (errors: {report.n_errors})",
        f"Wall time: {report.wall_time_s:.2f}s",
    ]
    extra = {k: v for k, v in report.aggregate.items() if k != "fitness"}
    if extra:
        lines.append("Aggregate:")
        for key, value in extra.items():
            lines.append(f"  {key}: {_fmt(value)}")
    return "\n".join(lines)


def render_markdown(report: Report) -> str:
    """A markdown summary -- for PR comments / benchmark logs."""
    lines = [
        f"### {report.benchmark}",
        "",
        f"- **System**: `{report.system}`",
        f"- **Scorer**: `{report.scorer}`",
        f"- **Fitness**: **{_fmt(report.fitness)}**",
        f"- **Samples**: {report.n_samples} (errors: {report.n_errors})",
        f"- **Wall time**: {report.wall_time_s:.2f}s",
    ]
    extra = {k: v for k, v in report.aggregate.items() if k != "fitness"}
    if extra:
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key, value in extra.items():
            lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def compare_table(reports: Sequence[Report]) -> str:
    """A markdown leaderboard across systems, ranked by fitness (best first)."""
    headers = ["system", "fitness", "n_samples", "n_errors", "wall_time_s"]
    ranked = sorted(reports, key=lambda r: r.fitness, reverse=True)
    rows = [
        [r.system, _fmt(r.fitness), str(r.n_samples), str(r.n_errors), f"{r.wall_time_s:.2f}"]
        for r in ranked
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
