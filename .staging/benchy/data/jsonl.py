"""JSONL dataset loader — bare metal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchy.core import Sample


def jsonl_dataset(path: str | Path):
    """Build a JSONL Dataset."""
    path = Path(path)
    _samples: list[Sample] | None = None

    def _load() -> list[Sample]:
        nonlocal _samples
        if _samples is not None:
            return _samples

        samples: list[Sample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(Sample(
                    id=row.get("id", str(i)),
                    input=row.get("input", row.get("text", row.get("image_path"))),
                    expected=row.get("expected", row.get("label", row.get("output"))),
                    meta=row.get("meta", {}),
                ))
        _samples = samples
        return samples

    class _JSONLDataset:
        def __init__(self, samples=None):
            self._samples = samples

        def __iter__(self) -> Iterator[Sample]:
            if self._samples is not None:
                return iter(self._samples)
            return iter(_load())

        def __len__(self) -> int:
            if self._samples is not None:
                return len(self._samples)
            return len(_load())

        def take(self, n: int):
            return _JSONLDataset(samples=_load()[:n])

        def split(self, name: str):
            return self

    return _JSONLDataset()
