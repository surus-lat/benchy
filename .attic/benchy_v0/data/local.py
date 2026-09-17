"""Local directory dataset loader — bare metal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchy.core import Sample


def local_dataset(path: str | Path, pattern: str = "*.jsonl"):
    """Build a local directory Dataset."""
    path = Path(path)
    _samples: list[Sample] | None = None

    def _load() -> list[Sample]:
        nonlocal _samples
        if _samples is not None:
            return _samples

        samples: list[Sample] = []
        for file_path in sorted(path.glob(pattern)):
            if file_path.suffix == ".jsonl":
                with open(file_path) as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        samples.append(Sample(
                            id=row.get("id", f"{file_path.stem}_{i}"),
                            input=row.get("input", row.get("text", row.get("image_path"))),
                            expected=row.get("expected", row.get("label", row.get("output"))),
                            meta=row.get("meta", {}),
                        ))
        _samples = samples
        return samples

    class _LocalDataset:
        def __init__(self):
            self._samples = None

        def __iter__(self) -> Iterator[Sample]:
            return iter(_load())

        def __len__(self) -> int:
            return len(_load())

        def take(self, n: int):
            new = local_dataset(path, pattern)
            new._samples = _load()[:n]
            return new

        def split(self, name: str):
            return self

    return _LocalDataset()
