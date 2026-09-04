"""The ``hf:`` source — HuggingFace Hub datasets.

Split into its own module so the one network/disk-touching call
(:func:`load_raw_dataset`) is a thin, easily-monkeypatched seam: hermetic
tests replace it with a fake dataset-like object (anything supporting
``__iter__`` and, ideally, ``__len__``) and get full coverage of the
mapping/caching logic without ever importing ``datasets`` for real.

``datasets`` (and its transitive numpy/pyarrow stack) is imported lazily,
inside :func:`load_raw_dataset`, so ``import benchy.data`` stays instant.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from benchy.data.cache import cache_root, compute_key
from benchy.data.sources import SourceResult, register_source

logger = logging.getLogger("benchy.data")


def load_raw_dataset(
    name: str,
    *,
    subset: str | None,
    split: str,
    revision: str | None,
    cache_dir: Path,
) -> Any:
    """Load a HF dataset, decode-free on any Audio column.

    Returns whatever ``datasets.load_dataset`` returns (a ``Dataset``);
    tests monkeypatch this function itself, so its return type only needs
    to support ``__iter__`` (and ``__len__`` for a length hint).
    """
    from datasets import load_dataset

    ds = load_dataset(
        name,
        subset,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir),
    )

    # Decode-free: HF's default Audio decoding needs torchcodec, which we
    # don't depend on. We only need the raw encoded bytes — the mapping
    # layer turns those straight into an AudioPart.
    features = getattr(ds, "features", None)
    if features is not None and hasattr(ds, "cast_column"):
        from datasets import Audio

        for column, feature in list(features.items()):
            if isinstance(feature, Audio):
                ds = ds.cast_column(column, Audio(decode=False))

    return ds


def hf_cache_key(spec: str, *, subset: str | None = None, split: str = "test", revision: str | None = None, **_: Any) -> str:
    """The resolved spec, hashed. No I/O — safe to call before touching the network."""
    return compute_key(scheme="hf", name=spec, subset=subset, split=split, revision=revision)


def load_hf_source(
    spec: str,
    *,
    subset: str | None = None,
    split: str = "test",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    **_: Any,
) -> SourceResult:
    """Build a :class:`SourceResult` for ``hf:<spec>``.

    Eagerly resolves the dataset (this is the "download" step — it's what
    makes the *next* run offline) but only that: row content is still
    pulled lazily, one row at a time, as :meth:`SourceResult.rows` is
    iterated.
    """
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else (cache_root() / "hf_raw")
    key = hf_cache_key(spec, subset=subset, split=split, revision=revision)

    loaded: dict[str, Any] = {}

    def get_dataset() -> Any:
        if "ds" not in loaded:
            loaded["ds"] = load_raw_dataset(
                spec, subset=subset, split=split, revision=revision, cache_dir=resolved_cache_dir
            )
        return loaded["ds"]

    ds = get_dataset()
    len_hint: int | None
    try:
        len_hint = len(ds)
    except TypeError:
        len_hint = None

    def rows() -> Iterator[Mapping[str, Any]]:
        return iter(get_dataset())

    def reload(new_split: str) -> SourceResult:
        return load_hf_source(spec, subset=subset, split=new_split, revision=revision, cache_dir=cache_dir)

    return SourceResult(
        rows=rows,
        len_hint=len_hint,
        base_dir=None,
        cache_key=key,
        cacheable=True,
        split_reloader=reload,
    )


register_source("hf", load_hf_source, key_fn=hf_cache_key)
