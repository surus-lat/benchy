"""``load()`` — the one function every benchmark author calls.

Ties together scheme parsing (:mod:`benchy.data.sources`), field mapping
(:mod:`benchy.data.mapping`) and on-disk caching (:mod:`benchy.data.cache`)
into the single required entry point:

    data = load("hf:google/fleurs", subset="pt_br", split="test", limit=200,
                 input={"audio": "audio"}, expected="transcription",
                 id="id", meta=["gender", "num_samples"])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchy.data.cache import is_cached, read_cached, wrap_with_cache_write
from benchy.data.dataset import Data
from benchy.data.mapping import FieldMapping
from benchy.data.sources import SourceResult, cache_key_fn_for, parse_spec, sources


def load(
    spec: str,
    *,
    subset: str | None = None,
    split: str = "test",
    revision: str | None = None,
    input: Mapping[str, str] | None = None,
    expected: str | None = None,
    id: str | None = None,
    meta: Sequence[str] | Mapping[str, str] | None = None,
    base_dir: str | Path | None = None,
    limit: int | None = None,
    cache: bool = True,
    cache_dir: str | Path | None = None,
    **source_kwargs: Any,
) -> Data:
    """Load a :class:`Data` from ``spec``.

    ``spec`` is ``"scheme:rest"`` (``hf:google/fleurs``, ``jsonl:./x.jsonl``,
    ``csv:./x.csv``, ``glob:./x/*.json``) or a bare path, whose source is
    inferred from its extension (``.jsonl``/``.ndjson``/``.json`` -> jsonl,
    ``.csv`` -> csv, ``.tsv`` -> tsv; a pattern containing ``*``/``?``/``[``
    with no scheme is treated as ``glob:``). An unregistered scheme, or an
    extension nothing recognizes, raises :class:`benchy.core.LoadError`
    naming what *is* available.

    Field mapping (all optional, sensible boring defaults when omitted):

    - ``id``: source column for ``Sample.id``. Default: a column literally
      named ``"id"`` if present, else a content hash of the row (falling
      back to the row index if that content can't be hashed) — always
      deterministic, never a random uuid.
    - ``expected``: source column for ``Sample.expected``. Default: a
      column literally named ``"expected"`` if present, else ``None``.
    - ``input``: ``{dest_key: source_column}`` for ``Sample.input``.
      Default: every column not claimed by ``id``/``expected``/``meta``,
      under its own name (identity mapping) — "leftover columns become
      input".
    - ``meta``: a list of column names (kept under their own name) or a
      ``{dest_key: source_column}`` mapping, for ``Sample.meta``.

    A mapping that names a column absent from the source raises
    ``LoadError`` listing the columns that *are* there.

    Any raw value shaped like audio (an HF ``Audio``-feature dict, a
    ``.wav``/``.mp3``/``.flac``/... path string, raw bytes) or an image (a
    PIL Image, a ``.png``/``.jpg``/... path string) is converted into a
    ``benchy.core.AudioPart`` / ``ImagePart``; everything else passes
    through as its native JSON value. Relative path strings are resolved
    against the source file's own directory (``jsonl:``/``csv:``: the file
    itself; ``glob:``: the pattern's fixed leading directory) unless
    ``base_dir`` is given explicitly, which always wins.

    ``limit`` truncates to the first N samples (equivalent to
    ``.take(limit)``, applied after mapping and after any cache lookup —
    the cache always holds the *full* resolved split).

    ``cache`` (default ``True``) enables the on-disk materialized cache for
    sources that support it (currently ``hf:``); see :mod:`benchy.data.cache`
    for the layout and the ``BENCHY_CACHE_DIR`` override. Set ``cache=False``
    to force a fresh fetch.
    """
    scheme, rest = parse_spec(spec)

    key_fn = cache_key_fn_for(scheme)
    cache_key: str | None = None
    if cache and key_fn is not None:
        cache_key = key_fn(rest, subset=subset, split=split, revision=revision, **source_kwargs)
        if cache_key is not None and is_cached(cache_key):
            data = read_cached(cache_key)
            return data.take(limit) if limit is not None else data

    loader_kwargs: dict[str, Any] = dict(source_kwargs)
    if scheme == "hf":
        loader_kwargs.update(subset=subset, split=split, revision=revision, cache_dir=cache_dir)
    result: SourceResult = sources[scheme](rest, **loader_kwargs)

    field_mapping = dict(input=input, expected=expected, id=id, meta=meta)
    data = _from_source_result(result, cache=cache, cache_key=cache_key, base_dir=base_dir, **field_mapping)

    return data.take(limit) if limit is not None else data


def _from_source_result(
    result: SourceResult,
    *,
    cache: bool,
    cache_key: str | None,
    base_dir: str | Path | None,
    input: Mapping[str, str] | None,
    expected: str | None,
    id: str | None,
    meta: Sequence[str] | Mapping[str, str] | None,
) -> Data:
    resolved_base_dir = Path(base_dir).resolve() if base_dir is not None else result.base_dir
    mapping = FieldMapping(input=input, expected=expected, id=id, meta=meta, base_dir=resolved_base_dir)

    def factory():
        return _map_rows(result.rows(), mapping)

    reload = None
    if result.split_reloader is not None:

        def reload(new_split: str) -> Data:  # noqa: F811 - intentional shadow
            new_result = result.split_reloader(new_split)
            new_key = new_result.cache_key if cache else None
            if new_key is not None and is_cached(new_key):
                return read_cached(new_key)
            new_data = _from_source_result(
                new_result,
                cache=cache,
                cache_key=new_key,
                base_dir=base_dir,
                input=input,
                expected=expected,
                id=id,
                meta=meta,
            )
            if cache and new_result.cacheable and new_key is not None:
                new_data = wrap_with_cache_write(new_data, new_key)
            return new_data

    data = Data(factory, len_hint=result.len_hint, cache_key=cache_key if cache else None, reload=reload)

    if cache and result.cacheable and cache_key is not None:
        data = wrap_with_cache_write(data, cache_key)

    return data


def _map_rows(rows, mapping: FieldMapping):
    for index, row in enumerate(rows):
        sample = mapping.build(row, index)
        if sample is not None:
            yield sample


__all__ = ["load"]
