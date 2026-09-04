"""The source registry: ``scheme:`` -> raw-row loader, plus the built-in
local sources (``jsonl:``, ``csv:``/``tsv:``, ``glob:``).

A source loader's job stops well short of building ``Sample`` objects — that
generic step (column mapping, id generation, Audio/ImagePart detection) is
:mod:`benchy.data.mapping`, shared by every source. A loader just needs to
produce a :class:`SourceResult`: a lazy, re-iterable factory over raw
``Mapping[str, Any]`` rows, plus enough metadata (a length hint, a directory
to resolve relative media paths against, an optional cache identity) for the
rest of the pipeline to do its job.

The ``hf:`` loader lives in :mod:`benchy.data.hf` (it needs its own module so
tests can monkeypatch the one network-touching call without pulling in the
rest of this file).
"""

from __future__ import annotations

import csv as csv_module
import glob as glob_module
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from benchy.core import LoadError

logger = logging.getLogger("benchy.data")


@dataclass(frozen=True)
class SourceResult:
    """What a source loader hands back to :func:`benchy.data.load`.

    ``rows`` must be a zero-argument callable that returns a *fresh*
    iterator over raw rows every time it's called — the same re-iterability
    requirement ``Data`` has, pushed down one layer.
    """

    rows: Callable[[], Iterator[Mapping[str, Any]]]
    len_hint: int | None = None
    base_dir: Path | None = None
    cache_key: str | None = None
    cacheable: bool = False
    #: ``split_reloader(new_split) -> SourceResult`` — re-resolve this same
    #: source with a different split. ``benchy.data.loader`` wraps this into
    #: ``Data.split()``; a source with no notion of "splits" leaves it None.
    split_reloader: Callable[[str], "SourceResult"] | None = None


#: ``loader(spec, **kwargs) -> SourceResult``, where ``spec`` is everything
#: after the ``scheme:`` prefix (or the bare path, for extension-inferred
#: sources).
SourceLoader = Callable[..., SourceResult]

#: ``key_fn(spec, **kwargs) -> str | None`` — a *pure, no-I/O* function that
#: computes a cache identity for a source's resolved spec, so ``load()`` can
#: check the on-disk cache before ever invoking the (possibly networked)
#: loader. Only cacheable sources (currently just ``hf:``) register one.
CacheKeyFn = Callable[..., "str | None"]

_REGISTRY: dict[str, SourceLoader] = {}
_KEY_FNS: dict[str, CacheKeyFn] = {}


def register_source(scheme: str, loader: SourceLoader, *, key_fn: CacheKeyFn | None = None) -> None:
    """Register (or override) the loader for a ``scheme:`` prefix."""
    _REGISTRY[scheme] = loader
    if key_fn is not None:
        _KEY_FNS[scheme] = key_fn


def cache_key_fn_for(scheme: str) -> CacheKeyFn | None:
    return _KEY_FNS.get(scheme)


#: Public, read-only view of the registry — ``sources.keys()``,
#: ``sources["hf"]``, ``"jsonl" in sources``, etc.
sources: Mapping[str, SourceLoader] = MappingProxyType(_REGISTRY)


#: Bare-path extension -> scheme, used when ``load()`` is given a path with
#: no ``scheme:`` prefix at all.
EXTENSION_SCHEMES: Mapping[str, str] = MappingProxyType(
    {
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".json": "jsonl",
        ".csv": "csv",
        ".tsv": "tsv",
    }
)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``"scheme:rest"`` into ``(scheme, rest)``.

    A bare path (no scheme) is resolved by extension via
    ``EXTENSION_SCHEMES``, unless it contains glob wildcards (``*`` or
    ``?``), in which case it's treated as ``glob:``. Raises ``LoadError``
    for an unrecognized scheme or an extension with no known source.
    """
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        # Guard against POSIX absolute paths and Windows drive letters being
        # mistaken for a scheme (single-letter "prefix", or the rest starts
        # with "//" for something like a literal path containing a colon).
        if prefix.isidentifier() or (prefix.isalpha() and len(prefix) > 1):
            if prefix in _REGISTRY:
                return prefix, rest
            raise LoadError(
                f"unknown source scheme {prefix!r}. Registered schemes: "
                f"{', '.join(sorted(_REGISTRY)) or '(none)'}"
            )
    # Bare path.
    if any(ch in spec for ch in ("*", "?", "[")):
        return "glob", spec
    suffix = Path(spec).suffix.lower()
    scheme = EXTENSION_SCHEMES.get(suffix)
    if scheme is None:
        raise LoadError(
            f"cannot infer a source from {spec!r} (extension {suffix!r} unrecognized). "
            f"Use an explicit 'scheme:' prefix. Known extensions: "
            f"{', '.join(sorted(EXTENSION_SCHEMES))}"
        )
    return scheme, spec


# --------------------------------------------------------------------------
# jsonl:
# --------------------------------------------------------------------------


def _load_jsonl(spec: str, **_: Any) -> SourceResult:
    path = Path(spec)
    if not path.exists():
        raise LoadError(f"jsonl source not found: {path}")

    def rows() -> Iterator[Mapping[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d: invalid JSON, skipping (%s)", path, lineno, exc)
                    continue
                if not isinstance(row, Mapping):
                    logger.warning("%s:%d: not a JSON object, skipping", path, lineno)
                    continue
                yield row

    return SourceResult(rows=rows, base_dir=path.resolve().parent)


# --------------------------------------------------------------------------
# csv: / tsv:
# --------------------------------------------------------------------------


def _load_csv(spec: str, *, delimiter: str | None = None, **_: Any) -> SourceResult:
    path = Path(spec)
    if not path.exists():
        raise LoadError(f"csv source not found: {path}")
    resolved_delimiter = delimiter or ("\t" if path.suffix.lower() == ".tsv" else ",")

    def rows() -> Iterator[Mapping[str, Any]]:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv_module.DictReader(fh, delimiter=resolved_delimiter)
            yield from reader

    return SourceResult(rows=rows, base_dir=path.resolve().parent)


def _load_tsv(spec: str, **kwargs: Any) -> SourceResult:
    kwargs.setdefault("delimiter", "\t")
    return _load_csv(spec, **kwargs)


# --------------------------------------------------------------------------
# glob:
# --------------------------------------------------------------------------

_MEDIA_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}


def _glob_base_dir(pattern: str) -> Path:
    """The largest fixed (non-wildcard) leading directory of a glob pattern."""
    parts = Path(pattern).parts
    fixed: list[str] = []
    for part in parts:
        if any(ch in part for ch in ("*", "?", "[")):
            break
        fixed.append(part)
    base = Path(*fixed) if fixed else Path(".")
    return base.resolve()


def _load_glob(spec: str, **_: Any) -> SourceResult:
    matches = sorted(Path(p) for p in glob_module.glob(spec, recursive=True))
    files = [p for p in matches if p.is_file()]
    if not files:
        raise LoadError(f"glob pattern matched no files: {spec!r}")
    base_dir = _glob_base_dir(spec)

    def rows() -> Iterator[Mapping[str, Any]]:
        for file in files:
            suffix = file.suffix.lower()
            if suffix == ".jsonl":
                with open(file, encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            logger.warning("%s:%d: invalid JSON, skipping (%s)", file, lineno, exc)
                            continue
                        yield row
            elif suffix == ".json":
                try:
                    payload = json.loads(file.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    logger.warning("%s: invalid JSON, skipping (%s)", file, exc)
                    continue
                if isinstance(payload, list):
                    yield from (r for r in payload if isinstance(r, Mapping))
                elif isinstance(payload, Mapping):
                    row = dict(payload)
                    row.setdefault("id", file.stem)
                    yield row
                else:
                    logger.warning("%s: JSON root is neither object nor array, skipping", file)
            elif suffix in _MEDIA_EXTENSIONS:
                yield {"id": file.stem, "path": str(file.resolve())}
            else:
                logger.warning("%s: unrecognized file type, skipping", file)

    return SourceResult(rows=rows, base_dir=base_dir)


register_source("jsonl", _load_jsonl)
register_source("csv", _load_csv)
register_source("tsv", _load_tsv)
register_source("glob", _load_glob)


def _register_hf() -> None:
    # Importing the module registers "hf" as a side effect (see hf.py's
    # bottom-of-module `register_source(...)` call). Kept as a lazy import
    # here purely so `import benchy.data.sources` never pulls in `datasets`.
    import benchy.data.hf  # noqa: F401


_register_hf()
