"""On-disk caching for :func:`benchy.data.load` and :meth:`Data.cached`.

Layout, under ``cache_root()`` (``$BENCHY_CACHE_DIR`` or
``~/.cache/benchy/data``)::

    <cache_root>/<source-hash>/samples.jsonl   one mapped Sample per line
    <cache_root>/<source-hash>/blobs/*         audio/image bytes, referenced
                                                by relative path from the
                                                JSONL (never base64-inlined)

``<source-hash>`` is a sha1 of the *resolved spec* — scheme + name + subset
+ split + revision — computed by each cacheable source's ``key_fn`` (see
:mod:`benchy.data.sources`). It deliberately does **not** include the field
mapping (``input=``/``expected=``/...): changing how you slice a dataset's
columns doesn't change what was downloaded. If you do change a mapping and
want fresh results, call :func:`clear_cache` (or point ``BENCHY_CACHE_DIR``
elsewhere) — the mapping is cheap to redo, the network fetch isn't.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchy.core import AudioPart, ImagePart, Sample
from benchy.data.dataset import Data

logger = logging.getLogger("benchy.data")

_CACHE_DIR_ENV = "BENCHY_CACHE_DIR"
_SAMPLES_FILE = "samples.jsonl"
_BLOBS_DIR = "blobs"


def cache_root() -> Path:
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "benchy" / "data"


def cache_dir_for_key(key: str) -> Path:
    return cache_root() / key


def is_cached(key: str) -> bool:
    return (cache_dir_for_key(key) / _SAMPLES_FILE).exists()


def clear_cache(key: str | None = None) -> None:
    """Delete one cache entry (``key``), or the entire cache root."""
    target = cache_dir_for_key(key) if key is not None else cache_root()
    if target.exists():
        shutil.rmtree(target)


# --------------------------------------------------------------------------
# Reading a cache
# --------------------------------------------------------------------------


def read_cached(key: str) -> Data:
    cache_dir = cache_dir_for_key(key)
    samples_path = cache_dir / _SAMPLES_FILE
    blobs_dir = cache_dir / _BLOBS_DIR

    def factory():
        with open(samples_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield _row_to_sample(json.loads(line), blobs_dir)

    return Data(factory, cache_key=key)


def _row_to_sample(row: dict[str, Any], blobs_dir: Path) -> Sample:
    return Sample(
        id=row["id"],
        input={k: _decode_value(v, blobs_dir) for k, v in row.get("input", {}).items()},
        expected=row.get("expected"),
        meta={k: _decode_value(v, blobs_dir) for k, v in row.get("meta", {}).items()},
    )


def _decode_value(value: Any, blobs_dir: Path) -> Any:
    if isinstance(value, Mapping) and value.get("__benchy_part__") in ("audio", "image"):
        kind = value["__benchy_part__"]
        ref = value.get("ref")
        path = str(blobs_dir / ref) if value.get("ref_kind") == "blob" else ref
        cls = AudioPart if kind == "audio" else ImagePart
        kwargs: dict[str, Any] = {"mime": value.get("mime", "application/octet-stream")}
        if value.get("ref_kind") == "url":
            kwargs["url"] = ref
        else:
            kwargs["path"] = path
        if kind == "audio" and value.get("sample_rate") is not None:
            kwargs["sample_rate"] = value["sample_rate"]
        return cls(**kwargs)
    return value


# --------------------------------------------------------------------------
# Writing a cache (materialize on full consumption; safe on partial)
# --------------------------------------------------------------------------


def materialize(data: Data) -> Data:
    """Force ``data`` to disk under a stable or session-scoped cache dir.

    Used by :meth:`Data.cached`. See the class docstring there for how the
    key is chosen.
    """
    key = data.cache_key
    if key is None:
        import uuid

        key = f"session/{uuid.uuid4().hex}"
    return wrap_with_cache_write(data, key)


def wrap_with_cache_write(data: Data, key: str) -> Data:
    """Return a ``Data`` that writes ``data`` to the cache as it is fully iterated.

    Reading it back is lazy and lives in :func:`read_cached`. Writing only
    finalizes (atomically) if the wrapped ``Data`` is iterated to
    exhaustion — ``take(5)`` on a fresh, uncached source leaves an
    incomplete ``.tmp`` file behind rather than a false-positive complete
    cache entry.
    """
    if is_cached(key):
        return read_cached(key)

    cache_dir = cache_dir_for_key(key)
    blobs_dir = cache_dir / _BLOBS_DIR

    def factory():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_dir / f".{_SAMPLES_FILE}.tmp"
        completed = False
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                for sample in data:
                    fh.write(json.dumps(_sample_to_row(sample, blobs_dir), ensure_ascii=False) + "\n")
                    yield sample
            completed = True
        finally:
            if completed:
                tmp_path.replace(cache_dir / _SAMPLES_FILE)
            else:
                tmp_path.unlink(missing_ok=True)

    return Data(factory, cache_key=key)


def _sample_to_row(sample: Sample, blobs_dir: Path) -> dict[str, Any]:
    return {
        "id": sample.id,
        "input": {k: _encode_value(k, sample.id, v, blobs_dir) for k, v in sample.input.items()},
        "expected": sample.expected,
        "meta": {k: _encode_value(k, sample.id, v, blobs_dir) for k, v in sample.meta.items()},
    }


def _encode_value(field: str, sample_id: str, value: Any, blobs_dir: Path) -> Any:
    if isinstance(value, (AudioPart, ImagePart)):
        kind = "audio" if isinstance(value, AudioPart) else "image"
        marker: dict[str, Any] = {"__benchy_part__": kind, "mime": value.mime}
        if isinstance(value, AudioPart) and value.sample_rate is not None:
            marker["sample_rate"] = value.sample_rate
        if value.data is not None:
            ext = _ext_for_mime(value.mime)
            safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_id)
            blob_name = f"{safe_id}__{field}{ext}"
            blobs_dir.mkdir(parents=True, exist_ok=True)
            (blobs_dir / blob_name).write_bytes(value.data)
            marker["ref"] = blob_name
            marker["ref_kind"] = "blob"
        elif value.path is not None:
            marker["ref"] = value.path
            marker["ref_kind"] = "path"
        elif value.url is not None:
            marker["ref"] = value.url
            marker["ref_kind"] = "url"
        return marker
    return value


def _ext_for_mime(mime: str) -> str:
    return {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(mime, "")


# --------------------------------------------------------------------------
# Cache key
# --------------------------------------------------------------------------


def compute_key(**parts: Any) -> str:
    """A stable sha1 over the resolved spec, prefixed for readability."""
    import hashlib

    scheme = parts.get("scheme", "source")
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{scheme}/{digest}"
