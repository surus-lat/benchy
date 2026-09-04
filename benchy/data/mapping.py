"""Field mapping: turn a raw source row into a ``benchy.core.Sample``.

This is the one generic layer every source (``hf:``, ``jsonl:``, ``csv:``,
``glob:``, and anything a caller registers) goes through, which is what lets
``load()`` accept the same ``input=`` / ``expected=`` / ``id=`` / ``meta=``
kwargs regardless of source.

Two responsibilities live here:

1. **Column mapping** — resolving the declarative ``input``/``expected``/
   ``id``/``meta`` arguments (or their boring, documented defaults) against
   a raw row's actual keys, and building the ``Sample``.
2. **Content-type detection** — a raw value that looks like audio or an
   image (an HF ``datasets.Audio``/``Image`` feature dict, a PIL Image, raw
   bytes, or a string with a recognized media extension) is converted into
   a ``benchy.core.AudioPart`` / ``ImagePart``. Everything else — strings,
   numbers, bools, lists, plain dicts — passes through untouched as the
   native JSON value. Task.render() only ever has to handle two shapes per
   input field: a core ``Part``, or plain data.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchy.core import AudioPart, ImagePart, LoadError, Sample

#: Boring, documented defaults — no fuzzy alias hunting. If a source uses
#: different column names, say so explicitly via ``id=``/``expected=``.
DEFAULT_ID_COLUMNS = ("id",)
DEFAULT_EXPECTED_COLUMNS = ("expected",)

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_AUDIO_DICT_KEYS = {"array", "sampling_rate", "bytes", "path"}
_IMAGE_DICT_KEYS = {"bytes", "path"}


class FieldMapping:
    """Resolved column mapping for one ``load()`` call.

    ``input_map`` is ``{dest_key_in_sample_input: source_column}``. When the
    caller doesn't pass ``input=``, it's left ``None`` until the first row is
    seen, at which point every column not claimed by ``id``/``expected``/
    ``meta`` becomes an identity entry (``{col: col}``) — "leftover columns
    become input", the same boring rule for every source.
    """

    def __init__(
        self,
        *,
        input: Mapping[str, str] | None = None,
        expected: str | None = None,
        id: str | None = None,
        meta: Sequence[str] | Mapping[str, str] | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.input_map = dict(input) if input is not None else None
        self.expected_col = expected
        self.id_col = id
        self.meta_map = _normalize_meta(meta)
        self.base_dir = base_dir
        self._resolved_input_map: dict[str, str] | None = self.input_map
        self._checked = False

    def resolve_against(self, columns: Iterable[str]) -> None:
        """Validate explicit mappings against real columns; fill in defaults.

        Called once, against the first available row's keys. Raises
        :class:`benchy.core.LoadError` naming the available columns when an
        explicit mapping points at a column that isn't there.
        """
        cols = list(columns)
        col_set = set(cols)

        def check(name: str, col: str) -> None:
            if col not in col_set:
                raise LoadError(
                    f"{name}={col!r} does not match any column. Available columns: "
                    f"{', '.join(sorted(col_set)) or '(none)'}"
                )

        if self.input_map is not None:
            for src in self.input_map.values():
                check("input", src)
        if self.expected_col is not None:
            check("expected", self.expected_col)
        if self.id_col is not None:
            check("id", self.id_col)
        for src in self.meta_map.values():
            check("meta", src)

        if self._resolved_input_map is None:
            claimed = set(self.meta_map.values())
            if self.expected_col is not None:
                claimed.add(self.expected_col)
            elif any(c in col_set for c in DEFAULT_EXPECTED_COLUMNS):
                claimed.update(c for c in DEFAULT_EXPECTED_COLUMNS if c in col_set)
            if self.id_col is not None:
                claimed.add(self.id_col)
            elif any(c in col_set for c in DEFAULT_ID_COLUMNS):
                claimed.update(c for c in DEFAULT_ID_COLUMNS if c in col_set)
            self._resolved_input_map = {c: c for c in cols if c not in claimed}
        self._checked = True

    def build(self, row: Mapping[str, Any], index: int) -> Sample | None:
        """Build a ``Sample`` from one raw row, or ``None`` to skip it (malformed)."""
        if not self._checked:
            self.resolve_against(row.keys())

        input_map = self._resolved_input_map or {}
        try:
            input_dict = {
                dest: _to_part_or_value(row[src], base_dir=self.base_dir)
                for dest, src in input_map.items()
                if src in row
            }
        except Exception as exc:  # noqa: BLE001 - malformed row, warn-and-skip
            import logging

            logging.getLogger("benchy.data").warning(
                "row %d: could not build input (%s), skipping", index, exc
            )
            return None

        expected: Any = None
        expected_col = self.expected_col or next(
            (c for c in DEFAULT_EXPECTED_COLUMNS if c in row), None
        )
        if expected_col is not None:
            expected = row.get(expected_col)

        id_col = self.id_col or next((c for c in DEFAULT_ID_COLUMNS if c in row), None)
        if id_col is not None and row.get(id_col) is not None:
            sample_id = str(row[id_col])
        else:
            sample_id = _fallback_id(row, index)

        meta = {dest: row[src] for dest, src in self.meta_map.items() if src in row}

        return Sample(id=sample_id, input=input_dict, expected=expected, meta=meta)


def _normalize_meta(meta: Sequence[str] | Mapping[str, str] | None) -> dict[str, str]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta)
    return {name: name for name in meta}


def _fallback_id(row: Mapping[str, Any], index: int) -> str:
    """Deterministic id when no id column is present: content hash, else index.

    Never a random uuid — run comparison and caching depend on stability.
    """
    try:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=True, default=_stringify)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - some value truly can't be serialized
        return f"row_{index:06d}"


def _stringify(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha1(bytes(value)).hexdigest()[:16]
    return str(value)


def _guess_mime(path: str, default: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or default


def _resolve_path(raw: str, base_dir: Path | None) -> str:
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if base_dir is not None:
        return str((base_dir / p).resolve())
    return str(p)


def _to_part_or_value(value: Any, *, base_dir: Path | None) -> Any:
    """Type-directed conversion: media-shaped values become Audio/ImagePart.

    Everything else (str, int, float, bool, list, plain dict, None) passes
    through untouched as the native JSON value.
    """
    if isinstance(value, Mapping):
        keys = set(value.keys())
        if keys & _AUDIO_DICT_KEYS and ("array" in keys or "bytes" in keys):
            return _audio_part_from_dict(value, base_dir=base_dir)
        if keys & _IMAGE_DICT_KEYS and "bytes" in keys and "array" not in keys:
            # Ambiguous with audio-bytes-only dicts; only image loaders should
            # produce this shape, callers can always pass explicit ImagePart.
            return value
        return value

    if isinstance(value, (bytes, bytearray)):
        return AudioPart(data=bytes(value))

    type_name = type(value).__name__
    if type_name == "Image" and hasattr(value, "save"):
        # A PIL.Image.Image from a `datasets.Image` feature — encode in-memory.
        import io

        buf = io.BytesIO()
        fmt = getattr(value, "format", None) or "PNG"
        value.save(buf, format=fmt)
        return ImagePart(data=buf.getvalue(), mime=f"image/{fmt.lower()}")

    if isinstance(value, str):
        suffix = Path(value).suffix.lower()
        if suffix in _AUDIO_EXTENSIONS:
            resolved = _resolve_path(value, base_dir)
            return AudioPart(path=resolved, mime=_guess_mime(resolved, "audio/wav"))
        if suffix in _IMAGE_EXTENSIONS:
            resolved = _resolve_path(value, base_dir)
            return ImagePart(path=resolved, mime=_guess_mime(resolved, "image/png"))
        return value

    return value


def _audio_part_from_dict(value: Mapping[str, Any], *, base_dir: Path | None) -> AudioPart:
    """An HF ``datasets.Audio``-feature-shaped dict -> ``AudioPart``.

    Prefers ``bytes`` (present when the column was cast with
    ``Audio(decode=False)``, or straight from a parquet/arrow blob column) —
    that's a ready-to-use encoded container (WAV/FLAC/...), no decoding
    dependency needed. Falls back to a decoded ``array`` + ``sampling_rate``
    pair, which is re-encoded to an in-memory WAV via ``soundfile``.
    """
    raw_bytes = value.get("bytes")
    if raw_bytes is not None:
        mime = "audio/wav"
        path = value.get("path")
        if isinstance(path, str) and path:
            mime = _guess_mime(path, "audio/wav")
        return AudioPart(data=bytes(raw_bytes), mime=mime)

    array = value.get("array")
    sampling_rate = value.get("sampling_rate")
    if array is not None and sampling_rate is not None:
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, array, int(sampling_rate), format="WAV")
        return AudioPart(data=buf.getvalue(), mime="audio/wav", sample_rate=int(sampling_rate))

    path = value.get("path")
    if isinstance(path, str) and path:
        resolved = _resolve_path(path, base_dir)
        return AudioPart(path=resolved, mime=_guess_mime(resolved, "audio/wav"))

    raise LoadError(f"audio-shaped dict has neither bytes, array+sampling_rate, nor path: {value!r}")
