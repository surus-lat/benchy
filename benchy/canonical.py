# VENDORED from programpipeline 158119be4b50a196b8a132339cd551496d54bc62 (program_pipeline/canonical.py, only the JSON encoding functions) — no editar
"""Canonical JSON: one encoder, one content-id per JSON value.

Keys sorted, compact separators, UTF-8 with non-ASCII preserved, and non-finite
numbers REJECTED (no non-standard ``NaN``/``Infinity`` tokens). VENDORED copy of
the reference; parity is locked by fixed vectors in
``tests/test_canonical_parity.py``.

Stdlib-only, imports nothing from the package.
"""
import hashlib
import json


def _canonical(value) -> bytes:
    """Canonical UTF-8 bytes of ``value``; the only encoder."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_bytes(value) -> bytes:
    """Canonical bytes, for callers that hash or write them directly."""
    return _canonical(value)


def canonical_text(value) -> str:
    """Canonical text form, for callers that keep the string around."""
    return _canonical(value).decode("utf-8")


def sha256_of(value) -> str:
    """Content id of a JSON value: identical on every code path."""
    return hashlib.sha256(_canonical(value)).hexdigest()
