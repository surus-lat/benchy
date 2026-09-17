"""Reserved input keys that carry non-text modalities.

A ``Sample.input`` mapping may carry audio under the key ``audio`` or
``audio_path``, and images under ``image`` or ``image_path``. For each,
two value shapes are accepted -- this is a deliberate, documented decision
(the sibling worktrees producing samples were not available to consult, so
both shapes seen in the salvaged codebase and the HF ``datasets`` ecosystem
are handled):

- a bare filesystem path (``str`` / ``os.PathLike``), or raw ``bytes``
- the HuggingFace ``datasets`` audio/image feature dict, e.g.
  ``{"path": ..., "array": <np.ndarray>, "sampling_rate": <int>}`` for
  audio, or ``{"path": ..., "bytes": <bytes>}`` for images

This module owns translating those shapes into ``benchy.core.AudioPart`` /
``benchy.core.ImagePart``, and owns the ``CapabilityError`` message an
author sees when a Task needs a modality the target System doesn't have.
"""

from __future__ import annotations

import io
import os
import wave
from collections.abc import Mapping
from typing import Any

from benchy.core import AudioPart, Capabilities, CapabilityError, ImagePart, Part, SchemaViolation

AUDIO_KEYS: tuple[str, ...] = ("audio", "audio_path")
IMAGE_KEYS: tuple[str, ...] = ("image", "image_path")


def _pcm16_wav_bytes(array: Any, sample_rate: int) -> bytes:
    """Encode a raw waveform (the HF ``array`` field) as a 16-bit PCM WAV.

    ``AudioPart`` only carries bytes/path/url -- it has no notion of "a
    numpy array plus a sample rate" -- so a Sample built from an HF audio
    feature has to be encoded to a real audio container before it can
    become an AudioPart. WAV via the stdlib ``wave`` module keeps this
    dependency-free (no soundfile/librosa needed just to render a prompt).
    """
    import numpy as np

    arr = np.asarray(array)
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2")
    else:
        pcm = arr.astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate or 16000))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_audio_part(value: Any, *, field: str) -> AudioPart:
    if isinstance(value, Mapping):
        sample_rate = value.get("sampling_rate")
        if value.get("array") is not None:
            return AudioPart(
                data=_pcm16_wav_bytes(value["array"], sample_rate or 16000),
                sample_rate=sample_rate,
            )
        if value.get("bytes") is not None:
            return AudioPart(data=bytes(value["bytes"]), sample_rate=sample_rate)
        if value.get("path"):
            return AudioPart(path=str(value["path"]), sample_rate=sample_rate)
        raise SchemaViolation(
            f"input field {field!r} looks like an audio mapping but has none "
            f"of 'array', 'bytes', or 'path' (keys={list(value.keys())!r})"
        )
    if isinstance(value, (str, os.PathLike)):
        return AudioPart(path=str(value))
    if isinstance(value, (bytes, bytearray)):
        return AudioPart(data=bytes(value))
    raise SchemaViolation(
        f"input field {field!r} must be a path, bytes, or a "
        "{'path'|'array'|'bytes', 'sampling_rate'} mapping; "
        f"got {type(value)!r}"
    )


def to_image_part(value: Any, *, field: str) -> ImagePart:
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return ImagePart(data=bytes(value["bytes"]))
        if value.get("path"):
            return ImagePart(path=str(value["path"]))
        raise SchemaViolation(
            f"input field {field!r} looks like an image mapping but has "
            f"neither 'bytes' nor 'path' (keys={list(value.keys())!r})"
        )
    if isinstance(value, (str, os.PathLike)):
        return ImagePart(path=str(value))
    if isinstance(value, (bytes, bytearray)):
        return ImagePart(data=bytes(value))
    if hasattr(value, "save") and hasattr(value, "tobytes"):  # duck-typed PIL.Image
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        return ImagePart(data=buf.getvalue(), mime="image/png")
    raise SchemaViolation(
        f"input field {field!r} must be a path, bytes, a PIL-like image "
        f"(has .save()), or a {{'path'|'bytes'}} mapping; got {type(value)!r}"
    )


def extract_media(data: Mapping[str, Any]) -> tuple[dict[str, Any], list[Part]]:
    """Split a Sample.input mapping into (remaining text fields, media parts).

    Only the first matching key per modality is honoured -- a Sample is not
    expected to carry both ``audio`` and ``audio_path`` at once.
    """
    remaining = dict(data)
    parts: list[Part] = []
    for key in AUDIO_KEYS:
        if key in remaining:
            parts.append(to_audio_part(remaining.pop(key), field=key))
            break
    for key in IMAGE_KEYS:
        if key in remaining:
            parts.append(to_image_part(remaining.pop(key), field=key))
            break
    return remaining, parts


def capability_error(task_name: str, part: Part, caps: Capabilities) -> CapabilityError:
    """Build an actionable CapabilityError for a Part the System can't accept.

    The Task module has no visibility into the system registry (that is a
    sibling worktree's job and cannot be imported here), so the message
    names the missing *capability flag* precisely rather than inventing
    concrete system names it cannot verify.
    """
    if isinstance(part, AudioPart):
        cap_name, label, hint = "audio_in", "audio", "an ASR-capable model or an audio-to-text preprocessing system"
    elif isinstance(part, ImagePart):
        cap_name, label, hint = "image_in", "image", "a vision-capable (multimodal) model"
    else:
        cap_name, label, hint = "text_in", "text", "a text-capable system"
    have = getattr(caps, cap_name, None)
    return CapabilityError(
        f"task {task_name!r} needs {label} input (Capabilities.{cap_name}=True) "
        f"but the target system declares {cap_name}={have!r}. Use {hint}, or "
        f"preprocess the {label} to text before invoking this system."
    )
