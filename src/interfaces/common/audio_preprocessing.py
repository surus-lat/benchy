"""Audio preprocessing helpers for transcription interfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_audio_array(array: np.ndarray, sampling_rate: int, output_path: Path) -> None:
    """Persist a numpy audio array as a WAV file.

    Uses soundfile when available (preserves dtype/precision), otherwise falls
    back to scipy.io.wavfile with int16 quantization. Skips if the file already
    exists so this can be called inside dataset loaders cheaply.
    """
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(output_path), array, sampling_rate)
    except ImportError:
        from scipy.io.wavfile import write as wav_write
        wav_write(str(output_path), sampling_rate, (np.array(array) * 32767).astype(np.int16))


def save_audio_bytes(data: bytes, output_path: Path) -> None:
    """Persist raw audio file bytes (e.g. a WAV/FLAC payload from HF parquet).

    Skips if the file already exists. Used when datasets are streamed with
    ``Audio(decode=False)`` so we avoid pulling in heavy decoders (torchcodec,
    torch) just to round-trip what's already a valid audio container.
    """
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
