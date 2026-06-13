"""Tests for save_audio_array WAV utility."""

import sys
from pathlib import Path

import numpy as np

from src.interfaces.common import audio_preprocessing
from src.interfaces.common.audio_preprocessing import save_audio_array


def test_save_audio_array_writes_file(tmp_path: Path):
    array = np.zeros(1600, dtype=np.float32)
    output = tmp_path / "out.wav"
    save_audio_array(array, 16000, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_save_audio_array_skips_if_exists(tmp_path: Path, monkeypatch):
    output = tmp_path / "already.wav"
    output.write_bytes(b"placeholder")
    original_size = output.stat().st_size

    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("soundfile.write should not be called when file exists")

    import soundfile as sf
    monkeypatch.setattr(sf, "write", boom)
    save_audio_array(np.zeros(10, dtype=np.float32), 16000, output)
    assert output.stat().st_size == original_size


def test_save_audio_array_falls_back_to_scipy(tmp_path: Path, monkeypatch):
    # Force the soundfile import inside save_audio_array to raise so the
    # scipy fallback path is exercised.
    monkeypatch.setitem(sys.modules, "soundfile", None)
    output = tmp_path / "fallback.wav"
    save_audio_array(np.zeros(800, dtype=np.float32), 8000, output)
    assert output.exists()
    assert output.stat().st_size > 0
