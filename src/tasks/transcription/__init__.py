"""Transcription task group.

ASR evaluation on Latin American Spanish and Brazilian Portuguese via the
google/fleurs benchmark. Powered by ``OpenAIAudioInterface`` for Whisper-style
endpoints. Subtasks are auto-discovered from the .py files in this directory.
"""

from .fleurs_es_latam import FleursEsLatam
from .fleurs_pt_br import FleursPtBr

__all__ = ["FleursEsLatam", "FleursPtBr"]
