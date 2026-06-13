"""FLEURS Brazilian Portuguese (pt_br) ASR subtask."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Audio, load_dataset

from ...interfaces.common.audio_preprocessing import save_audio_bytes
from ._transcription_handler import TranscriptionHandler

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent


class FleursPtBr(TranscriptionHandler):
    """ASR evaluation on FLEURS Brazilian Portuguese (pt_br, test split)."""

    name = "fleurs_pt_br"
    display_name = "FLEURS Portuguese (Brazil)"
    description = "FLEURS pt_br test split — Brazilian Portuguese ASR"

    language = "pt"
    locale = "pt_br"
    dataset_name = "google/fleurs"
    dataset_config = "pt_br"
    split = "test"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if self.config:
            self.split = self.config.get("split", self.split)
        self.data_dir = _REPO_ROOT / ".data" / "transcription" / self.locale

    def load_dataset(self) -> List[Dict[str, Any]]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Loading {self.dataset_name} config={self.dataset_config} split={self.split}"
        )
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
        )
        # Stream raw audio bytes instead of decoded arrays — avoids pulling in
        # torchcodec/torch just to re-encode bytes that are already valid WAV.
        ds = ds.cast_column("audio", Audio(decode=False))

        samples: List[Dict[str, Any]] = []
        for item in ds:
            audio_path = self.data_dir / f"{item['id']}.wav"
            save_audio_bytes(item["audio"]["bytes"], audio_path)
            samples.append(
                {
                    "id": f"{self.locale}_{item['id']}",
                    "audio_path": str(audio_path),
                    "expected": item["transcription"],
                    "language": self.language,
                    "locale": self.locale,
                }
            )

        logger.info(f"Loaded {len(samples)} samples for {self.locale}")
        return samples
