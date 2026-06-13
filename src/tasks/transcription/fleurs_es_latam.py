"""FLEURS Latin American Spanish (es_419) ASR subtask."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from ...interfaces.common.audio_preprocessing import save_audio_array
from ._transcription_handler import TranscriptionHandler

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent


class FleursEsLatam(TranscriptionHandler):
    """ASR evaluation on FLEURS Latin American Spanish (es_419, test split)."""

    name = "fleurs_es_latam"
    display_name = "FLEURS Spanish (Latin America)"
    description = "FLEURS es_419 test split — Latin American Spanish ASR"

    language = "es"
    locale = "es_419"
    dataset_name = "google/fleurs"
    dataset_config = "es_419"
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
            trust_remote_code=True,
        )

        samples: List[Dict[str, Any]] = []
        for item in ds:
            audio_path = self.data_dir / f"{item['id']}.wav"
            save_audio_array(
                item["audio"]["array"],
                item["audio"]["sampling_rate"],
                audio_path,
            )
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
