"""Multimodal image-to-image (artifact) format handler.

This handler targets tasks where the request is multimodal (an input image +
text prompt) and the model returns an image artifact (commonly via base64).

The core engine currently treats model outputs as JSON/text. For artifact tasks,
the "prediction" passed to calculate_metrics() is expected to be either:
- a base64 string (optionally wrapped in a data URL), or
- a path to an image file on disk, or
- raw bytes.

Subclasses implement `_load_samples(dataset_root)` to build samples containing at
least:
- id: unique identifier
- image_path: path to the input image (used by interfaces that support multimodal)
- expected: task-specific expected payload (often a mask path for segmentation-like eval)
- text: prompt/instruction (used by get_prompt)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseHandler

logger = logging.getLogger(__name__)


class MultimodalImageArtifactHandler(BaseHandler):
    """Handler for image+prompt -> image artifact tasks."""

    input_type: str = "image"
    requires_multimodal: bool = True
    requires_files: bool = True
    requires_schema: bool = False
    answer_type: str = "image_artifact"

    image_field: str = "image_path"
    source_dir: Optional[Any] = None
    copy_source_to_data_dir: bool = False
    allow_missing_dataset: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Optional override source_dir from config if provided.
        # Handler runner passes dataset config under config["dataset"].
        source_dir = None
        if config:
            if "source_dir" in config:
                source_dir = config["source_dir"]
            else:
                dataset_cfg = config.get("dataset")
                if isinstance(dataset_cfg, dict) and dataset_cfg.get("source_dir"):
                    source_dir = dataset_cfg.get("source_dir")
        if source_dir is not None:
            self.source_dir = source_dir

        self.source_path: Optional[Path] = Path(str(self.source_dir)) if self.source_dir else None
        self.dataset_root: Optional[Path] = None

    def _dataset_present(self, root: Path) -> bool:
        if not root.exists():
            return False
        try:
            return any(root.iterdir())
        except OSError:
            return False

    def _ensure_dataset_root(self) -> Optional[Path]:
        """Resolve a dataset root directory.

        Preference order:
        1) task-local `.data/` folder (self.data_dir) if it looks populated
        2) config-provided `source_dir` (self.source_path), optionally copied into `.data/`
        """
        if self._dataset_present(self.data_dir):
            return self.data_dir

        if self.source_path and self.source_path.exists():
            if self.copy_source_to_data_dir:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copytree(self.source_path, self.data_dir, dirs_exist_ok=True)
                except TypeError:
                    # Python <3.8 fallback not needed (repo is 3.12), but keep defensive.
                    for entry in self.source_path.iterdir():
                        dest = self.data_dir / entry.name
                        if entry.is_dir():
                            shutil.copytree(entry, dest)
                        else:
                            shutil.copy2(entry, dest)
                return self.data_dir
            return self.source_path

        return None

    def load(self) -> None:
        dataset_root = self._ensure_dataset_root()
        if dataset_root is None:
            if self.allow_missing_dataset:
                logger.warning(
                    "Dataset not found for %s. Provide config['dataset']['source_dir'] "
                    "(or ship a populated `.data/` folder). Loading 0 samples.",
                    self.get_task_name(),
                )
                self.dataset_root = None
                self.dataset_data = []
                return
            raise FileNotFoundError(
                f"Dataset not found for {self.get_task_name()}.\n"
                "Provide a dataset source via config['dataset']['source_dir'] (or config['source_dir']), "
                "or ship a populated `.data/` folder next to the task file."
            )

        self.dataset_root = dataset_root
        self.dataset_data = self._load_samples(dataset_root)

    def _load_samples(self, dataset_root: Path) -> List[Dict[str, Any]]:
        """Load samples from dataset_root.

        Subclasses must implement. Return a list of sample dicts.
        """
        raise NotImplementedError

