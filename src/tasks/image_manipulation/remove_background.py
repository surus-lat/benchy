"""Background removal (image cutout) task.

Dataset (expected local layout):
  <root>/
    image/ or images/   (input photos)
    mask/ or masks/     (ground-truth binary masks)

Configure via task config:
  dataset:
    name: ICM57  # or kapaxia, or any custom dataset
    source_dir: /path/to/custom-dataset  # optional, overrides default location

The model is expected to return an image artifact (ideally a PNG with alpha)
encoded as base64. The evaluation derives a predicted mask from the alpha
channel and compares it to the ground-truth mask.
"""

from __future__ import annotations

import json
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from ..common import BackgroundRemovalMetrics, MultimodalImageArtifactHandler

logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    "ICM57": {
        "download_url": "https://www.kaggle.com/api/v1/datasets/download/fineyouthpe/icm57-dataset",
        "zip_name": "icm57-dataset.zip",
        "extract_subdir": "ICM57",
        "allow_download": True,
    },
    "kapaxia": {
        "download_url": "",  # No download for local datasets
        "zip_name": "",
        "extract_subdir": "kapaxia",
        "allow_download": False,
    },
}


class RemoveBackground(MultimodalImageArtifactHandler):
    """Remove the background from an input image."""

    name = "remove_background"

    # Dataset bootstrap: download/unzip if needed.
    allow_missing_dataset = False
    dataset_name: str = "ICM57"  # Default dataset
    dataset_download_url: str = ""
    dataset_zip_name: str = ""
    dataset_extract_subdir: str = ""
    jsonl_name: str = "data.jsonl"

    # Prompting (keep it short - API has prompt length limits)
    system_prompt = "Remove the background from this image"
    user_prompt_template = "Remove the background"

    # Dataset layout defaults (can be overridden in config via config['dataset']).
    images_dirname: str = "image"
    masks_dirname: str = "mask"

    # Metrics
    mask_metrics = BackgroundRemovalMetrics()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Process dataset configuration
        if config and isinstance(config.get("dataset"), dict):
            ds = config["dataset"]
            
            # Get dataset name (can be overridden)
            self.dataset_name = ds.get("name", self.dataset_name)
            
            # Load dataset-specific configuration
            # If dataset not in DATASET_CONFIGS, use smart defaults (local dataset with same name as folder)
            if self.dataset_name in DATASET_CONFIGS:
                dataset_config = DATASET_CONFIGS[self.dataset_name]
            else:
                # Auto-detect: assume it's a local dataset in .data/<dataset_name>
                logger.info(
                    f"Dataset '{self.dataset_name}' not in DATASET_CONFIGS, "
                    f"assuming local dataset at .data/{self.dataset_name}"
                )
                dataset_config = {
                    "download_url": "",
                    "zip_name": "",
                    "extract_subdir": self.dataset_name,
                    "allow_download": False,
                }
            
            self.dataset_download_url = ds.get("download_url", dataset_config["download_url"])
            self.dataset_zip_name = ds.get("zip_name", dataset_config["zip_name"])
            self.dataset_extract_subdir = ds.get("extract_subdir", dataset_config["extract_subdir"])
            
            # Allow customizing directory names
            self.images_dirname = ds.get("images_dirname", self.images_dirname)
            self.masks_dirname = ds.get("masks_dirname", self.masks_dirname)
            self.jsonl_name = ds.get("jsonl_name", self.jsonl_name)
            
            # If source_dir is specified, override the default location
            if "source_dir" in ds:
                self.source_path = Path(ds["source_dir"])
        else:
            # Use default dataset config
            dataset_config = DATASET_CONFIGS.get(self.dataset_name, DATASET_CONFIGS["ICM57"])
            self.dataset_download_url = dataset_config["download_url"]
            self.dataset_zip_name = dataset_config["zip_name"]
            self.dataset_extract_subdir = dataset_config["extract_subdir"]

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        # This task uses a fixed prompt; sample-specific prompts can be added later.
        return self.system_prompt, self.user_prompt_template

    def _resolve_dir(self, root: Path, primary: str, fallback: str) -> Optional[Path]:
        cand = root / primary
        if cand.exists() and cand.is_dir():
            return cand
        cand = root / fallback
        if cand.exists() and cand.is_dir():
            return cand
        return None

    def _expected_dataset_root_candidates(self, base_dir: Path) -> List[Path]:
        # Common layouts:
        # - <base>/image + <base>/mask
        # - <base>/<extract_subdir>/image + <base>/<extract_subdir>/mask
        return [
            base_dir,
            base_dir / self.dataset_extract_subdir,
        ]

    def _find_dataset_root(self, base_dir: Path) -> Optional[Path]:
        # 1) direct candidates
        for root in self._expected_dataset_root_candidates(base_dir):
            images_dir = self._resolve_dir(root, self.images_dirname, "images")
            masks_dir = self._resolve_dir(root, self.masks_dirname, "masks")
            if images_dir and masks_dir:
                return root

        # 2) search one level deep for a folder that contains both image + mask dirs
        if base_dir.exists():
            for child in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
                images_dir = self._resolve_dir(child, self.images_dirname, "images")
                masks_dir = self._resolve_dir(child, self.masks_dirname, "masks")
                if images_dir and masks_dir:
                    return child

        return None

    def _download_zip(self, dest_zip: Path) -> None:
        dest_zip.parent.mkdir(parents=True, exist_ok=True)

        # Prefer curl (matches how you downloaded it); fall back to urllib.
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(dest_zip), self.dataset_download_url],
                check=True,
            )
            return
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"curl download failed: {exc}") from exc

        req = Request(self.dataset_download_url, headers={"User-Agent": "benchy/0.1"})
        try:
            with urlopen(req) as resp, open(dest_zip, "wb") as handle:
                handle.write(resp.read())
        except Exception as exc:
            raise RuntimeError(f"urllib download failed: {exc}") from exc

    def _extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

    def _ensure_dataset_present(self) -> Path:
        # If a source_dir was configured and looks valid, use it.
        if self.source_path:
            root = self._find_dataset_root(self.source_path)
            if root is not None:
                return root
            # If source_dir was specified but not found, raise error
            raise FileNotFoundError(
                f"{self.dataset_name} dataset not found at {self.source_path}. "
                f"Expected structure: {self.dataset_extract_subdir}/image/ and "
                f"{self.dataset_extract_subdir}/masks/ (or at the root level)"
            )

        # For local datasets without download support, check .data/ directory
        dataset_local_path = self.data_dir / self.dataset_extract_subdir
        if not self.dataset_download_url:
            # Local-only dataset (e.g., kapaxia)
            root = self._find_dataset_root(self.data_dir)
            if root is not None:
                return root
            raise FileNotFoundError(
                f"{self.dataset_name} dataset not found at {dataset_local_path}. "
                f"Expected structure: {self.dataset_extract_subdir}/image/ and "
                f"{self.dataset_extract_subdir}/masks/"
            )

        # For downloadable datasets, use the task-local `.data/` folder and bootstrap if needed.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        root = self._find_dataset_root(self.data_dir)
        if root is not None:
            return root

        zip_path = self.data_dir / self.dataset_zip_name
        if not zip_path.exists():
            logger.info("Downloading %s dataset zip to %s", self.dataset_name, zip_path)
            self._download_zip(zip_path)

        logger.info("Extracting dataset zip %s into %s", zip_path, self.data_dir)
        self._extract_zip(zip_path, self.data_dir)

        root = self._find_dataset_root(self.data_dir)
        if root is None:
            raise FileNotFoundError(
                f"Dataset extracted but expected folders not found under {self.data_dir}."
            )
        return root

    def _write_jsonl(self, jsonl_path: Path, samples: List[Dict[str, Any]]) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def load(self) -> None:
        dataset_root = self._ensure_dataset_present()
        self.dataset_root = dataset_root
        self.dataset_data = self._load_samples(dataset_root)

        # Build a JSONL dataset for inference/debugging (same sample payload the engine uses).
        jsonl_path = self.data_dir / self.jsonl_name
        if not jsonl_path.exists():
            self._write_jsonl(jsonl_path, list(self.dataset_data or []))

    def _load_samples(self, dataset_root: Path) -> List[Dict[str, Any]]:
        images_dir = self._resolve_dir(dataset_root, self.images_dirname, "images")
        masks_dir = self._resolve_dir(dataset_root, self.masks_dirname, "masks")

        if not images_dir or not masks_dir:
            logger.warning(
                "Dataset root does not contain expected folders. "
                "Expected %s/%s (or images/masks). Found: %s",
                self.images_dirname,
                self.masks_dirname,
                str(dataset_root),
            )
            return []

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        mask_exts = [".png", ".jpg", ".jpeg", ".webp"]

        image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        samples: List[Dict[str, Any]] = []
        for idx, image_path in enumerate(image_paths):
            mask_path: Optional[Path] = None
            for ext in mask_exts:
                candidate = masks_dir / f"{image_path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is None:
                continue

            samples.append(
                {
                    "id": f"{self.get_task_name()}_{idx:06d}_{image_path.stem}",
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "expected": str(mask_path.resolve()),
                    "text": "",  # reserved for future sample-specific prompts
                }
            )

        logger.info("Loaded %d background-removal samples", len(samples))
        return samples

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if error:
            return self.get_error_metrics(error, error_type)

        mask_path = sample.get("mask_path") or expected
        if not mask_path:
            return self.get_error_metrics("Missing mask_path in sample", "invalid_response")

        metrics, metrics_error = self.mask_metrics.per_sample(
            prediction=prediction,
            mask_path=str(mask_path),
        )
        if metrics_error:
            return self.get_error_metrics(metrics_error, "invalid_response")
        return {"valid": True, **metrics}

    def get_error_metrics(self, error: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "mask_iou": 0.0,
            "mask_precision": 0.0,
            "mask_recall": 0.0,
            "mask_f1": 0.0,
            "mask_accuracy": 0.0,
            "mask_mae": 1.0,
            "pred_resized": 0.0,
        }

    def aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate background removal metrics with an overall score.
        
        The score is based on F1 (harmonic mean of precision and recall),
        which is the standard metric for segmentation tasks.
        """
        aggregated = super().aggregate_metrics(all_metrics)
        
        # Add overall score (use F1 as it balances precision and recall)
        if "mask_f1" in aggregated:
            aggregated["score"] = aggregated["mask_f1"]
        elif "mask_iou" in aggregated:
            # Fallback to IoU if F1 not available
            aggregated["score"] = aggregated["mask_iou"]
        else:
            aggregated["score"] = 0.0
        
        # Add "acceptable" threshold metric (90% IoU)
        # Counts samples meeting quality threshold
        acceptable_threshold = 0.80
        valid_samples = [m for m in all_metrics if m.get("valid", True)]
        acceptable_count = sum(
            1 for m in valid_samples 
            if m.get("mask_iou", 0) >= acceptable_threshold
        )
        total_valid = len(valid_samples)
        
        aggregated["acceptable_samples"] = acceptable_count
        aggregated["acceptable_threshold"] = acceptable_threshold
        aggregated["acceptable_rate"] = acceptable_count / total_valid if total_valid > 0 else 0.0
        
        return aggregated
