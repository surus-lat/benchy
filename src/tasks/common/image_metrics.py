"""Image / mask metric utilities (dependency-light).

These helpers are used by image manipulation tasks such as background removal,
where evaluation compares a predicted cutout (often with alpha) to a ground-truth
mask.

External Usage Example:
-----------------------
The mask comparison metrics can be called from outside benchy to validate results:

    from benchy.tasks.common import BackgroundRemovalMetrics
    
    # Initialize the metrics calculator
    metrics = BackgroundRemovalMetrics()
    
    # Calculate metrics for a prediction (base64, bytes, or file path)
    result, error = metrics.per_sample(
        prediction="<base64_string_or_path_or_bytes>",
        mask_path="/path/to/ground_truth_mask.png"
    )
    
    # Check results
    if error:
        print(f"Error: {error}")
    else:
        print(f"IoU: {result['mask_iou']:.3f}")
        print(f"Precision: {result['mask_precision']:.3f}")
        print(f"Recall: {result['mask_recall']:.3f}")
        print(f"F1: {result['mask_f1']:.3f}")
        print(f"Accuracy: {result['mask_accuracy']:.3f}")
        print(f"MAE: {result['mask_mae']:.3f}")

Alternative - Using individual functions:
    
    from benchy.tasks.common import (
        coerce_prediction_to_image_bytes,
        predicted_image_to_mask,
        load_mask_as_bool,
        binary_mask_metrics,
    )
    
    # Convert prediction to image bytes
    img_bytes = coerce_prediction_to_image_bytes(prediction)
    
    # Extract foreground mask from alpha channel
    pred_mask = predicted_image_to_mask(img_bytes, alpha_threshold=1)
    
    # Load ground truth mask
    true_mask = load_mask_as_bool("/path/to/mask.png", threshold=127)
    
    # Calculate metrics
    metrics = binary_mask_metrics(pred_mask, true_mask)
    print(metrics)  # {'mask_iou': 0.95, 'mask_precision': 0.97, ...}
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _require_pillow():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Image metrics require Pillow. Install `pillow` to evaluate image-manipulation tasks."
        ) from exc


_DATA_URL_RE = re.compile(r"^\s*data:(?P<mime>image/[^;]+);base64,(?P<b64>.+)\s*$", re.DOTALL)


def decode_base64_image(payload: str) -> bytes:
    """Decode a base64 image payload.

    Accepts raw base64 or a data URL of the form: data:image/png;base64,<...>.
    Strips common markdown fences and whitespace.
    """
    if payload is None:
        raise ValueError("Empty payload")

    text = str(payload).strip()

    # Strip common markdown fences.
    if text.startswith("```"):
        text = text.strip("`").strip()

    match = _DATA_URL_RE.match(text)
    if match:
        text = match.group("b64")

    # Remove whitespace/newlines.
    text = re.sub(r"\s+", "", text)

    # Fix missing padding.
    missing = (-len(text)) % 4
    if missing:
        text = text + ("=" * missing)

    return base64.b64decode(text, validate=False)


def coerce_prediction_to_image_bytes(prediction: Any) -> bytes:
    """Best-effort conversion of a model prediction into image bytes."""
    if prediction is None:
        raise ValueError("Prediction is None")

    if isinstance(prediction, (bytes, bytearray)):
        return bytes(prediction)

    # Future-proof: allow dict payloads.
    if isinstance(prediction, dict):
        for key in ("image_base64", "png_base64", "base64", "data"):
            if key in prediction and isinstance(prediction[key], str):
                return decode_base64_image(prediction[key])
        raise ValueError("Unsupported dict prediction: missing base64 field")

    # If it's a path to a file on disk.
    if isinstance(prediction, str):
        candidate = prediction.strip()
        if candidate.startswith(("http://", "https://")):
            raise ValueError(
                "Got an image URL output; set provider config `image_response_format: b64_json` "
                "or use a provider that returns base64 artifacts."
            )
        # Try to detect if it's a file path vs base64
        # Base64 is typically very long and contains specific patterns
        is_likely_path = (
            len(candidate) < 500 and  # Paths are usually short
            ("/" in candidate or "\\" in candidate or "." in candidate) and  # Path separators
            not candidate.startswith(("iVBOR", "R0lGO", "/9j/", "AAAA"))  # Common base64 image headers
        )
        
        if is_likely_path:
            try:
                path_obj = Path(candidate)
                if path_obj.exists():
                    return path_obj.read_bytes()
            except (OSError, ValueError):
                # Path invalid, fall through to base64 decoding
                pass
        
        # Assume it's base64
        return decode_base64_image(candidate)

    raise ValueError(f"Unsupported prediction type: {type(prediction).__name__}")


def _to_grayscale_array(image: Any) -> np.ndarray:
    Image = _require_pillow()
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image")
    gray = image.convert("L")
    return np.asarray(gray, dtype=np.uint8)


def load_mask_as_bool(mask_path: str, *, threshold: int = 127) -> np.ndarray:
    """Load a mask image (path) as a boolean array (foreground=True)."""
    Image = _require_pillow()
    with Image.open(mask_path) as img:
        gray = _to_grayscale_array(img)
    return gray > threshold


def predicted_image_to_mask(
    image_bytes: bytes,
    *,
    alpha_threshold: int = 1,
    grayscale_threshold: int = 250,
) -> np.ndarray:
    """Derive a foreground mask from a predicted image.

    Strategy:
    - If the image has an alpha channel, foreground = alpha > alpha_threshold.
    - Otherwise, foreground = grayscale < grayscale_threshold (best-effort fallback).
    """
    Image = _require_pillow()
    with Image.open(BytesIO(image_bytes)) as img:
        mode = (img.mode or "").upper()
        if "A" in mode:  # RGBA / LA / PA
            alpha = np.asarray(img.getchannel("A"), dtype=np.uint8)
            return alpha > alpha_threshold
        gray = _to_grayscale_array(img)
        return gray < grayscale_threshold


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def binary_mask_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Compute standard binary segmentation metrics from boolean arrays."""
    if pred.shape != true.shape:
        raise ValueError(f"Mask shape mismatch: pred={pred.shape} true={true.shape}")

    pred_bool = pred.astype(bool)
    true_bool = true.astype(bool)

    tp = float(np.logical_and(pred_bool, true_bool).sum())
    tn = float(np.logical_and(~pred_bool, ~true_bool).sum())
    fp = float(np.logical_and(pred_bool, ~true_bool).sum())
    fn = float(np.logical_and(~pred_bool, true_bool).sum())

    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    # Mean absolute error over {0,1} mask values.
    mae = float(np.mean(np.abs(pred_bool.astype(np.float32) - true_bool.astype(np.float32))))

    return {
        "mask_iou": iou,
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_f1": f1,
        "mask_accuracy": acc,
        "mask_mae": mae,
    }


@dataclass(frozen=True)
class BackgroundRemovalMetrics:
    """Convenience wrapper for background-removal evaluation."""

    alpha_threshold: int = 1
    grayscale_threshold: int = 250
    mask_threshold: int = 127

    def per_sample(
        self,
        *,
        prediction: Any,
        mask_path: str,
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Compute metrics and return (metrics, error_message)."""
        try:
            image_bytes = coerce_prediction_to_image_bytes(prediction)
            pred_mask = predicted_image_to_mask(
                image_bytes,
                alpha_threshold=self.alpha_threshold,
                grayscale_threshold=self.grayscale_threshold,
            )
            true_mask = load_mask_as_bool(mask_path, threshold=self.mask_threshold)
            resized = 0.0
            if pred_mask.shape != true_mask.shape:
                Image = _require_pillow()
                h, w = true_mask.shape
                img = Image.fromarray((pred_mask.astype(np.uint8) * 255), mode="L")
                img = img.resize((w, h), resample=getattr(Image, "NEAREST", 0))
                pred_mask = (np.asarray(img, dtype=np.uint8) > 127)
                resized = 1.0

            metrics = binary_mask_metrics(pred_mask, true_mask)
            metrics["pred_resized"] = resized
            return metrics, None
        except Exception as exc:
            return {}, str(exc)
