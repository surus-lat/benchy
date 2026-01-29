"""Visualization utilities for image-based tasks.

This module provides functions to visualize predictions compared to ground truth,
particularly useful for segmentation and background removal tasks.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _require_pillow():
    """Import PIL/Pillow with helpful error message."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "Visualization requires Pillow. Install with: pip install pillow"
        ) from exc


def create_mask_comparison(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    source_image_path: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> "Image.Image":
    """Create a visual comparison of two masks.
    
    Args:
        pred_mask: Predicted mask (boolean array, True=foreground)
        true_mask: Ground truth mask (boolean array, True=foreground)
        source_image_path: Optional path to source image to overlay on
        metrics: Optional metrics dictionary to display
        
    Returns:
        PIL Image with visualization
    """
    Image, ImageDraw, ImageFont = _require_pillow()
    
    h, w = true_mask.shape
    
    # Load and prepare base image
    if source_image_path and Path(source_image_path).exists():
        try:
            base = Image.open(source_image_path).resize((w, h), Image.Resampling.LANCZOS).convert('RGBA')
            # Darken for better overlay visibility
            base = Image.blend(
                Image.new('RGBA', base.size, (0, 0, 0, 255)),
                base,
                alpha=0.6
            )
        except Exception as e:
            logger.warning(f"Could not load source image {source_image_path}: {e}")
            base = Image.new('RGBA', (w, h), (255, 255, 255, 255))
    else:
        base = Image.new('RGBA', (w, h), (255, 255, 255, 255))
    
    # Create color-coded overlay
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    overlay_data = np.array(overlay)
    
    # True Positive (both correct: foreground) - Green
    tp_mask = np.logical_and(true_mask, pred_mask)
    overlay_data[tp_mask] = [0, 255, 0, 180]
    
    # False Positive (predicted foreground, actually background) - Red
    fp_mask = np.logical_and(pred_mask, ~true_mask)
    overlay_data[fp_mask] = [255, 0, 0, 200]
    
    # False Negative (predicted background, actually foreground) - Blue
    fn_mask = np.logical_and(~pred_mask, true_mask)
    overlay_data[fn_mask] = [0, 100, 255, 200]
    
    overlay = Image.fromarray(overlay_data)
    result = Image.alpha_composite(base, overlay)
    
    # Add legend and metrics
    if metrics:
        result = _add_metrics_legend(result, metrics, Image, ImageDraw, ImageFont)
    
    return result.convert('RGB')


def _add_metrics_legend(img, metrics: dict, Image, ImageDraw, ImageFont):
    """Add a legend and metrics text to the image."""
    legend_height = 140
    new_height = img.height + legend_height
    result = Image.new('RGB', (img.width, new_height), (255, 255, 255))
    result.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(result)
    
    # Try to use a nice font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    y_offset = img.height + 10
    x_margin = 10
    
    # Legend
    draw.text((x_margin, y_offset), "Legend:", fill=(0, 0, 0), font=font_large)
    y_offset += 25
    
    legend_items = [
        ("█ Correct (TP)", (0, 200, 0)),
        ("█ False Positive", (255, 0, 0)),
        ("█ False Negative", (0, 100, 255)),
    ]
    
    for text, color in legend_items:
        draw.text((x_margin, y_offset), text, fill=color, font=font_small)
        y_offset += 22
    
    # Metrics (on the right side)
    metrics_x = img.width // 2 + 50
    y_offset = img.height + 10
    
    draw.text((metrics_x, y_offset), "Metrics:", fill=(0, 0, 0), font=font_large)
    y_offset += 25
    
    metric_items = [
        ("IoU", metrics.get('mask_iou', 0)),
        ("F1", metrics.get('mask_f1', 0)),
        ("Precision", metrics.get('mask_precision', 0)),
        ("Recall", metrics.get('mask_recall', 0)),
    ]
    
    for name, value in metric_items:
        text = f"{name}: {value:.3f}"
        draw.text((metrics_x, y_offset), text, fill=(0, 0, 0), font=font_small)
        y_offset += 22
    
    return result


def save_mask_comparison_for_sample(
    prediction_bytes: bytes,
    ground_truth_path: str,
    source_image_path: Optional[str],
    output_path: Path,
    metrics: Optional[dict] = None,
) -> bool:
    """Generate and save a comparison visualization for a sample.
    
    Args:
        prediction_bytes: Predicted image as bytes (PNG with alpha or grayscale)
        ground_truth_path: Path to ground truth mask
        source_image_path: Optional path to source image
        output_path: Where to save the comparison image
        metrics: Optional metrics to display
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .image_metrics import (
            predicted_image_to_mask,
            load_mask_as_bool,
        )
        
        # Convert prediction to mask
        pred_mask = predicted_image_to_mask(prediction_bytes, alpha_threshold=1)
        
        # Load ground truth mask
        true_mask = load_mask_as_bool(ground_truth_path, threshold=127)
        
        # Resize if needed
        if pred_mask.shape != true_mask.shape:
            Image, _, _ = _require_pillow()
            h, w = true_mask.shape
            pred_img = Image.fromarray((pred_mask.astype(np.uint8) * 255), mode='L')
            pred_img = pred_img.resize((w, h), Image.Resampling.NEAREST)
            pred_mask = (np.array(pred_img, dtype=np.uint8) > 127)
        
        # Create comparison
        comparison = create_mask_comparison(
            pred_mask=pred_mask,
            true_mask=true_mask,
            source_image_path=source_image_path,
            metrics=metrics,
        )
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(output_path)
        logger.debug(f"Saved comparison image to {output_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Could not generate comparison visualization: {e}")
        return False
