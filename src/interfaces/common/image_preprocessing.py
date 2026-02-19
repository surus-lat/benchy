"""Shared image preprocessing helpers for interfaces.

These utilities are intentionally interface-layer only:
- tasks define sample content/prompts
- interfaces decide how to package images for providers
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple


_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def coerce_positive_int(
    value: object,
    *,
    option_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """Parse a positive int option, returning None for unset/invalid."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        if logger:
            logger.warning("Invalid integer value for %s: %r", option_name, value)
        return None
    if parsed <= 0:
        if logger:
            logger.warning("%s must be > 0, got %s", option_name, parsed)
        return None
    return parsed


def image_media_type(image_path: str) -> str:
    """Best-effort media type from filename extension."""
    return _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")


def _read_base64_raw(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_base64(
    image_path: str,
    *,
    max_edge: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, str]:
    """Encode image as base64, optionally resizing in memory.

    Returns:
      (base64_payload, media_type)
    """
    media_type = image_media_type(image_path)
    parsed_max = coerce_positive_int(
        max_edge,
        option_name="image_max_edge",
        logger=logger,
    )
    if parsed_max is None:
        return _read_base64_raw(image_path), media_type

    try:
        from PIL import Image
    except ImportError:
        if logger:
            logger.warning(
                "image_max_edge is set but Pillow is not installed; sending original image bytes"
            )
        return _read_base64_raw(image_path), media_type

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            longest_edge = max(width, height)
            if longest_edge <= parsed_max:
                return _read_base64_raw(image_path), media_type

            scale = parsed_max / float(longest_edge)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            resized = image.resize((new_width, new_height), resample=resample)

            save_format = (image.format or "").upper()
            if save_format == "JPG":
                save_format = "JPEG"
            if save_format not in {"JPEG", "PNG", "WEBP", "GIF"}:
                save_format = "JPEG"

            if save_format == "JPEG" and resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")

            output = io.BytesIO()
            if save_format == "JPEG":
                resized.save(output, format=save_format, quality=90, optimize=True)
                out_media_type = "image/jpeg"
            elif save_format == "PNG":
                resized.save(output, format=save_format, optimize=True)
                out_media_type = "image/png"
            elif save_format == "WEBP":
                resized.save(output, format=save_format, quality=90)
                out_media_type = "image/webp"
            else:
                resized.save(output, format=save_format)
                out_media_type = "image/gif"

            encoded = base64.b64encode(output.getvalue()).decode("utf-8")
            if logger:
                logger.debug(
                    "Resized image %s from %sx%s to %sx%s (max edge %s)",
                    image_path,
                    width,
                    height,
                    new_width,
                    new_height,
                    parsed_max,
                )
            return encoded, out_media_type
    except Exception as exc:
        if logger:
            logger.warning(
                "Failed to resize image %s: %s. Sending original bytes.",
                image_path,
                exc,
            )
        return _read_base64_raw(image_path), media_type


def encode_image_data_url(
    image_path: str,
    *,
    max_edge: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Encode image as a data URL, optionally resizing in memory."""
    payload, media_type = encode_image_base64(
        image_path,
        max_edge=max_edge,
        logger=logger,
    )
    return f"data:{media_type};base64,{payload}"


def load_pil_image(
    image_path: str,
    *,
    max_edge: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    """Load an image as PIL.Image, optionally resizing in memory.

    Returns a detached PIL image object (safe after file is closed).
    """
    from PIL import Image

    parsed_max = coerce_positive_int(
        max_edge,
        option_name="image_max_edge",
        logger=logger,
    )

    with Image.open(image_path) as image:
        loaded = image.copy()
        if parsed_max is None:
            return loaded

        width, height = loaded.size
        longest_edge = max(width, height)
        if longest_edge <= parsed_max:
            return loaded

        scale = parsed_max / float(longest_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        resized = loaded.resize((new_width, new_height), resample=resample)

        if logger:
            logger.debug(
                "Resized PIL image %s from %sx%s to %sx%s (max edge %s)",
                image_path,
                width,
                height,
                new_width,
                new_height,
                parsed_max,
            )
        return resized

