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

# Extensions that are already usable as multimodal images
_NATIVE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Extensions that need rendering to an image before sending to vision APIs
_RENDERABLE_EXTS = {".pdf", ".tif", ".tiff", ".heic", ".heif", ".jfif", ".docx"}


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


def needs_rendering(file_path: str) -> bool:
    """Return True if the file needs rendering to an image before use."""
    ext = Path(file_path).suffix.lower()
    return ext in _RENDERABLE_EXTS


def render_document_to_image(
    source_path: str,
    output_path: Optional[str] = None,
    *,
    dpi: int = 200,
    max_pages: int = 1,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Render the first page(s) of a document to a PNG image.

    Supports PDF (via pymupdf), TIFF, HEIC, JFIF (via Pillow).
    Returns the path to the rendered PNG file.

    Args:
        source_path: Path to the source document.
        output_path: Where to write the PNG.  Defaults to
            ``<source_path_without_ext>.png``.
        dpi: Resolution for PDF rendering (default 200).
        max_pages: How many pages to render (default 1 = first page only).
        logger: Optional logger.

    Returns:
        Path to the rendered PNG image.

    Raises:
        ImportError: If pymupdf is not installed (for PDF rendering).
        ValueError: If file format is not supported.
    """
    src = Path(source_path)
    ext = src.suffix.lower()

    if output_path is None:
        output_path = str(src.with_suffix(".png"))

    out = Path(output_path)

    # Skip if already rendered
    if out.exists() and out.stat().st_size > 0:
        return str(out)

    if ext == ".pdf":
        return _render_pdf(source_path, str(out), dpi=dpi, max_pages=max_pages, logger=logger)
    elif ext in {".tif", ".tiff", ".heic", ".heif", ".jfif"}:
        return _render_pillow(source_path, str(out), logger=logger)
    else:
        raise ValueError(
            f"Cannot render '{ext}' files to images. "
            f"Supported: {', '.join(sorted(_RENDERABLE_EXTS))}"
        )


def _render_pdf(
    source_path: str,
    output_path: str,
    *,
    dpi: int = 200,
    max_pages: int = 1,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Render PDF page(s) to PNG using pymupdf."""
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF rendering. Install it with: "
            "pip install pymupdf"
        )

    doc = pymupdf.open(source_path)
    try:
        pages_to_render = min(max_pages, len(doc))
        if pages_to_render == 0:
            raise ValueError(f"PDF has no pages: {source_path}")

        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        if pages_to_render == 1:
            pix = doc[0].get_pixmap(matrix=matrix)
            pix.save(output_path)
        else:
            # Multiple pages → stitch vertically
            from PIL import Image as _Image

            images = []
            for page_idx in range(pages_to_render):
                pix = doc[page_idx].get_pixmap(matrix=matrix)
                img = _Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)

            total_height = sum(im.height for im in images)
            max_width = max(im.width for im in images)
            combined = _Image.new("RGB", (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for im in images:
                combined.paste(im, (0, y_offset))
                y_offset += im.height
            combined.save(output_path, format="PNG")

        if logger:
            logger.debug(
                "Rendered %d page(s) of %s to %s at %d DPI",
                pages_to_render, source_path, output_path, dpi,
            )
    finally:
        doc.close()

    return output_path


def _render_pillow(
    source_path: str,
    output_path: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Convert image formats (TIFF, HEIC, JFIF) to PNG via Pillow."""
    from PIL import Image as _Image

    with _Image.open(source_path) as img:
        # For multi-frame TIFFs, take just the first frame
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)
        rgb = img.convert("RGB") if img.mode not in {"RGB", "L", "RGBA"} else img
        rgb.save(output_path, format="PNG")

    if logger:
        logger.debug("Converted %s to %s via Pillow", source_path, output_path)

    return output_path

