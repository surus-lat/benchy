"""Probe test assets and constants."""

from pathlib import Path
from typing import Dict, Any

# Inline minimal test schema
MINIMAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "string"}
    },
    "required": ["ok"],
    "additionalProperties": False
}

# Prompt constants
PROMPT_BASIC_CHAT = "Reply with OK."
PROMPT_JSON_TEST = "Return JSON with field 'ok' and value 'yes'."
PROMPT_MULTIMODAL_TEST = "Reply with OK if you can see an image."
PROMPT_LOGPROBS_TEST = "Choose A or B. Answer with a single letter."


def get_test_image_path() -> Path:
    """Get path to deterministic JPEG test image, generating if needed.
    
    Returns:
        Path to test_image.jpg in assets directory
    """
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    image_path = assets_dir / "test_image.jpg"
    if not image_path.exists():
        _generate_test_image(image_path)
    
    return image_path


def _generate_test_image(path: Path) -> None:
    """Generate deterministic 100x100 JPEG (not PNG) for multimodal testing.
    
    Uses JPEG because:
    - More realistic (most document/image tasks use JPEG)
    - Tests MIME type handling (image/jpeg vs image/png)
    - Some stacks behave differently with JPEG vs PNG
    
    Args:
        path: Path where to save the generated image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create deterministic image (blue square with white text "TEST")
        img = Image.new('RGB', (100, 100), color='#4169E1')  # Royal blue
        draw = ImageDraw.Draw(img)
        
        # Add text (use default font if custom font not available)
        try:
            # Try to use a larger font if available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Center text
        text = "TEST"
        # Get text bounding box for centering
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)
        
        x = (100 - text_width) // 2
        y = (100 - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font)
        
        # Save as JPEG with consistent quality
        img.save(path, 'JPEG', quality=85)  # ~5KB
        
    except ImportError:
        # If PIL not available, create a minimal valid JPEG
        # This is a fallback - PIL should be available in benchy environment
        raise ImportError(
            "PIL (Pillow) is required for probe image generation. "
            "Install with: pip install Pillow"
        )
