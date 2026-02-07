import base64
import io

from src.interfaces.common.image_preprocessing import (
    coerce_positive_int,
    encode_image_base64,
    image_media_type,
    load_pil_image,
)


def _create_image(path, size=(300, 100), mode="RGB"):
    from PIL import Image

    image = Image.new(mode, size=size, color=(255, 0, 0))
    image.save(path, format="PNG")


def test_coerce_positive_int() -> None:
    assert coerce_positive_int("12", option_name="x") == 12
    assert coerce_positive_int(0, option_name="x") is None
    assert coerce_positive_int("bad", option_name="x") is None


def test_image_media_type_fallback() -> None:
    assert image_media_type("foo.png") == "image/png"
    assert image_media_type("foo.unknown") == "image/jpeg"


def test_encode_image_base64_without_resize(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    _create_image(image_path, size=(40, 20))

    payload, media_type = encode_image_base64(str(image_path))
    decoded = base64.b64decode(payload)

    assert media_type == "image/png"
    assert len(decoded) > 0


def test_encode_image_base64_resizes_to_max_edge(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "large.png"
    _create_image(image_path, size=(300, 100))

    payload, media_type = encode_image_base64(str(image_path), max_edge=120)
    decoded = base64.b64decode(payload)

    with Image.open(io.BytesIO(decoded)) as image:
        assert max(image.size) == 120

    assert media_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}


def test_load_pil_image_resizes(tmp_path) -> None:
    image_path = tmp_path / "pil_resize.png"
    _create_image(image_path, size=(500, 250))

    loaded = load_pil_image(str(image_path), max_edge=200)
    assert max(loaded.size) == 200
