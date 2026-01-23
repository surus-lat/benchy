"""SURUS interface implementations."""

from .surus_interface import SurusInterface
from .surus_ocr_interface import SurusOCRInterface
from .surus_factura_interface import SurusFacturaInterface
from .surus_classify_interface import SurusClassifyInterface

__all__ = [
    "SurusInterface",
    "SurusOCRInterface",
    "SurusFacturaInterface",
    "SurusClassifyInterface",
]
