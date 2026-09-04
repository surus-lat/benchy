"""Dataset handling for the new benchy engine.

Bare metal: functions that build Datasets.
"""

from .dataset import Data
from .jsonl import jsonl_dataset
from .loader import load
from .local import local_dataset

__all__ = ["Data", "jsonl_dataset", "local_dataset", "load"]