"""Dataset handling for the new benchy engine.

Bare metal: functions that build Datasets.
"""

from .jsonl import jsonl_dataset
from .local import local_dataset

__all__ = ["jsonl_dataset", "local_dataset"]
