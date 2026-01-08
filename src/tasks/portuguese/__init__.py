"""Portuguese language benchmarks.

This module uses the new handler-based task system.
Tasks are discovered automatically from the .py files in this directory.
"""

from .assin2_rte import Assin2Rte
from .assin2_sts import Assin2Sts
from .bluex import Bluex
from .enem_challenge import EnemChallenge
from .oab_exams import OabExams

__all__ = [
    "Assin2Rte",
    "Assin2Sts",
    "Bluex",
    "EnemChallenge",
    "OabExams",
]
