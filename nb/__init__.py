"""nb — the engine. STDLIB ONLY.

Four pillars. Bare metal:
  TASK    the description of the program we search for: in -> out.
  SCORING the grading function: what good means, and the loss.
  DATA    the exam: n cases. A system takes it; we get graded evidence.
  SYSTEM  the exam taker: invoke(input) -> prediction. One method. That is
          the whole AI-API. Backends compile learned programs into it.
"""

from .task import Task
from .scoring import Scoring
from .data import Exam
from .system import compile_system
from .load import load, compile_systems
from .benchmark import Benchmark

__all__ = ["Task", "Scoring", "Exam", "compile_system", "Benchmark"]