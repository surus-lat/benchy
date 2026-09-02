"""nb — the engine. STDLIB ONLY.

Four pillars. Bare metal:
  TASK    the description of the program we search for: in -> out (data).
  SCORING the grading function: what good means, and the loss (data + score).
  DATA    the exam: n (input, expected) cases (data). A system takes it;
          we get graded evidence.
  SYSTEM  the exam taker: invoke(input) -> prediction. One method. That is
          the whole AI-API. Backends compile learned programs into it.
"""

from .scoring import score
from .system import compile_system
from .load import load
from .benchmark import Benchmark

__all__ = ["score", "compile_system", "Benchmark"]