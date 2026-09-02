"""nb — the four pillars of benchy as data + a pure engine of functions.

A benchmark is DATA: task (input->output), scoring (rule+aggregate), cases.
A system is DATA too: a spec (kind+params) the engine compiles to a callable.
The engine is four pure functions over that data:

    compile(system_spec) -> invoke(text) -> prediction
    grade(benchmark, invoke) -> artifact (dict with per-case + aggregate)
    run(benchmark, system) -> artifact          # system is the ARGUMENT
    as_loss(benchmark) -> (system) -> float     # the exported loss

Files are the persistence layer, owned by the CLI (nb/__main__.py).
"""

from . import engine
from .engine import as_loss, compile, grade, run

__all__ = ["engine", "compile", "grade", "run", "as_loss"]