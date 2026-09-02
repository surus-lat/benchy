"""nb — the four pillars of benchy as data + a thin engine of functions.

A benchmark is DATA: task (input->output), scoring (rule+aggregate), cases.
A system is DATA too: a spec (kind+params) the engine compiles to a callable.
The engine is five functions over that data:

    load(bench_root, path) -> benchmark (dict)
    compile(system_spec) -> invoke(text) -> prediction
    grade(benchmark, invoke) -> artifact (dict with per-case + aggregate)
    run(benchmark, system) -> artifact          # system is the ARGUMENT
    as_loss(benchmark) -> (system) -> float     # the exported loss
"""

from . import engine
from .engine import as_loss, compile, grade, load, run

__all__ = ["engine", "load", "compile", "grade", "run", "as_loss"]