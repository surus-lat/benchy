"""Benchy — a semantic language and execution engine for benchmarking AI programs.

    B = (P, S, D)      a benchmark is a program, a scoring function, and a dataset
    R = (B, AI)        a run binds a benchmark to an AI-system

A benchmark is separate from the AI-system taking it. The engine knows one runtime
contract — a named-field input object in, a named-field output object out — and
everything about how a particular AI-system is invoked lives in an adapter on the
far side of that line.

    YAML -> parse -> validate -> compile -> JSON IR -> engine -> adapter -> AI-system

Eight modules, each with one job:

    errors      the one diagnostic shape
    types       semantic types: schemas, validation, equality
    ontology    /task/domain/language, and P ∈ P_T
    compiler    YAML -> canonical JSON IR
    data        the exam: streaming JSONL inside a workspace
    score       field correctness -> instance score -> benchmark score
    adapter     the runtime boundary
    run         the engine loop

The only dependency is PyYAML.
"""

from __future__ import annotations

from benchy.adapter import bind, invoker, resolve
from benchy.compiler import compile_benchmark, parse
from benchy.errors import BenchyError
from benchy.run import run

__all__ = ["BenchyError", "bind", "compile_benchmark", "invoker", "parse", "resolve", "run"]

__version__ = "1.0"
