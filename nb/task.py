"""TASK — the description of the program we are searching for.

A Task is an input/output type declaration: what the program takes and
what it must return. Nothing else. If it grows a method that runs
things, that method is SYSTEM's job; if it grows data, that is DATA's
job.
"""


class Task:
    """in: type name of the input. out: type name of the output."""

    def __init__(self, in_: str, out: str):
        self.in_ = in_
        self.out = out