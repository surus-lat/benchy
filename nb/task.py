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
        out_split = out.rsplit('.', 1)
        if len(out_split) == 2 and out_split[1] in ('pos', 'neg', 'bool', 'int', 'float'):
            self.out = out_split[0] + '.' + 'label'
            self.out_enum = frozenset({out_split[0] + '.label', out})
        else:
            self.out = out
            self.out_enum = frozenset({out})
        self.pred_enum = self.out_enum

    def __repr__(self):
        return f"Task(in={self.in_!r}, out={self.out!r})"