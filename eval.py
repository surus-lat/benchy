"""Legacy entrypoint (deprecated).

Use `benchy eval ...` instead.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from src.benchy_cli import main as benchy_main

    argv = list(sys.argv[1:] if argv is None else argv)
    return benchy_main(["eval", *argv])


if __name__ == "__main__":
    raise SystemExit(main())

