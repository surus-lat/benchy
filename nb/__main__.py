"""`python -m nb <bench_dir> <system>` — run one exam, write the graded artifact.

<system> names an attribute of <bench_dir>/stubs.py — the offline demo takers.
Real takers are any invoked program (an AI-API endpoint, a workflow, ...).
"""
import json
import sys
from pathlib import Path

from .exam import load


def main() -> None:
    bench, name = sys.argv[1], sys.argv[2]
    sys.path.insert(0, str(Path(bench).resolve()))
    artifact = load(bench).run(getattr(__import__("stubs"), name))
    out = Path(bench) / f"artifact_{name}.json"
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"{artifact['benchmark']} {name}: score={artifact['score']:.2f} "
          f"loss={artifact['loss']:.2f} -> {out}")


if __name__ == "__main__":
    main()