"""python -m nb <bench_root> /sentiment <artifact.json> [system]

Runnable by a person: loads the benchmark by ontology path, runs the named
system (from bench_root systems.json) or defaults to the first one, writes
the graded artifact JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import engine


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    bench_root, path, artifact_path = argv[0], argv[1], argv[2]
    system_name = argv[3] if len(argv) > 3 else None

    benchmark = json.loads(
        (Path(bench_root) / (path.strip("/") + ".json")).read_text(encoding="utf-8"))
    systems = json.loads(
        (Path(bench_root) / "systems.json").read_text(encoding="utf-8"))
    if system_name is None:
        system = systems[0]
    else:
        system = next(s for s in systems if s["name"] == system_name)

    artifact = engine.run(benchmark, system)
    Path(artifact_path).write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"{artifact['benchmark']} system={system['name']} score={artifact['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))