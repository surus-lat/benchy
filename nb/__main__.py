"""python -m nb <bench_root> /sentiment <artifact.json> <system>

Runnable by a person: loads the benchmark by ontology path, runs the NAMED
system (from bench_root systems.json — no silent default, c10), writes the
graded artifact JSON.
"""

import json
import sys
from pathlib import Path

from . import engine


def main(argv: list[str]) -> int:
    # c20: the two argv-length refusals (was: <3 -> __doc__, <4 -> usage)
    # fused into ONE — the c11 law, two same-kind refusals are one check.
    # c10: no silent default — the system taking the exam is named, always.
    # defaulting to systems[0] was a silent surprise (which stub ran?).
    if len(argv) < 4:
        print("usage: python -m nb <bench_root> <path> <artifact.json> <system>",
              file=sys.stderr)
        return 2
    bench_root, path, artifact_path, system_name = argv[0], argv[1], argv[2], argv[3]

    benchmark = json.loads(
        (Path(bench_root) / (path.strip("/") + ".json")).read_text(encoding="utf-8"))
    # c18, donated from the old spine (.staging/benchy/core.py OntologyPath:
    # "simultaneously the registry key and the on-disk layout"; old
    # load_benchmark resolved benchmarks BY ontology): the requested path
    # and the file's declared path must be the SAME path — a file that
    # declares /other when you asked for /sentiment is a broken exam
    # install (copied/renamed without editing), and running it would
    # silently produce artifacts under the wrong identity. refusal beats
    # surprise; the artifact's benchmark field must be trustworthy.
    if benchmark["path"] != path:
        print(f"benchmark declares {benchmark['path']!r} but was requested as {path!r}",
              file=sys.stderr)
        return 2
    systems = json.loads(
        (Path(bench_root) / "systems.json").read_text(encoding="utf-8"))
    system = next(s for s in systems if s["name"] == system_name)

    artifact = engine.run(benchmark, system)
    Path(artifact_path).write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    # c12: metal (s07 c3 proved CLI metal) — a person with no Python
    # knowledge needs ONE human-readable line: which exam, which system,
    # what score. the artifact file is for programs; this line is for people.
    print(f"{artifact['benchmark']} system={system['name']} score={artifact['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))