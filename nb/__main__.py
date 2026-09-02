# nb/__main__.py — offline end-to-end: `python -m nb <root> /ontology/path out.json`
# CLI is metal only if the bar is read as "runnable without pytest". kept
# minimal: two args, writes the artifact, prints the score.
import json
import sys
from pathlib import Path

from .engine import locate, run

USAGE = "usage: python -m nb <bench_root> <ontology_path> <artifact.json>"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(USAGE, file=sys.stderr)
        return 2
    root, path, out = argv
    exam = locate(root, path)
    artifact = {"path": exam["path"], "systems": {}}
    for name, spec in exam["systems"].items():
        artifact["systems"][name] = run(exam, spec)
    Path(out).write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    for name, rep in artifact["systems"].items():
        print(f"{exam['path']} {name} score={rep['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())