# nb/__main__.py — offline end-to-end: `python -m nb <root> /ontology/path out.json`
# cycle 3 attempted deletion: without this, the engine runs only under pytest
# — the bar says "runs offline, end to end" and a non-engineer must be able to
# take the exam without writing a test. CLI = metal. 3 concepts only.
import json
import sys
from pathlib import Path

from .engine import locate, run


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("usage: python -m nb <bench_root> <ontology_path> <artifact.json>", file=sys.stderr)
        return 2
    root, path, out = argv
    exam = locate(root, path)
    # cycle 15 tried to delete the path echo (cycle 11 deleted run's echoes:
    # "callers hold their own args") and restored: c11's law has TWO tiers.
    # a RETURN VALUE leans on its caller; a FILE ON DISK outlives the
    # invocation — whoever opens the artifact later holds no argv and no
    # stdout. without path, evidence from two exams is indistinguishable.
    # a file must interpret alone: identity is metal.
    artifact = {"path": exam["path"], "systems": {}}
    for name, spec in exam["systems"].items():
        artifact["systems"][name] = run(exam, spec)
    Path(out).write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    for name, rep in artifact["systems"].items():
        print(f"{exam['path']} {name} score={rep['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())