"""`python -m nb <bench_root> <ontology_path> <system>` — run one exam.

Benchmarks are addressed by ontology path /<task?>/<domain?>/<language?>;
<bench_root> is the directory tree they live in.  <system> names an attribute
of the benchmark's stubs.py — the offline demo takers.  Real takers are any
invoked program (an AI-API endpoint, a workflow, ...).
"""
import json
import sys

from .exam import locate


def main() -> None:
    root, path, name = sys.argv[1], sys.argv[2], sys.argv[3]
    exam = locate(root, path)
    sys.path.insert(0, str(exam.dir))
    artifact = exam.run(getattr(__import__("stubs"), name))
    out = exam.dir / f"artifact_{name}.json"
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"{artifact['benchmark']} {name}: score={artifact['score']:.2f} "
          f"loss={artifact['loss']:.2f} -> {out}")


if __name__ == "__main__":
    main()