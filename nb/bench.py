"""nb — the engine. A benchmark is a directory; this interprets it.
Files: task.json, scoring.json, cases.jsonl, systems/*.json (schema with
examples in DESIGN.md). Everything is data; zero user Python."""
import json
import re
from pathlib import Path


def load(bench_dir, system=None):
    """Interpret a benchmark directory; optional system is a NAME,
    read from bench_dir/systems/<name>.json."""
    d = Path(bench_dir)
    b = {
        "task": json.loads((d / "task.json").read_text(encoding="utf-8")),
        "scoring": json.loads((d / "scoring.json").read_text(encoding="utf-8")),
        "cases": [json.loads(l) for l in
                  (d / "cases.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip()],
    }
    if system is not None:
        b["system"] = json.loads(
            (d / "systems" / f"{system}.json").read_text(encoding="utf-8"))
    return b


def invoke(system, case):
    """The ONLY way a system takes the exam. system is a data dict.
    regex has two shapes, dispatched loudly: extraction (fields: pattern
    per field, group 1 = value, miss = default) and classification
    (if: one pattern, then/else). The keyword kind was subsumed: a
    keyword list IS an alternation pattern (cycle 14)."""
    kind = system["kind"]
    if kind == "constant":
        return system["out"]
    if kind == "regex":
        if "fields" in system:
            return {f: m.group(1) if (m := re.search(p, case["input"]))
                    else system["miss"] for f, p in system["fields"].items()}
        if "if" in system:
            return (system["then"]
                    if re.search(system["if"], case["input"]) else system["else"])
        raise ValueError(f"regex system needs 'fields' or 'if': {sorted(system)}")
    raise ValueError(f"unknown system kind: {kind}")


def run(bench_dir, system):
    """result = benchmark.run(system). system: NAME (str) or data dict.
    Returns the graded artifact: per-case scores + aggregate."""
    bench = load(bench_dir, system if isinstance(system, str) else None)
    if isinstance(system, str):
        system = bench["system"]
    scoring = bench["scoring"]
    cases = []
    for c in bench["cases"]:
        p = invoke(system, c)
        if scoring == {"match": "exact"}:
            s = 1.0 if p == c["expected"] else 0.0
        else:
            w = scoring.get("weights")
            loud = (scoring.get("match") == "fields"
                    and set(scoring) == {"match", "weights"}
                    and isinstance(w, dict)
                    and w and set(w) == set(c["expected"]))
            if not loud:
                raise ValueError(f"unsupported scoring: {scoring!r}")
            s = sum(v for f, v in w.items()
                    if p.get(f) == c["expected"][f]) / sum(w.values())
        cases.append({**c, "predicted": p, "score": s})
    score = sum(c["score"] for c in cases) / len(cases)
    return {"task": bench["task"]["task"],
            "cases": cases, "score": score}


def as_loss(result):
    """loss = benchmark.as_loss() — the identity. 1 - score."""
    return 1 - result["score"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python3 nb/bench.py <bench_dir> <system>", file=sys.stderr)
        sys.exit(2)
    _d, _s = sys.argv[1], sys.argv[2]
    _r = run(_d, _s)
    print(json.dumps(_r, indent=2))