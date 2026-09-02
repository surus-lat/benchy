"""nb — the engine. A benchmark is a directory; this interprets it.
Files: task.json, scoring.json, cases.jsonl, systems/*.json (schema with
examples in DESIGN.md). Everything is data; zero user Python."""
import json
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
    """The ONLY way a system takes the exam. system is a data dict."""
    kind = system["kind"]
    if kind == "constant":
        return system["out"]
    if kind == "keyword":
        hit = any(k in case["input"] for k in system["if_contains"])
        return system["then"] if hit else system["else"]
    raise ValueError(f"unknown system kind: {kind}")


def run(bench_dir, system):
    """result = benchmark.run(system). system: NAME (str) or data dict.
    Returns the graded artifact: per-case scores + aggregate."""
    bench = load(bench_dir, system if isinstance(system, str) else None)
    if isinstance(system, str):
        system = bench["system"]
    scoring = bench["scoring"]
    if scoring["match"] != "exact":
        raise ValueError(f"unknown match: {scoring['match']}")
    points = scoring["points"]
    cases = []
    for c in bench["cases"]:
        predicted = invoke(system, c)
        cases.append({"input": c["input"], "expected": c["expected"],
                      "predicted": predicted,
                      "score": points if predicted == c["expected"] else 0})
    score = sum(c["score"] for c in cases) / len(cases)
    return {"benchmark": str(bench_dir), "task": bench["task"]["task"],
            "cases": cases, "score": score}


def as_loss(result):
    """loss = benchmark.as_loss() — the identity. 1 - score."""
    return 1 - result["score"]


def save(result, bench_dir, system_name, out_dir="runs"):
    out = Path(out_dir) / Path(bench_dir).name / f"{system_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python3 nb/bench.py <bench_dir> <system>", file=sys.stderr)
        sys.exit(2)
    _d, _s = sys.argv[1], sys.argv[2]
    _r = run(_d, _s)
    save(_r, _d, _s)
    print(json.dumps(_r, indent=2))