"""nb — the engine. A benchmark is a directory; this interprets it.

Pillars map 1:1 to files:
  task.json -> Task, scoring.json -> Scoring, cases.jsonl -> Data,
  systems/*.json -> System.

Everything is data. Zero user Python. as_loss() = 1 - score.
"""
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load(bench_dir):
    """Interpret a benchmark directory. bench_dir is a Path or str."""
    d = Path(bench_dir)
    return {
        "task": _read_json(d / "task.json"),
        "scoring": _read_json(d / "scoring.json"),
        "cases": _read_jsonl(d / "cases.jsonl"),
    }


def load_system(bench_dir, system_name):
    """A system is data too: bench_dir/systems/<name>.json"""
    return _read_json(Path(bench_dir) / "systems" / f"{system_name}.json")


def invoke(system, case):
    """The ONLY way a system takes the exam. system is a data dict."""
    kind = system["kind"]
    if kind == "constant":
        return system["out"]
    if kind == "keyword":
        text = case["input"]
        hit = any(k in text for k in system["if_contains"])
        return system["then"] if hit else system["else"]
    raise ValueError(f"unknown system kind: {kind}")


def grade(scoring, prediction, expected):
    """Score one case. scoring is data: match/points/aggregate."""
    if scoring["match"] != "exact":
        raise ValueError(f"unknown match: {scoring['match']}")
    return scoring["points"] if prediction == expected else 0


def run(bench_dir, system):
    """result = benchmark.run(system) — the vision invariant.

    Returns the graded artifact: per-case scores + aggregate.
    """
    bench = load(bench_dir)
    cases = [
        {"input": c["input"], "expected": c["expected"],
         "predicted": invoke(system, c), "score": None}
        for c in bench["cases"]
    ]
    for c in cases:
        c["score"] = grade(bench["scoring"], c["predicted"], c["expected"])
    score = sum(c["score"] for c in cases) / len(cases)
    return {"benchmark": str(bench_dir), "task": bench["task"]["task"],
            "cases": cases, "score": score}


def as_loss(result):
    """loss = benchmark.as_loss() — the identity. 1 - score."""
    return 1 - result["score"]


def save(result, bench_dir, system_name, out_dir="runs"):
    bench_name = Path(bench_dir).name
    name = f"{system_name}.json"
    out = Path(out_dir) / bench_name / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python3 nb/bench.py <bench_dir> <system>", file=sys.stderr)
        sys.exit(2)
    _d, _s = sys.argv[1], sys.argv[2]
    _r = run(_d, load_system(_d, _s))
    save(_r, _d, _s)
    print(json.dumps(_r, indent=2))