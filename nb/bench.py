"""nb — the engine. A benchmark is a directory; this interprets it.

Pillars map 1:1 to files:
  task.json -> Task, scoring.json -> Scoring, cases.jsonl -> Data,
  systems/*.json -> System.

Everything is data. Zero user Python. as_loss() = 1 - score.
"""
import json
from pathlib import Path


def load(bench_dir):
    """Interpret a benchmark directory: task.json + scoring.json + cases.jsonl."""
    d = Path(bench_dir)
    cases = [json.loads(l) for l in
             (d / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "task": json.loads((d / "task.json").read_text(encoding="utf-8")),
        "scoring": json.loads((d / "scoring.json").read_text(encoding="utf-8")),
        "cases": cases,
    }


def load_system(bench_dir, system_name):
    """A system is data too: bench_dir/systems/<name>.json"""
    return json.loads((Path(bench_dir) / "systems" / f"{system_name}.json")
                      .read_text(encoding="utf-8"))


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
    """result = benchmark.run(system) — the vision invariant.

    Scoring is inline: the scoring.json data is simple enough
    (exact match -> points, mean) that a separate grade() was noise.
    Returns the graded artifact: per-case scores + aggregate.
    """
    bench = load(bench_dir)
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
    _r = run(_d, load_system(_d, _s))
    save(_r, _d, _s)
    print(json.dumps(_r, indent=2))