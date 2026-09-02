"""nb — benchy, bare metal, runner-first.

An Exam is task + scoring + cases, locatable by its ontology path. A system
(a JSON spec, data) TAKES the exam: the runner fans cases out over threads,
retries failures, and rewrites the whole artifact after every completed case
— so a kill loses nothing and the artifact IS the resume contract.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

WORKERS, TRIES = 8, 3


def locate(path, root="bench"):
    """Resolve an ontology path like '/sentiment' to its exam directory."""
    hits = [p.parent for p in Path(root).rglob("exam.json")
            if json.loads(p.read_text()).get("path") == path]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one exam at {path!r} under {root}/, found {len(hits)}")
    return hits[0]


def _score(want, got, weights):
    """Shape-dispatch on want: scalar = exact match, dict = weighted per-field."""
    if isinstance(want, dict):
        got = got if isinstance(got, dict) else {}
        w = {k: (weights or {}).get(k, 1.0) for k in want}
        return sum(v for k, v in w.items() if got.get(k) == want[k]) / sum(w.values())
    return 1.0 if got == want else 0.0


def _compile(spec):
    """System spec (data) -> invoke(input) -> prediction. The compiler pillar."""
    kind = spec["kind"]
    if kind == "always":
        return lambda text: spec["value"]
    if kind == "keyword":
        return lambda text: ("pos" if any(k in str(text).lower() for k in spec["pos"])
                             else "neg")
    if kind == "flaky":
        inner, script, delay = _compile(spec["of"]), spec["script"], spec.get("sleep", 0)
        seen = {}

        def invoke(text):
            n = seen.get(text, 0) + 1
            seen[text] = n
            if delay:
                sleep(delay)
            if script[(n - 1) % len(script)] == "F":
                raise RuntimeError(f"flaky script {script!r}: attempt {n} fails")
            return inner(text)

        invoke.seen = {}
        return invoke
    raise ValueError(f"unknown system kind {kind!r}")


def _attempt(invoke, case, weights, tries):
    """One case: invoke with up to `tries` attempts; returns graded evidence."""
    rec = {"id": case["id"], "input": case["input"], "want": case["want"]}
    err = None
    for t in range(1, tries + 1):
        try:
            got = invoke(case["input"])
            return rec | {"got": got, "score": _score(case["want"], got, weights),
                          "status": "ok", "tries": t}
        except Exception as e:
            err = e
    return rec | {"got": None, "score": 0.0, "status": "error", "tries": tries,
                  "error": f"{type(err).__name__}: {err}"}


def _write(out, art):
    """Atomic artifact write: temp + rename — a kill never leaves torn JSON."""
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(art))
    os.replace(tmp, out)


class Exam:
    """The benchmark: task + scoring + cases. run(system) takes it, as_loss ranks systems."""

    def __init__(self, dir):
        self.dir = Path(dir)
        self.spec = json.loads((self.dir / "exam.json").read_text())
        self.path = self.spec["path"]
        self.cases = self.spec["cases"]
        self.weights = (self.spec.get("scoring") or {}).get("weights")
        outs = self.spec.get("out")
        for c in self.cases:
            if not outs:
                continue
            bad = [v for v in (c["want"].values() if isinstance(c["want"], dict)
                               else [c["want"]]) if v not in outs]
            if bad:
                raise ValueError(
                    f"case {c['id']!r}: want {c['want']!r} outside declared out {outs}")

    def run(self, system, out=None, workers=WORKERS, tries=TRIES):
        """Take the exam concurrently (serial fails the 1000-case bar 16x over);
        `out` re-run = resume: ok cases kept, errored cases re-attempted."""
        spec = json.loads(Path(system).read_text()) if isinstance(system, (str, Path)) else system
        art = {"exam": self.path, "system": spec, "scoring": self.spec.get("scoring"),
               "total": len(self.cases), "cases": []}
        out = Path(out) if out else None
        if out and out.exists():
            old = json.loads(out.read_text())
            cur = {c["id"]: c for c in self.cases}
            stale = [r for r in old.get("cases", [])
                     if (cur.get(r["id"]) or {}).get("input") != r.get("input")
                     or (cur.get(r["id"]) or {}).get("want") != r.get("want")]
            if stale or (old.get("exam"), old.get("system"), old.get("scoring")) != (
                    self.path, spec, self.spec.get("scoring")):
                raise ValueError(f"{out} belongs to a different exam/system — resume must match")
            art["cases"] = [r for r in old["cases"] if r.get("status") == "ok"]
        done = {r["id"]: r for r in art["cases"]}
        todo = [c for c in self.cases if c["id"] not in done]
        invoke = _compile(spec)
        with ThreadPoolExecutor(workers) as pool:
            futs = [pool.submit(_attempt, invoke, c, self.weights, tries) for c in todo]
            for f in as_completed(futs):
                rec = f.result()
                done[rec["id"]] = rec
                art["cases"] = list(done.values())
                art["score"] = sum(r["score"] for r in art["cases"]) / art["total"]
                art["errors"] = sum(r["status"] == "error" for r in art["cases"])
                if out:
                    _write(out, art)
        return art

    def as_loss(self, system, workers=WORKERS, tries=TRIES):
        """The benchmark as a loss over systems: loss = 1 - exam score."""
        return 1.0 - self.run(system, workers=workers, tries=tries)["score"]