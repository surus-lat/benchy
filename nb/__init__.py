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


def locate(path, root="bench"):
    """Resolve an ontology path like '/sentiment' to its exam directory.
    # survival: the tree IS the ontology — bench/sentiment/ literally is
    # /sentiment; no walk, no index file, no second address."""
    d = Path(root) / path.strip("/").replace("/", "-")
    if not (d / "exam.json").is_file():
        raise FileNotFoundError(f"no exam at {path!r} under {root}/")
    return d


def _score(want, got, weights):
    if isinstance(want, dict):
        # c10: the weighted branch is honest — weights are not decoration,
        # they RANK: right-on-critical beats right-on-nice at equal field
        # count. `weights or {}` means an unweighted dict-want scores as the
        # per-field mean (every field weight 1) — one shape, no special case.
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
        # survival (BARE_METAL c9): deleting flaky broke 4 runner tests
        # instantly — a deterministic scriptable failure is the ONLY honest
        # probe for retries/loud-errors/kill-resume; without a failing system
        # the retry path is claimed-but-untested. `fails` = attempts that
        # fail before the first success; always-fail = fails >= tries (data,
        # no special encoding). c9 deleted the "FP" script mini-language:
        # periodic flake semantics no test exercised (fails-count covers all
        # scripted shapes the bar demands). A second `of` system wraps.
        inner, fails, delay = _compile(spec["of"]), spec["fails"], spec.get("sleep", 0)
        seen = {}

        def invoke(text):
            n = seen.get(text, 0) + 1
            seen[text] = n
            if delay:
                sleep(delay)
            if n <= fails:
                raise RuntimeError(f"flaky: attempt {n} (first {fails} fail)")
            return inner(text)

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

    def run(self, system, out=None, workers=8, tries=3):
        """Take the exam concurrently (serial fails the 1000-case bar 16x over);
        `out` re-run = resume: ok cases kept, errored cases re-attempted."""
        spec = json.loads(Path(system).read_text()) if isinstance(system, (str, Path)) else system
        # total = the exam size, fixed: score = sum/total makes the mid-run
        # value an honest lower bound (ungraded cases count 0), and the final
        # value the exact mean. A partial artifact still interprets alone.
        art = {"system": spec, "scoring": self.spec.get("scoring"),
               "total": len(self.cases), "cases": []}
        out = Path(out) if out else None
        if out and out.exists():
            old = json.loads(out.read_text())
            cur = {c["id"]: c for c in self.cases}
            stale = [r for r in old.get("cases", [])
                     if (cur.get(r["id"]) or {}).get("input") != r.get("input")
                     or (cur.get(r["id"]) or {}).get("want") != r.get("want")]
            if stale or (old.get("system"), old.get("scoring")) != (
                    spec, self.spec.get("scoring")):
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
                # score = sum/total: ungraded cases count 0, so the mid-run
                # value is a lower bound that converges to the exact mean.
                art["score"] = sum(r["score"] for r in art["cases"]) / art["total"]
                if out:
                    _write(out, art)
        # a resume that found nothing to do never enters the loop — the
        # artifact still owes its reader (and as_loss) the aggregate.
        # errors is NOT stored: it is a projection, sum(status == "error").
        art["score"] = sum(r["score"] for r in art["cases"]) / art["total"]
        return art

    def as_loss(self, system, **kw):
        """The benchmark as a loss over systems: loss = 1 - exam score."""
        return 1.0 - self.run(system, **kw)["score"]