# nb/engine.py — the exam engine.
# s07 datacentric: the Sample record (id, input, expected, context?) is the
# root record. a benchmark is ONE data file (exam.json) plus two tiny pure
# functions: grade (the scoring lens) and run (take the exam). the system is
# a spec in data, interpreted by invoke — the compiler pillar (cloud specs
# later, same slot). the declared lenses are data, not inference: the answer
# space cannot be derived from observed expecteds (a typo'd expected would
# silently become a new class; an unrepresented class would shrink the space).
# explicit beats implicit: unknown keys raise, nothing is inferred silently.

import json
from pathlib import Path

EXAM_KEYS = {"path", "task", "scoring", "samples", "systems"}
SCORE_KEYS = {"match"}
SAMPLE_REQUIRED = {"id", "input", "expected"}
SAMPLE_ALLOWED = SAMPLE_REQUIRED | {"context"}
KINDS = {"const": {"kind", "value"}, "keyword": {"kind", "any", "then", "else"}}


def _check(allowed, got, where, required=None):
    # loud: unknown keys raise, missing required keys raise. no silent defaults.
    req = required if required is not None else allowed
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {unknown}")
    missing = sorted(req - set(got))
    if missing:
        raise ValueError(f"missing key(s) in {where}: {missing}")


def load(path):
    # load one exam file (a benchmark): data in, loudly validated, data out.
    # no wrapper objects — the exam IS its data. validation lives here because
    # load is the only entry to exam data (locate re-enters through load).
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    where = str(path)
    _check(EXAM_KEYS, data, where)
    task = data["task"]
    if (not isinstance(task, list) or not task
            or any(not isinstance(t, str) for t in task) or len(set(task)) != len(task)):
        raise ValueError(f"{where} task must be a non-empty list of unique outputs")
    _check(SCORE_KEYS, data["scoring"], f"{where} scoring")
    if data["scoring"]["match"] != "exact":
        raise ValueError(f"unknown scoring policy: {data['scoring']['match']!r}")
    if not isinstance(data["samples"], list) or not data["samples"]:
        raise ValueError(f"{where} samples must be a non-empty list")
    for s in data["samples"]:
        _check(SAMPLE_ALLOWED, s, f"{where} sample {s.get('id')!r}", required=SAMPLE_REQUIRED)
        if not isinstance(s["input"], str):
            raise ValueError(f"sample {s['id']!r} input must be text")
        if s["expected"] not in task:
            raise ValueError(f"sample {s['id']!r} expected {s['expected']!r} not in task {task}")
    if not isinstance(data["systems"], dict) or not data["systems"]:
        raise ValueError(f"{where} systems must be a non-empty dict of specs")
    for name, spec in data["systems"].items():
        kind = spec.get("kind") if isinstance(spec, dict) else None
        if kind not in KINDS:
            raise ValueError(f"{where} system {name!r}: unknown kind {kind!r}")
    return data


def locate(root, path):
    # resolve an ontology path (/<task?>/<domain?>/<language?>) to its exam data
    for p in sorted(Path(root).rglob("exam.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("path") == path:
            return load(p)
    raise FileNotFoundError(f"no exam.json with ontology path {path!r} under {root}")


def invoke(system, inp, context=None):
    # the compiler pillar: a system spec (data) -> prediction. the exam-taker.
    if not isinstance(system, dict):
        raise ValueError(f"system spec must be a dict, got: {type(system).__name__}")
    kind = system.get("kind")
    keys = KINDS.get(kind) if isinstance(kind, str) else None
    if keys is None:
        raise ValueError(f"unknown system kind: {kind!r}")
    _check(keys, system, f"system {kind}")
    if kind == "const":
        return system["value"]
    # keyword: any needle in the input -> then, otherwise -> else
    return system["then"] if any(k in inp for k in system["any"]) else system["else"]


def grade(sample, got, scoring):
    # the scoring lens: the exam's declared comparison policy applied to one case
    if scoring.get("match") != "exact":
        raise ValueError(f"unknown scoring policy: {scoring.get('match')!r}")
    return 1.0 if got == sample["expected"] else 0.0


def run(exam, system):
    # take the exam: the system answers every sample. evidence per case + aggregate
    cases = []
    for s in exam["samples"]:
        got = invoke(system, s["input"], s.get("context"))
        cases.append({"id": s["id"], "input": s["input"], "context": s.get("context"),
                      "want": s["expected"], "got": got,
                      "score": grade(s, got, exam["scoring"])})
    score = sum(c["score"] for c in cases) / len(cases)
    return {"path": exam["path"], "system": system, "cases": cases, "score": score}


def as_loss(exam):
    # the vision invariant: the benchmark as a loss over systems. lower = better.
    def loss(system):
        return 1.0 - run(exam, system)["score"]
    return loss