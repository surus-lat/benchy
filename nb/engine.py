# nb/engine.py — the exam engine.
# s07 datacentric: the Sample record (id, input, expected) is the root record.
# a benchmark is ONE data file (exam.json) plus two tiny pure functions: grade
# (the scoring lens) and run (take the exam). the system is
# a spec in data, interpreted by invoke — the compiler pillar (cloud specs
# later, same slot). the declared lenses are data, not inference: the answer
# space cannot be derived from observed expecteds (a typo'd expected would
# silently become a new class; an unrepresented class would shrink the space).
# explicit beats implicit: unknown keys raise, nothing is inferred silently.

import json
from pathlib import Path

EXAM_KEYS = {"path", "task", "scoring", "samples", "systems"}
# cycle 13 inlined SCORE_KEYS: a one-entry set consulted once — its name was
# the indirection, the check it fed is the metal (unknown scoring keys still
# raise, right here, load being the only entry to exam data).
# cycle 10 deleted sample `id`: write-only metadata — echoed into the artifact
# but never read. the list INDEX is the case id; the root record is
# (input, expected). data got smaller, error messages now carry the input.
SAMPLE_KEYS = {"input", "expected"}
# cycle 8 escalation tried to drop "kind" from the key set (presence is
# guaranteed by the kind gate) — broke: kind is a REAL key of the spec in
# data, the loud check must see every key the file carries.
KINDS = {"keyword": {"kind", "any", "then", "else"}}


def _check(allowed, got, where):
    # loud: unknown keys raise, missing keys raise. no silent defaults.
    # cycle 8 deleted the `required` param: at every call site the required
    # set equalled the allowed set (SAMPLE_REQUIRED == SAMPLE_ALLOWED), so
    # allowed==required is the only honest shape: a schema's keys are its keys.
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {unknown}")
    missing = sorted(allowed - set(got))
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
    # cycle 13 inlined SCORE_KEYS: there is ONE scoring policy and its canonical
    # form is the literal below — any deviation (unknown key, missing key,
    # wrong value) is "not the policy". load is the only entry, still loud.
    if data["scoring"] != {"match": "exact"}:
        raise ValueError(f"unknown scoring policy: {data['scoring']!r}")
    if not isinstance(data["samples"], list) or not data["samples"]:
        raise ValueError(f"{where} samples must be a non-empty list")
    for i, s in enumerate(data["samples"]):
        _check(SAMPLE_KEYS, s, f"{where} sample {i}")
        if not isinstance(s["input"], str):
            raise ValueError(f"sample {i} input must be text")
        if s["expected"] not in task:
            raise ValueError(f"sample {i} expected {s['expected']!r} not in task {task}")
    if not isinstance(data["systems"], dict) or not data["systems"]:
        raise ValueError(f"{where} systems must be a non-empty dict of specs")
    for name, spec in data["systems"].items():
        kind = spec.get("kind") if isinstance(spec, dict) else None
        if kind not in KINDS:
            raise ValueError(f"{where} system {name!r}: unknown kind {kind!r}")
    return data


def locate(root, path):
    # resolve an ontology path (/<task?>/<domain?>/<language?>) to its exam data.
    # cycle 9 tried to fuse the double-read (probe with load, skip invalid) and
    # restored it: the two reads have DIFFERENT duties. the raw read is a PROBE
    # (never validates — garbage siblings crash loudly, wrong-path files skip);
    # load is the ENTRY (the matching file must validate loudly, or be reported
    # as its real error, not as "not found"). probe != entry — that is the shape.
    for p in sorted(Path(root).rglob("exam.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("path") == path:
            return load(p)
    raise FileNotFoundError(f"no exam.json with ontology path {path!r} under {root}")


def invoke(system, inp):
    # the compiler pillar: a system spec (data) -> prediction. the exam-taker.
    # cycle 4: context param deleted — no system kind ever read it (dead param).
    if not isinstance(system, dict):
        raise ValueError(f"system spec must be a dict, got: {type(system).__name__}")
    kind = system.get("kind")
    keys = KINDS.get(kind) if isinstance(kind, str) else None
    if keys is None:
        raise ValueError(f"unknown system kind: {kind!r}")
    _check(keys, system, f"system {kind}")
    # keyword: any needle in the input -> then, otherwise -> else. cycle 12
    # tried to delete the gate above (load already validates specs) and
    # restored it: load guards DATA entry, invoke guards ARGUMENT entry —
    # a spec handed straight to as_loss/run (a prompt-optimizer's candidate)
    # never passes load. invoke is the compiler pillar's own loud boundary:
    # no validated file, no trusted spec. gate != duplicate.
    # cycle 7 deleted "const": a constant IS keyword with any=[] (no needle
    # ever matches -> always else). one kind, one code path.
    return system["then"] if any(k in inp for k in system["any"]) else system["else"]


def run(exam, system):
    # take the exam: the system answers every sample. evidence per case +
    # aggregate. the scoring lens is fused here (cycle 6): the declared policy
    # is validated at load — the only entry to exam data — so grading is just
    # its application: exact match of the declared expected, 1 point per case.
    # cycle 9 deleted run's re-check of the policy: load already rejected any
    # match != "exact", a second gate was a duplicate of load's validation.
    cases = []
    for i, s in enumerate(exam["samples"]):
        got = invoke(system, s["input"])
        cases.append({"id": i, "input": s["input"],
                      "want": s["expected"], "got": got,
                      "score": 1.0 if got == s["expected"] else 0.0})
    score = sum(c["score"] for c in cases) / len(cases)
    # cycle 11 deleted the `path` and `system` echoes: unread — the caller
    # passed the system and knows which exam it loaded. the report is evidence
    # per case + aggregate, not a copy of its own arguments.
    return {"cases": cases, "score": score}


def as_loss(exam):
    # the vision invariant: the benchmark as a loss over systems. lower = better.
    def loss(system):
        return 1.0 - run(exam, system)["score"]
    return loss
