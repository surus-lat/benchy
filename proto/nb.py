"""nb.py — a benchmark is ONE yaml file; this runs it to a score.

The 4 pillars of benchy's ontology in one file (simplest representation
found by the 10-angle bare-metal search):
  task    the program: input -> output            (program pillar)
  scoring what good means: the comparison policy (scoring pillar)
  cases   the exam: n inputs + expected outputs   (data pillar)
  system  the exam-taker: a spec COMPILED to      (compiler/ai-endpoint
          invoke(input) -> prediction             pillar)

Everything is data; zero user Python (a callable is the escape hatch).
Cloud-first: kind: http speaks openai-compatible chat completions —
together.ai, openai, vllm, anything that speaks the dialect.
loss = 1 - score (lower = better; the optimizer's bridge).

    python3 nb.py bench/sentiment.yaml            # default system
    python3 nb.py bench/sentiment.yaml good      # a named system
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def load(bench_path):
    """yaml -> the 4 pillars as plain data."""
    return yaml.safe_load(Path(bench_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- scoring

def score_case(scoring, predicted, expected):
    """The grading policy, interpreted from data. Loud on the unknown."""
    if scoring == {"match": "exact"}:
        return 1.0 if predicted == expected else 0.0
    if (set(scoring) == {"match", "weights"} and scoring["match"] == "fields"
            and isinstance(scoring["weights"], dict) and scoring["weights"]):
        w = scoring["weights"]
        return sum(v for f, v in w.items()
                   if predicted.get(f) == expected[f]) / sum(w.values())
    raise ValueError(f"unsupported scoring: {scoring!r}")


# ---------------------------------------------------------------- system backends

def _b_stub(spec):
    """keyword table {pattern: label}; last match wins."""
    rules, default = spec.get("rules", {}), spec.get("default")
    def invoke(text):
        best = default
        for pat, label in rules.items():
            if re.search(pat, str(text), re.IGNORECASE):
                best = label
        return best
    return invoke


def _b_regex(spec):
    """pure-data regex systems: extraction (fields, group 1, miss default)
    or classification (if/then/else)."""
    if "fields" in spec:
        fields, miss = spec["fields"], spec.get("miss", "")
        def invoke(text):
            return {f: (m.group(1) if (m := re.search(p, str(text))) else miss)
                    for f, p in fields.items()}
        return invoke
    if "if" in spec:
        pat, then, els = spec["if"], spec["then"], spec["else"]
        def invoke(text):
            return then if re.search(pat, str(text)) else els
        return invoke
    raise ValueError(f"regex system needs 'fields' or 'if': {sorted(spec)}")


def _b_http(spec):
    """openai-compatible chat completions over stdlib urllib — the
    cloud-first exam-taker (together.ai, openai, vllm, ...). The proctor
    retries transient failures (429/5xx/network) with backoff."""
    url = spec["url"].rstrip("/") + "/chat/completions"
    model, prompt = spec["model"], spec.get("system", "")
    key_env, timeout = spec.get("api_key_env"), spec.get("timeout", 60)
    retries, path = spec.get("retries", 2), spec.get(
        "response_path", "choices.0.message.content")
    parse, opts = spec.get("parse"), spec.get("options", {})

    def invoke(text):
        body = {"model": model, **opts, "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(text)}]}
        headers = {"Content-Type": "application/json",
                    "User-Agent": "nb/0.1",  # urllib's default UA is 403'd
                    **spec.get("headers", {})}
        if key_env and (key := os.environ.get(key_env)):
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers)
        out = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    out = json.loads(resp.read().decode())
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                transient = not isinstance(e, urllib.error.HTTPError) or \
                    e.code in (429, 500, 502, 503)
                if attempt == retries or not transient:
                    raise
                time.sleep(2 ** attempt)
        for k in path.split("."):
            out = out[int(k)] if k.isdigit() else out[k]
        if parse == "json":
            s = str(out).strip()
            if s.startswith("```"):  # models fence; the compiler unfences
                s = s.split("```")[1].removeprefix("json").strip()
            return json.loads(s)
        return out
    return invoke


def _b_chain(spec):
    """a workflow is just a chain whose links are systems. No new concept."""
    compiled = [compile_system(s) for s in spec["steps"]]
    def invoke(text):
        out = text
        for sys_fn in compiled:
            out = sys_fn(out)
        return out
    return invoke


def _b_agent(spec):
    """a tool loop: model + tools + budget. Config, not core code."""
    model = compile_system(spec["model"])
    tools = {n: compile_system(s) for n, s in spec.get("tools", {}).items()}
    budget = spec.get("max_iters", 5)
    def invoke(text):
        out = model(text)
        for _ in range(budget):
            if not (isinstance(out, list) and out and out[0] == "tool"):
                return out  # a plain value IS the final answer
            name, arg = out[1], out[2] if len(out) > 2 else text
            if name not in tools:
                return out  # unknown tool: the utterance is the prediction
            text = f"{text}\nresult: {tools[name](arg)}"
            out = model(text)
        return out  # budget out: the last utterance, graded honestly
    return invoke


_BACKENDS = {"stub": _b_stub, "regex": _b_regex, "http": _b_http,
             "chain": _b_chain, "agent": _b_agent}


def compile_system(spec):
    """spec (data) -> callable input->prediction. The compiler's front door."""
    if callable(spec):
        return spec  # already a system; Python is the escape hatch
    backend = _BACKENDS.get(spec.get("kind"))
    if backend is None:
        raise ValueError(f"unknown system kind: {spec.get('kind')!r}")
    return backend(spec)


# ---------------------------------------------------------------- the run

def run(bench, system=None, limit=None):
    """benchmark.run(system) -> the graded artifact. Self-contained receipt."""
    if isinstance(bench, (str, Path)):
        bench = load(bench)
    spec = bench["system"] if system is None else (
        bench["systems"][system] if isinstance(system, str) else system)
    name = system if isinstance(system, str) else "default"
    invoke, scoring = compile_system(spec), bench["scoring"]
    cases = bench["cases"][:limit] if limit else bench["cases"]
    graded = []
    for c in cases:
        p = invoke(c["input"])
        graded.append({**c, "predicted": p,
                       "score": float(score_case(scoring, p, c["expected"]))})
    score = sum(c["score"] for c in graded) / len(graded)  # g = mean (loud on 0)
    return {"task": bench["task"], "system": name, "cases": graded,
            "score": score, "loss": 1 - score}


def as_loss(bench, system=None):
    """loss = benchmark.as_loss() — the identity. Software-3.0's bridge."""
    return run(bench, system)["loss"]


if __name__ == "__main__":
    args, system, _limit = sys.argv[1:], None, None
    bench_path = args.pop(0) if args else None
    while args:
        a = args.pop(0)
        if a == "--limit":
            _limit = int(args.pop(0))
        elif a.startswith("--"):
            print(f"unknown flag: {a}", file=sys.stderr)
            sys.exit(2)
        else:
            system = a
    if bench_path is None:
        print("usage: python3 nb.py <bench.yaml> [system] [--limit N]",
              file=sys.stderr)
        sys.exit(2)
    print(json.dumps(run(bench_path, system, _limit), indent=2))