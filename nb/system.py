"""SYSTEM — the exam taker. The AI-API: one method.

    invoke(input) -> prediction

That is the entire protocol. Everything else — model, node, workflow,
agent, old-ML — is a BACKEND: something that compiles a learned program
into this one method. The core never sees beyond it.

A system spec is data (dict). `compile_system(spec)` picks the backend
by `kind` and returns a callable: input -> prediction. Workflows and
agents are configuration over systems, not new concepts.
"""

import json
import re
import urllib.request

# ---------------------------------------------------------------- backends


def _backend_stub(spec):
    """keyword table: {pattern: label}. Pattern source is data, not code."""
    def invoke(text):
        best = None
        for pat, label in spec.get("rules", {}).items():
            if re.search(pat, str(text), re.IGNORECASE):
                best = label  # last matching rule wins; dict order is spec order
        return best if best is not None else spec.get("default", None)
    return invoke


def _backend_const(spec):
    """always return a constant. The dumbest system that can take an exam."""
    const = spec.get("const", None)
    return lambda text: const


def _backend_http(spec):
    """openai-compatible chat completion over stdlib urllib.

    request:  POST {url}/chat/completions  body {"model":..., "messages":[...]}
    response: {"choices": [{"message": {"content": ...}}]}
    input is placed in the message text; output is the content string.
    No keys, no retries — the exam proctor handles retries (angle s09).
    """
    url = spec["url"].rstrip("/") + "/chat/completions"
    model = spec.get("model", "")
    sys_prompt = spec.get("system", "")

    def invoke(text):
        body = {"model": model, "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": str(text)},
        ]}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=spec.get("timeout", 30)) as resp:
            out = json.loads(resp.read().decode())
        return out["choices"][0]["message"]["content"]
    return invoke


def _backend_chain(spec):
    """compose: run systems in sequence, feeding each output as next input.

    A workflow is just a chain whose links are systems. No new core
    concept: `kind: chain` is one backend among others.
    """
    steps = spec["steps"]
    compiled = [compile_system(s) for s in steps]
    first = compiled[0]
    rest = compiled[1:]

    def invoke(text):
        out = first(text)
        for sys_fn in rest:
            out = sys_fn(out)
        return out
    return invoke


def _backend_agent(spec):
    """a tool loop: model + tools + a fixed controller.

    An agent is a system whose spec names a model (a system), tools
    (systems), and a budget. The model emits a plain value (final
    answer) or ["tool", name, arg]; the controller runs the tool,
    appends its result, and asks again until a final answer or budget
    out. The loop is COMPILER code — a backend like any other; the
    core never learns what an agent is.
    """
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


_BACKENDS = {
    "stub": _backend_stub,
    "const": _backend_const,
    "http": _backend_http,
    "chain": _backend_chain,
    "agent": _backend_agent,
}


# ---------------------------------------------------------------- the API


def compile_system(spec):
    """spec (dict) -> callable: input -> prediction. The compiler's front door."""
    if callable(spec):
        return spec  # already a system; Python is the escape hatch
    kind = spec.get("kind", "stub")
    backend = _BACKENDS.get(kind)
    if backend is None:
        raise ValueError(f"unknown system kind: {kind!r}")
    return backend(spec)