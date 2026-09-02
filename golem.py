#!/usr/bin/env python3
"""the bare-metal golem — mechanical guard of the search.

    python3 golem.py check              audit engine vs last state (run after every push)
    python3 golem.py check --final      plus: refuse TOO_SOFT endings, require >= 12 cycles
    python3 golem.py check --allow-growth "<why>"
    python3 golem.py report             final metrics for SUMMARY.md

exit codes: 0 PASS · 1 NOISE · 2 GREW · 3 TOO_SOFT
stdlib only. the golem must itself be bare metal.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ENGINE = Path("nb")
STATE = Path(".golem_state.json")
ITERLOG = Path("ITERATIONS.md")
DESIGN = Path("DESIGN.md")
BANNED = re.compile(r"(Manager|Factory|Pipeline|Handler|Base[A-Z_]|Abstract)")


def engine_metrics() -> dict:
    if not ENGINE.is_dir():
        return {"files": 0, "loc": 0, "deps": 0, "concepts": []}
    files = sorted(p for p in ENGINE.rglob("*.py") if "__pycache__" not in p.parts)
    loc = deps = 0
    concepts: list[str] = []
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    for p in files:
        src = p.read_text(encoding="utf-8")
        loc += sum(
            1 for ln in src.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        )
        for m in re.finditer(r"^\s*(?:from|import)\s+([.\w]+)", src, re.M):
            mod = m.group(1)
            if mod.startswith(".") or mod == "__future__":
                continue
            if mod.split(".")[0] not in stdlib:
                deps += 1
        concepts += re.findall(r"^(?:class|def|async def)\s+(\w+)", src, re.M)
    return {"files": len(files), "loc": loc, "deps": deps,
            "concepts": sorted(set(concepts))}


def cycles() -> list[str]:
    if not ITERLOG.exists():
        return []
    return [l for l in ITERLOG.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def last_verdict(cs: list[str]) -> str | None:
    if not cs:
        return None
    m = re.search(r"verdict=(\S+)", cs[-1])
    return m.group(1) if m else None


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "report":
        cur = engine_metrics()
        cs = cycles()
        verdicts: dict[str, int] = {}
        for c in cs:
            m = re.search(r"verdict=(\S+)", c)
            if m:
                verdicts[m.group(1)] = verdicts.get(m.group(1), 0) + 1
        print(json.dumps({
            "cycles": len(cs), "verdicts": verdicts,
            "files": cur["files"], "loc": cur["loc"],
            "deps": cur["deps"], "concepts": len(cur["concepts"]),
        }, indent=2))
        return

    final = "--final" in args
    why = None
    if "--allow-growth" in args:
        i = args.index("--allow-growth")
        if i + 1 < len(args):
            why = args[i + 1]

    cur = engine_metrics()
    cs = cycles()

    # law 4: the engine is stdlib-only
    if cur["deps"]:
        print(f"NOISE: {cur['deps']} non-stdlib import(s) — the engine is stdlib-only.")
        sys.exit(1)

    # law 3: every public concept justified in DESIGN.md
    design = DESIGN.read_text(encoding="utf-8") if DESIGN.exists() else ""
    orphans = [c for c in cur["concepts"] if c not in design]
    if orphans:
        print(f"NOISE: concepts missing from DESIGN.md: {orphans}")
        sys.exit(1)
    for c in cur["concepts"]:
        if BANNED.search(c):
            print(f"golem growls: '{c}' smells like framework noise. "
                  f"prove it in DESIGN.md or fuse it away.")

    # law 2: metrics must not grow without recorded justification
    prev = json.loads(STATE.read_text()) if STATE.exists() else None
    if prev and why is None:
        grew: dict[str, list[int]] = {}
        for k in ("files", "loc", "deps"):
            if cur[k] > prev.get(k, 0):
                grew[k] = [prev.get(k, 0), cur[k]]
        if len(cur["concepts"]) > len(prev.get("concepts", [])):
            grew["concepts"] = [len(prev.get("concepts", [])), len(cur["concepts"])]
        if grew:
            print(f"GREW without justification: {grew}. delete something, "
                  f"or pass --allow-growth \"<why>\".")
            sys.exit(2)

    # law 1: final endings refuse softness (HARD_PUSH opened soft but ended
    # hard — that is the law working, not softness)
    if final:
        v = last_verdict(cs)
        if v is None or v == "TOO_SOFT":
            print(f"TOO_SOFT: last cycle verdict={v!r}. if nothing broke you "
                  f"are not pushing hard enough — push a bigger deletion NOW, "
                  f"log it, re-run.")
            sys.exit(3)
        if len(cs) < 12:
            print(f"TOO_SOFT: only {len(cs)} cycles logged — minimum 12 to finish.")
            sys.exit(3)

    STATE.write_text(json.dumps({
        "files": cur["files"], "loc": cur["loc"], "deps": cur["deps"],
        "concepts": cur["concepts"], "cycles": len(cs), "why": why,
    }, indent=1))
    print(f"PASS files={cur['files']} loc={cur['loc']} deps={cur['deps']} "
          f"concepts={len(cur['concepts'])} cycles={len(cs)}")


if __name__ == "__main__":
    main()