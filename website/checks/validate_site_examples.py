#!/usr/bin/env python3
"""Validate the code examples shown on the benchy website against the real engine.

Extracts every CodeBlock from website/src/pages/Index.tsx and checks:
  - yaml blocks: the full benchmark.yaml compiles with `benchy compile`;
    partial blocks (program/scoring/ai-system) splice into the canonical
    example and also compile.
  - bash blocks: every `benchy <subcommand>` exists per `benchy --help`.
  - json blocks: parse as JSON; diagnostics use the engine's error shape.

Exit 0 if all pass, 1 otherwise. Used by CI (on engine or website changes)
and by the verify-benchy-website skill.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WEBSITE = Path(sys.argv[1] if len(sys.argv) > 1 else "website")
ENGINE = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

INDEX = WEBSITE / "src/pages/Index.tsx"
failures = []
passes = []


def ok(msg):
    passes.append(msg)
    print(f"  PASS  {msg}")


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def extract_blocks():
    src = INDEX.read_text()
    blocks = []
    # CodeBlock props: code={`...`} plus language/title
    for m in re.finditer(r'<CodeBlock\s+code=\{`(.*?)`\}(.*?)>', src, re.S):
        code, attrs = m.group(1), m.group(2)
        lang = (re.search(r'language="([^"]+)"', attrs) or [None, "bash"])[1]
        title = (re.search(r'title="([^"]+)"', attrs) or [None, ""])[1]
        blocks.append({"code": code, "language": lang, "title": title})
    return blocks


def benchy_compile(yaml_text, workdir):
    f = Path(workdir) / "benchmark.yaml"
    f.write_text(yaml_text)
    r = subprocess.run(["benchy", "compile", str(f), "-o", str(Path(workdir) / "ir.json")],
                       capture_output=True, text=True, cwd=workdir)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def main():
    if not INDEX.exists():
        print(f"ERROR: {INDEX} not found")
        return 1
    import shutil
    if not shutil.which("benchy"):
        print("ERROR: `benchy` CLI not on PATH. Install the engine first: pip install -e <engine repo>")
        return 1

    blocks = extract_blocks()
    print(f"extracted {len(blocks)} code blocks from {INDEX}")
    yaml_blocks = [b for b in blocks if b["language"] == "yaml"]
    bash_blocks = [b for b in blocks if b["language"] == "bash"]
    json_blocks = [b for b in blocks if b["language"] == "json"]

    full = next((b for b in yaml_blocks if b["title"] == "benchmark.yaml"), None)

    # --- bash: subcommands exist
    help_out = subprocess.run(["benchy", "--help"], capture_output=True, text=True).stdout
    for b in bash_blocks:
        for cmd in re.findall(r"benchy\s+([a-z-]+)", b["code"]):
            if re.search(rf"\b{re.escape(cmd)}\b", help_out):
                ok(f"bash: `benchy {cmd}` exists in CLI")
            else:
                fail(f"bash: `benchy {cmd}` NOT in CLI help ({b['title']!r})")

    # --- yaml: full example compiles
    with tempfile.TemporaryDirectory() as td:
        if full:
            good, out = benchy_compile(full["code"], td)
            ok("yaml: full benchmark.yaml compiles") if good else fail(f"yaml: full benchmark.yaml FAILED to compile:\n{out}")
        else:
            fail("yaml: no full benchmark.yaml block found (title='benchmark.yaml')")

        # --- yaml: partial blocks splice into canonical example
        canonical = full["code"] if full else ""
        for b in yaml_blocks:
            if b is full:
                continue
            first_line = next((ln for ln in b["code"].splitlines()
                               if ln.strip() and not ln.strip().startswith("#")), "")
            top_key = first_line.split(":")[0].strip()
            if not re.match(r"^[a-z-]+$", top_key):
                continue
            # replace the top-level section in the canonical example
            pattern = rf"(?ms)^{re.escape(top_key)}:.*?(?=^[a-z-]+:|\Z)"
            if not re.search(pattern, canonical):
                fail(f"yaml: canonical example has no top-level `{top_key}:` to splice ({b['title']!r})")
                continue
            if b["code"].count("ai-system:") > 1:
                variants = re.split(r"\n\s*#.*\n(?=ai-system:)", b["code"])
                variants = [v[v.index("ai-system:"):] for v in variants if "ai-system:" in v]
            else:
                variants = [b["code"]]
            for i, variant in enumerate(variants):
                spliced = re.sub(pattern, variant.rstrip() + "\n", canonical, count=1)
                good, out = benchy_compile(spliced, td)
                label = f"yaml: `{top_key}` block (variant {i+1}) splices and compiles"
                ok(label) if good else fail(f"{label.replace('splices and compiles', 'FAILED')}:\n{out}")

    # --- json blocks parse
    for b in json_blocks:
        code = b["code"]
        if "..." in code:
            code = re.sub(r'"results": \[.*?\]', '"results": []', code, flags=re.S)
        try:
            json.loads(code)
            ok(f"json: {b['title']!r} parses")
        except Exception as e:
            fail(f"json: {b['title']!r} does not parse: {e}")

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
