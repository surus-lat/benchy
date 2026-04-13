#!/usr/bin/env python3
"""End-to-end smoke test for the compiler layer.

Requires: TOGETHER_API_KEY set in environment or .env file.
Run from project root: python scripts/smoke_test_compiler.py

Tests:
  1. benchy validate  — spec passes validation
  2. generate_data()  — generates 3 examples with correct keys
  3. benchy eval      — smoke run completes and produces run_outcome.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── load .env if present ──────────────────────────────────────────────────────
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

SPEC = "benchmarks/request-by-email.yaml"
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
errors = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
        errors.append(label)


# ── 1. benchy validate ────────────────────────────────────────────────────────
print("\n── 1. benchy validate ───────────────────────────────────────")
result = subprocess.run(
    [sys.executable, "-m", "src.benchy_cli", "validate", "--benchmark", SPEC],
    capture_output=True, text=True,
)
check("exits 0", result.returncode == 0, result.stderr.strip())
check("prints valid/✓", "valid" in (result.stdout + result.stderr).lower()
      or "✓" in (result.stdout + result.stderr))

# ── 2. generate_data ──────────────────────────────────────────────────────────
print("\n── 2. generate_data (3 examples) ───────────────────────────")
if not os.environ.get("TOGETHER_API_KEY"):
    print("  ⚠  TOGETHER_API_KEY not set — skipping generation test")
else:
    try:
        sys.path.insert(0, ".")
        from src.data_generator import generate_data
        out_path = generate_data(SPEC, count=3)
        rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
        check("generates 3 rows", len(rows) == 3, f"got {len(rows)}")
        check("rows have 'input' key", all("input" in r for r in rows))
        check("rows have 'expected_output' key", all("expected_output" in r for r in rows))
        check("rows have 'id' key", all("id" in r for r in rows))
        check("'text' key absent (old format)", all("text" not in r for r in rows))
        check("'expected' key absent (old format)", all("expected" not in r for r in rows))
        if rows:
            check("expected_output is a dict (structured task)",
                  isinstance(rows[0]["expected_output"], dict),
                  f"got {type(rows[0]['expected_output']).__name__}")
    except Exception as exc:
        check("generate_data() completed", False, str(exc))

# ── 3. benchy eval smoke run ──────────────────────────────────────────────────
print("\n── 3. benchy eval smoke (--limit 2) ────────────────────────")
if not os.environ.get("TOGETHER_API_KEY"):
    print("  ⚠  TOGETHER_API_KEY not set — skipping eval smoke test")
else:
    run_id = "smoke-compiler-test"
    result = subprocess.run(
        [
            sys.executable, "-m", "src.benchy_cli", "eval",
            "--benchmark", SPEC,
            "--limit", "2",
            "--exit-policy", "smoke",
            "--run-id", run_id,
        ],
        capture_output=True, text=True, timeout=300,
    )
    check("eval exits 0", result.returncode == 0, result.stderr[-500:] if result.returncode != 0 else "")

    # Find run_outcome.json
    outcome_files = list(Path("outputs/benchmark_outputs").rglob(f"{run_id}/**/run_outcome.json"))
    if not outcome_files:
        outcome_files = list(Path("outputs/benchmark_outputs").rglob("run_outcome.json"))
    check("run_outcome.json exists", bool(outcome_files))
    if outcome_files:
        outcome = json.loads(outcome_files[-1].read_text())
        check("outcome has 'status' field", "status" in outcome, str(outcome))
        print(f"     status: {outcome.get('status')}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[31m✗ {len(errors)} check(s) failed:\033[0m")
    for e in errors:
        print(f"    • {e}")
    sys.exit(1)
else:
    print("\033[32m✓ All checks passed.\033[0m")
