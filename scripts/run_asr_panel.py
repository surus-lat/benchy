#!/usr/bin/env python3
"""Run a FLEURS ASR benchmark across a panel of STT models.

Iterates each (model config, locale) pair through `benchy eval` with a
sample limit, then prints a WER/CER summary table sourced from each
run's metrics.json.

Usage:
    scripts/run_asr_panel.py --limit 50 --locales es pt
    scripts/run_asr_panel.py --limit 50 --models whisper-small-transformers \
        whisper-large-v3-turbo-transformers --skip-failures
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = [
    "whisper-tiny-transformers",
    "whisper-base-transformers",
    "whisper-small-transformers",
    "whisper-large-v3-turbo-transformers",
    "whisper-large-v3-transformers",
    "qwen3-asr-0.6b-transformers",
    "canary-1b-flash-transformers",
    "voxtral-mini-4b-transformers",
]
LOCALE_TO_SUBTASK = {"es": "transcription.fleurs_es_latam", "pt": "transcription.fleurs_pt_br"}


def _benchy_bin() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "benchy"
    return str(candidate if candidate.exists() else "benchy")


def _run_one(
    model_config: str,
    subtask: str,
    limit: int,
    extra_args: List[str],
) -> Tuple[int, Path]:
    cmd = [
        _benchy_bin(),
        "eval",
        "-c",
        model_config,
        "--tasks",
        subtask,
        "--limit",
        str(limit),
        "--log-samples",
        *extra_args,
    ]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - started
    print(f"<<< exit={proc.returncode}  wall={elapsed:.1f}s", flush=True)
    return proc.returncode, _latest_outputs_dir()


def _latest_outputs_dir() -> Path:
    outputs = REPO_ROOT / "outputs" / "benchmark_outputs"
    runs = sorted(
        (p for p in outputs.glob("*_LIMITED") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else outputs


def _harvest_metrics(run_dir: Path, subtask: str) -> Optional[Dict]:
    leaf = subtask.split(".")[-1]  # fleurs_es_latam
    for metrics in run_dir.rglob(f"*/transcription/{leaf}/*_metrics.json"):
        try:
            with metrics.open() as f:
                data = json.load(f)
            return {"path": str(metrics), "metrics": data}
        except (OSError, json.JSONDecodeError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="Samples per locale per model")
    parser.add_argument(
        "--locales",
        nargs="+",
        default=["es", "pt"],
        choices=list(LOCALE_TO_SUBTASK.keys()),
        help="Which FLEURS locales to include",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model config names (without .yaml). Defaults to the full panel.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Continue past failing model runs instead of stopping.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed through to `benchy eval`. Place after --extra.",
    )
    args = parser.parse_args()

    summary: List[Dict] = []
    for model_cfg in args.models:
        for locale in args.locales:
            subtask = LOCALE_TO_SUBTASK[locale]
            rc, run_dir = _run_one(model_cfg, subtask, args.limit, args.extra)
            entry = {"model": model_cfg, "locale": locale, "exit": rc, "run_dir": str(run_dir)}
            if rc == 0:
                harvested = _harvest_metrics(run_dir, subtask)
                if harvested:
                    m = harvested["metrics"]
                    entry["wer"] = m.get("wer")
                    entry["cer"] = m.get("cer")
                    entry["exact_match"] = m.get("exact_match")
                    entry["valid_samples"] = m.get("valid_samples")
                    entry["error_rate"] = m.get("error_rate")
                    entry["metrics_path"] = harvested["path"]
            summary.append(entry)
            if rc != 0 and not args.skip_failures:
                print(f"!! Aborting after failure on {model_cfg} / {locale}.", file=sys.stderr)
                _print_summary(summary)
                return rc

    _print_summary(summary)
    return 0


def _print_summary(rows: List[Dict]) -> None:
    print("\n========== ASR PANEL SUMMARY ==========")
    header = f"{'model':<40s} {'locale':<6s} {'exit':>4s} {'wer':>8s} {'cer':>8s} {'em':>6s} {'n':>4s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        wer = r.get("wer")
        cer = r.get("cer")
        em = r.get("exact_match")
        n = r.get("valid_samples")
        wer_s = f"{wer:.4f}" if isinstance(wer, (int, float)) else "  —"
        cer_s = f"{cer:.4f}" if isinstance(cer, (int, float)) else "  —"
        em_s = f"{em:.3f}" if isinstance(em, (int, float)) else "  —"
        n_s = f"{n}" if n is not None else " —"
        print(f"{r['model']:<40s} {r['locale']:<6s} {r['exit']:>4d} {wer_s:>8s} {cer_s:>8s} {em_s:>6s} {n_s:>4s}")

    out_path = REPO_ROOT / "outputs" / "asr_panel_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
