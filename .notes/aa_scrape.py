#!/usr/bin/env python3
"""Scrape Artificial Analysis per-eval leaderboards for the evals in the AA
Intelligence Index v4.1.1 + coding/agentic evals, and build a comparison table
for the models we benchmarked on Together (Kimi K3, GLM-5.3, GLM-5.2,
DeepSeek V4 Flash, MiniMax M3) vs Claude Fable/Opus (user's reference points).

Usage: python3 aa_scrape.py [--out .notes/aa_evals.json]
"""
import argparse
import html
import json
import re
import sys
import urllib.request

BASE = "https://artificialanalysis.ai/evaluations/"
UA = {"User-Agent": "Mozilla/5.0"}

EVALS = [
    ("gdpval-aa", "GDPval-AA", "agentic real-world work (Elo)"),
    ("tau3-banking", "\U0001d70f\U000000b3-Banking", "agentic tool use"),
    ("terminalbench-v2-1", "Terminal-Bench v2.1", "agentic coding/terminal"),
    ("scicode", "SciCode", "coding (research-level)"),
    ("humanitys-last-exam", "Humanity's Last Exam", "reasoning & knowledge"),
    ("gpqa-diamond", "GPQA Diamond", "scientific reasoning"),
    ("critpt", "CritPt", "physics reasoning"),
    ("artificial-analysis-long-context-reasoning", "AA-LCR", "long context reasoning"),
    ("omniscience", "omniscienceIndex", "knowledge reliability"),
]

# models we care about (match by lowercase substring)
WANT = [
    "kimi k3", "glm-5.3", "glm-5.2", "glm-5.3-flash",
    "deepseek v4", "minimax", "claude fable", "claude opus",
    "gpt-5.6", "grok 4.6", "gemini 3.8", "qwen3.8", "kimi k2",
]


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return html.unescape(r.read().decode("utf-8", "replace"))


def scrape_eval(slug, key):
    try:
        t = fetch(BASE + slug)
    except Exception as e:
        return [], f"fetch failed: {e}"
    if key is None:  # GDPval-AA: nested {"gdpvalAaElo":[{"name":"mid","value":N}]}
        rows = re.findall(
            r'"label":"([^"]+)","gdpvalAaElo":\[[^]]*"name":"mid","value":([0-9.]+)', t)
        return rows, None
    # row shape: {"label":..., "<Eval Name>":score, "detailsUrl":...}
    rows = re.findall(r'"label":"([^"]+)","' + re.escape(key) + r'":([0-9.]+),"detailsUrl"', t)
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".notes/aa_evals.json")
    args = ap.parse_args()

    result = {}
    for slug, name, desc in EVALS:
        if slug == "gdpval-aa":
            rows, err = scrape_eval(slug, None)  # special: nested elo mid
        else:
            rows, err = scrape_eval(slug, name)
        if err:
            print(f"  {slug}: {err}", file=sys.stderr)
            continue
        # dedupe by label, keep highest variant per model family
        best = {}
        for label, value in rows:
            lbl = label.strip()
            key = None
            for w in WANT:
                if w in lbl.lower():
                    key = w
                    break
            if key is None:
                continue
            v = float(value)
            if key not in best or v > best[key][1]:
                best[key] = (lbl, v)
        result[name] = {"slug": slug, "desc": desc, "rows": rows[:200],
                        "best_of_wanted": {k: {"label": l, "value": v}
                                           for k, (l, v) in best.items()}}
        got = result[name]["best_of_wanted"]
        print(f"{name:28s} scraped {len(rows):3d} rows; wanted found: {len(got)}")

    json.dump(result, open(args.out, "w"), indent=1)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()