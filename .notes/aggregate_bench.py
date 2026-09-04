#!/usr/bin/env python3
"""Aggregate the together_*_full benchy runs into one comparison table.

Reads the *_metrics.json artifacts (metrics nested under "metrics"),
covers spanish (accuracy per subtask) + structured_extraction (document_extraction_score),
and folds in the latency probe results.
"""
import glob
import json
import statistics

RUNS = {
    "moonshotai/Kimi-K3": "together_kimik3_full_LIMITED/Kimi-K3",
    "zai-org/GLM-5.3": "together_glm53_full_LIMITED/GLM-5.3",
    "zai-org/GLM-5.2": "together_glm52_full_LIMITED/GLM-5.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "together_dsv4flash_full_LIMITED/DeepSeek-V4-Flash-0731",
    "MiniMaxAI/MiniMax-M3": "together_minimaxm3_full_LIMITED/MiniMax-M3",
}

ES_SUBTASKS = [
    "copa_es", "escola", "mgsm_direct_es_spanish_bench", "openbookqa_es",
    "paws_es_spanish_bench", "teleia_cervantes_ave", "teleia_pce",
    "teleia_siele", "wnli_es", "xnli_es_spanish_bench",
]
SE_SUBTASKS = ["chat_extract", "email_extract", "paraloq"]


def load_metrics(base, group, subtask):
    files = [f for f in sorted(glob.glob(f"{base}/{group}/{subtask}/*_metrics.json"))
             if "per_sample" not in f]
    if not files:
        return None
    d = json.load(open(files[-1]))
    return d.get("metrics", {})


def main():
    lat = {}
    try:
        lat = json.load(open(".notes/latency_probe_results.json"))["summary"]
    except FileNotFoundError:
        pass

    report = {}
    for model, run_dir in RUNS.items():
        base = f"outputs/benchmark_outputs/{run_dir}"
        try:
            outcome = json.load(open(f"{base}/run_outcome.json"))
        except FileNotFoundError:
            report[model] = {"missing": True}
            continue
        entry = {
            "status": outcome["status"],
            "es_acc": {},
            "se_scores": {},
            "errors": 0,
        }
        es_vals, se_vals = [], []
        for st in ES_SUBTASKS:
            m = load_metrics(base, "spanish", st)
            if m is None:
                entry["es_acc"][st] = None
                continue
            acc = m.get("accuracy")
            entry["es_acc"][st] = acc
            entry["errors"] += m.get("error_count", 0) or 0
            if acc is not None:
                es_vals.append(acc)
        for st in SE_SUBTASKS:
            m = load_metrics(base, "structured_extraction", st)
            if m is None:
                entry["se_scores"][st] = None
                continue
            entry["se_scores"][st] = {
                "score": m.get("document_extraction_score"),
                "f1_partial": m.get("field_f1_partial"),
                "error_rate": m.get("error_rate"),
            }
            entry["errors"] += m.get("error_count", 0) or 0
            s = m.get("document_extraction_score")
            if s is not None:
                se_vals.append(s)
        entry["es_mean"] = statistics.mean(es_vals) if es_vals else None
        entry["se_mean"] = statistics.mean(se_vals) if se_vals else None
        if model in lat:
            entry["latency"] = lat[model]
        report[model] = entry

    print(json.dumps(report, indent=1))
    with open(".notes/bench_report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()