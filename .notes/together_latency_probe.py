#!/usr/bin/env python3
"""Latency probe for Together serverless models — measures what benchy doesn't:
TTFT (streaming), total time, reasoning-token burn, and tail latency (the "se traba").

Usage: python3 together_latency_probe.py [--models m1 m2 ...] [--n 12]
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.request

API = "https://api.together.xyz/v1/chat/completions"
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

# Prompts representative of Hermes usage: a short one and a longer reasoning one
PROMPTS = [
    # short classification-style (spanish benchmark style)
    [
        {"role": "system", "content": "Respond with only the letter of the correct option."},
        {"role": "user", "content": "¿Cuál es la capital de Argentina?\n\nOpciones:\nA) Santiago\nB) Buenos Aires\nC) Montevideo\n\nRespuesta:"},
    ],
    # medium instruction-following
    [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Extrae nombre y DNI de este texto y responde en JSON {\"nombre\": ..., \"dni\": ...}: 'El señor Juan Pérez, DNI 30.123.456, firmó el contrato.'"},
    ],
    # a trickier reasoning prompt
    [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Un tren sale de A a 60 km/h. Dos horas después sale otro de A a 90 km/h tras el primero. ¿A cuántos km de A alcanza el segundo al primero? Responde solo con el número."},
    ],
]

DEFAULT_MODELS = [
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.3",
    "zai-org/GLM-5.2",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "MiniMaxAI/MiniMax-M3",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
]


def stream_once(model: str, messages: list, max_tokens: int, timeout: float = 180.0):
    """One streaming request. Returns dict with ttft, total, tokens, empty_content."""
    key = os.environ.get("TOGETHER_API_KEY", "")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={**UA, "Authorization": f"Bearer {key}"})

    ttft = None
    t0 = time.monotonic()
    usage = {}
    content_chars = 0
    reasoning_chars = 0
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ttft is None and (
                    evt.get("choices", [{}])[0].get("delta", {}).get("content")
                    or evt.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
                ):
                    ttft = time.monotonic() - t0
                d = evt.get("choices", [{}])[0].get("delta", {})
                content_chars += len(d.get("content") or "")
                reasoning_chars += len(d.get("reasoning_content") or "")
                if evt.get("usage"):
                    usage = evt["usage"]
                fr = evt.get("choices", [{}])[0].get("finish_reason")
                if fr:
                    finish = fr
        total = time.monotonic() - t0
        return {
            "ok": True, "ttft": ttft, "total": total,
            "content_chars": content_chars, "reasoning_chars": reasoning_chars,
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "finish_reason": finish,
            "empty_content": content_chars == 0,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "total": time.monotonic() - t0}


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=12, help="iterations per model")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=1, help="models probed concurrently")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}

    async def probe(model):
        rows = []
        for i in range(args.n):
            messages = PROMPTS[i % len(PROMPTS)]
            r = await asyncio.to_thread(stream_once, model, messages, args.max_tokens)
            r["iter"] = i
            rows.append(r)
            tag = "ok" if r["ok"] else "ERR"
            ttft = f"{r['ttft']:.2f}s" if r.get("ttft") else "-"
            print(f"[{model}] iter {i+1}/{args.n} {tag} total={r['total']:.2f}s ttft={ttft}", file=sys.stderr)
        return model, rows

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(model):
        async with sem:
            return await probe(model)

    out = await asyncio.gather(*[guarded(m) for m in args.models])
    for model, rows in out:
        results[model] = rows

    # Summary
    summary = {}
    print("\n" + "=" * 100)
    hdr = f"{'model':42s} {'p50':>7s} {'p90':>7s} {'max':>8s} {'ttft50':>7s} {'err%':>5s} {'empty%':>7s} {'reason_tok':>10s}"
    print(hdr)
    print("-" * 100)
    for model, rows in results.items():
        ok = [r for r in rows if r["ok"]]
        totals = [r["total"] for r in ok]
        ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
        err_rate = 100 * (1 - len(ok) / len(rows))
        empty_rate = 100 * sum(1 for r in ok if r.get("empty_content")) / max(1, len(ok))
        rtoks = [r.get("reasoning_tokens") or 0 for r in ok]
        med_r = statistics.median(rtoks) if rtoks else 0
        s = {
            "n_ok": len(ok), "n_total": len(rows),
            "p50": pct(totals, 50), "p90": pct(totals, 90), "max": max(totals) if totals else None,
            "ttft_p50": pct(ttfts, 50), "ttft_max": max(ttfts) if ttfts else None,
            "err_rate": err_rate, "empty_content_rate": empty_rate,
            "median_reasoning_tokens": med_r,
        }
        summary[model] = s
        print(f"{model:42s} {s['p50']:6.2f}s {s['p90']:6.2f}s {s['max']:7.2f}s "
              f"{(s['ttft_p50'] or 0):6.2f}s {err_rate:4.0f}% {empty_rate:6.0f}% {med_r:10.0f}")
    print("=" * 100)

    payload = {"summary": summary, "raw": results}
    outp = args.out or ".notes/latency_probe_results.json"
    with open(outp, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"saved -> {outp}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())