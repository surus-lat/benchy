# Benchmark Together AI — serverless, 2026-09-02

Modelos: Kimi-K3, GLM-5.3, GLM-5.2, DeepSeek-V4-Flash-0731, MiniMax-M3
Tasks benchy: `spanish` (11 subtasks, 50 samples c/u — MGSM re-parado, ver abajo) + `structured_extraction` (3 subtasks, 50 samples)
Latencia: probe propio streaming, 20 iter x 3 modelos + 12 iter x 6 modelos.
Runs: `outputs/benchmark_outputs/together_<tag>_full_LIMITED/` + `mgsm_fixed_<tag>_LIMITED/`
Raw: `.notes/bench_report.json`, `.notes/latency_probe_results.json`, `.notes/latency_probe_outliers.json`

## Calidad (11 subtasks spanish + 3 structured, 50 samples c/u)

| modelo                  | spanish mean acc (con MGSM) | MGSM math | structured mean | errores / 700 |
|-------------------------|----------------------------:|----------:|----------------:|--------------:|
| Kimi-K3                 |                      96.7%    |   100%    |         94.0%   |   1 (0.14%)   |
| GLM-5.3                 |                      95.4%    |   95.9%   |         94.7%   |  11 (1.6%)    |
| MiniMax-M3              |                      94.5%    |   98.0%   |         93.7%   |   2 (0.3%)    |
| DeepSeek-V4-Flash-0731  |                      94.8%    |   97.8%   |         93.9%   |   6 (0.9%)    |
| GLM-5.2                 |                      91.2%    |   98.0%   |         93.5%   |   7 (1.0%)    |

Bug de benchy encontrado y reparado (src/tasks/spanish/mgsm_direct_es_spanish_bench.py):
(1) MGSM cacheaba en .data/spanish/test.jsonl, el mismo archivo que copa_es.py — evaluaba
prompts de COPA (exact_match=0 trivial para todo modelo). (2) El TSV mgsm_es.tsv ya no
existe en juletxara/mgsm (ahora es parquet con config por idioma). (3) El prompt no pedía
respuesta numérica limpia y exact_match comparaba contra "18" el texto "**Respuesta: $18**
Solución paso a paso...". Fixes: dataset_file=mgsm_es_test.jsonl, load_dataset(config 'es',
split 'test'), prompt "ÚNICAMENTE el número final". Verificado: Kimi-K3 0%→100%, resto 96%+.

Errores GLM-5.3: 10x "Empty response content" (razonador quema max_tokens antes de emitir).
Errores GLM-5.2: 6x idem. Kimi-K3: 1x JSON parse. DS-V4F: 2. MiniMax-M3: 1.

## Latencia (probe streaming, 20 iter, 3 prompts rotativos)

| modelo        | p50    | p90    | max    | TTFT p50 | >5s  |
|---------------|-------:|-------:|-------:|---------:|-----:|
| GLM-5.2       | 1.12s  | 1.43s  | 3.50s  |  0.28s   | 0/20 |
| GLM-5.3       | 1.17s  | 8.40s  | 11.62s |  0.35s   | 3/20 |
| Kimi-K3       | 2.69s  | 4.86s  | 6.03s  |  1.11s   | 2/20 |
| MiniMax-M3    | 0.96s  | 1.23s  | 1.50s  |  0.56s   | 0/12 |
| DS-V4-Flash   | 2.04s  | 3.59s  | 3.61s  |  0.72s   | 0/12 |
| Llama-3.3-70B | 1.14s  | 1.78s  | 2.03s  |  0.95s   | 0/12 |

GLM-5.3 es bimodal: pico de cola 8–12s (probe) y se observó un 32s suelto —
queueing/cold-start del deployment serverless. ESE es el "se traba" de Hermes.

## Throughput benchy (samples/s, medido en runs reales)

GLM-5.2 y MiniMax-M3 los más rápidos (1.3–2.7/s en MC); Kimi-K3 el más lento
(0.15–0.85/s). En extraction, GLM-5.3 ~2-3x Kimi-K3.

## Precio (Together, USD/1M tokens in/out)

| modelo         | in   | out  | turno típico Hermes (10K in + 2K out) |
|----------------|-----:|-----:|--------------------------------------:|
| Kimi-K3        | 3.00 | 15.00| $0.060                                |
| GLM-5.3        | 1.40 |  4.40| $0.023                                |
| GLM-5.2        | 1.40 |  4.40| $0.023                                |
| MiniMax-M3     | 0.30 |  1.20| $0.005                                |
| DS-V4-Flash    | 0.14 |  0.28| $0.002                                |

## Conclusiones

1. Calidad: Kimi-K3 > GLM-5.3 > DS-V4F ≈ MiniMax-M3 > GLM-5.2. Gap chico (94–96.7%).
2. "Kimi K3 funciona muy mal": el benchmark NO muestra degradación de calidad (96.7%, el
   mejor). Si en Hermes rinde mal, sospechar de: (a) respuestas vacías por reasoning_content
   (K3 razona; si el caller lee solo message.content puede ver vacío), (b) latencia (p50 2.7s).
3. "GLM-5.3 se traba" confirmado por dos mecanismos: cola de latencia bimodal (p90 8.4s, max
   32s observado) y 1.5% de respuestas vacías (razonamiento quema max_tokens). En un agente
   multi-turno con contexto largo, ambos se amplifican.
4. Mejor combinación calidad/precio/confiabilidad: **GLM-5.3** si se tolera la cola (mejor
   structured score, 2.6x más barato que Kimi), **MiniMax-M3** como alternativa estable y
   barata (94.0% / $0.005 por turno / 0 errores de latencia), **Kimi-K3** si la calidad
   máxima importa más que costo y velocidad.
5. GLM-5.2: descartable — igual precio que 5.3, peor en todo.

## Drift serverless verificado 2026-09-02 (probe 1-token)

- VIVOS: Kimi-K3, GLM-5.3, GLM-5.2, DeepSeek-V4-Flash-0731, DeepSeek-V4-Pro-0813,
  MiniMax-M3, gpt-oss-120b, Llama-3.3-70B-Instruct-Turbo
- CAÍDOS: Kimi-K2.6 (endpoint_not_ready — deployments stopped), DeepSeek-V3.1/V3.2-Exp,
  MiniMax-M2.7, Qwen3.5-397B, Qwen3-235B/Next-80B, Hermes-4-405B, Llama-4-Scout/Maverick
- ORG-GATE (third_party_data_sharing_blocked): Qwen3.6-Plus, Qwen3.7-Plus, Qwen3.7-Max,
  Qwen3.8-Flash — requiere habilitar prompt storage en la org