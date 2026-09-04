# Complex Problem Solving — Kimi K3 / GLM / vs Claude Fable-Opus
Fuente: Artificial Analysis leaderboards, scrapeado 2026-09-02 (.notes/aa_evals.json, scraper: .notes/aa_scrape.py)

Los benchmarks que usa el ecosistema para comparar estos modelos en "solve complex problems"
(AA Intelligence Index v4.1.1, 9 evals) + agentic/coding:

| eval (qué mide)                | Kimi K3 | GLM-5.3 | Claude Fable 5.1 | Claude Opus 5 | Grok 4.6 | DS V4 Pro |
|--------------------------------|--------:|--------:|----------------:|--------------:|---------:|----------:|
| AA Intelligence Index (0-100)  |   59.7  |   59.5  |     65.7         |     63.1      |   60.9   |   53.2    |
| GDPval-AA (agentic work, Elo)  |  1668   |  1765*  |     1853         |     1824      |   1755   |   1577    |
| τ³-Banking (agentic tool use)  |  46.0%  |  50.3%  |     47.2%        |     42.1%     |  50.7%   |   39.6%   |
| Terminal-Bench v2.1 (agentic) |  85.0%  |  84.3%* |     91.4%        |     89.1%     |  88.4%   |     —     |
| SciCode (coding research)      |  58.7%  |  56.5%  |     62.0%        |     55.7%     |  53.6%   |   49.2%   |
| Humanity's Last Exam           |  46.9%  |  42.3%  |     59.1%        |     54.9%     |  42.9%   |   41.0%   |
| GPQA Diamond (ciencia)         |  93.5%  |  91.7%  |     93.7%        |     93.7%     |  94.9%   |   92.8%   |
| CritPt (física)                |  23.4%  |  19.1%  |     31.1%        |     29.1%     |  17.1%   |   18.0%   |
| AA-LCR (long context reasoning)|  82.7%  |  78.0%* |     80.0%        |     75.7%     |    —     |     —     |
| AA-Omniscience Index (-100..100)|  19.7  |   14.3  |     43.5         |     37.1      |  30.5    |     —     |

* GLM-5.3-Flash (variante flash) en algunos evals porque GLM-5.3 max no aparece en ese top-20.
Notas:
- MiniMax-M3: GPQA 92.9%, AA-LCR 80.3%, Omniscience 1.35 — medido pero lejos de la cima en agentic.
- GLM-5.2 no figura en ningún leaderboard AA (no lo evaluaron / no califica al top-20).
- LiveCodeBench/AIME/Math-500/MMLU-Pro en AA están congelados (generación de modelos vieja:
  Gemini 3, GLM-4.7, GPT-5.2 era) — no útiles para comparar modelos actuales.
- DS V4 Pro es la variante Pro (la Flash que probamos en Together ni siquiera está evaluada en la
  mayoría).

## Lectura

1. En complex problem solving, Kimi K3 y GLM-5.3 están EN LA MISMA LIGA (59.7 vs 59.5 en el
   Index) y ambos compiten de cerca con Claude Fable/Opus (63-66). El gap con Anthropic es real
   pero chico: ~5 puntos de Index, y en evals puntuales GLM-5.3 GANA a Opus 5 en τ³-Banking,
   SciCode y AA-LCR.
2. Kimi K3 > GLM-5.3 en razonamiento "puro" (HLE +4.6pts, CritPt +4.3, GPQA +1.8, LCR +4.7,
   Omniscience +5.4). GLM-5.3 > Kimi en agentic (GDPval +97 Elo, τ³ +4.3pts).
3. Claude Fable 5.1 domina HLE (59.1%) y CritPt — el mejor razonador general, como esperás.
4. Si tu uso es agentic (Hermes = tool use, código, terminal), GLM-5.3 es sorprendentemente
   competitivo vs Opus/Fable; si es razonamiento profundo one-shot, Kimi K3.
5. Ninguna eval de AA mide lo que medimos con benchy (español/structured): son complementarias.