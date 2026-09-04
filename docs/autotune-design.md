# autotune — design (DRAFT, in loop)

## First-principles contract

The final autotune score MUST equal a fresh benchy run's score.
Therefore: the objective function is a real `benchy eval` subprocess call,
and the final reported number is one more `benchy eval` with the winning config.
No surrogate in the loop. No proxy metric. Search strategy is the only variable.

## The search space (for the handwritten-extraction case)

Exactly the knobs benchy exposes as CLI flags:

| knob | kind | benchy flag |
|------|------|-------------|
| system_prompt | text | `--system-prompt` |
| user_prompt_template | text | `--user-prompt-template` |
| temperature | float | `--temperature` |
| max_tokens | int | `--max-tokens` |

Smaller than DSPy (no demos, no signatures, no module graph). Good.

## The objective (loss function)

1. Run `benchy eval ... --exit-policy smoke --limit <K> --run-id <hash>`.
2. Parse `<out>/<run_id>/<model>/run_summary.json`.
3. Score = `tasks[<task>].document_extraction_score` (or task-appropriate metric).
4. Cache by run-id so identical configs are never re-paid (benchy resume).

## The ai_program

(TO RESOLVE — the hard part)
