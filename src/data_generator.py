"""Synthetic data generator for benchmark.yaml tasks.

Generates ``{text, expected}`` pairs from a task spec using an LLM.
Called by the ``synthesize-data`` skill and ``benchy create`` wizard.

Usage::

    from src.data_generator import generate_data

    out_path = generate_data("benchmark.yaml", count=30)
    # → .data/<benchmark_name>/train.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Generator model — not user-configurable; internal engine detail
_GENERATOR_MODEL = "google/gemma-4-31B-it"
_GENERATOR_BASE_URL = "https://api.together.xyz/v1"
_GENERATOR_API_KEY_ENV = "TOGETHER_API_KEY"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(spec: Dict[str, Any]) -> str:
    """Build a generation prompt from a benchmark spec."""
    task_section = spec.get("task", spec)
    task_type = task_section.get("type", "extraction")
    input_desc = task_section.get("input", {}).get("description", "text input")
    output_section = task_section.get("output", {})
    output_type = output_section.get("type", "text")
    fields: List[Dict] = output_section.get("fields", [])
    seed = spec.get("data", {}).get("seed_description", "") or spec.get("seed_description", "")

    header = (
        "You are generating test data for an AI evaluation benchmark.\n\n"
        f"Task type: {task_type}\n"
        f"Input: {input_desc}\n"
    )

    if fields:
        field_lines = "\n".join(
            f"  - {f['name']} ({f.get('type', 'string')}): {f.get('description', '')}"
            + (" [required]" if f.get("required", True) else " [optional]")
            for f in fields
        )
        body = (
            f"Output fields:\n{field_lines}\n\n"
            "Generate ONE realistic and varied example.\n"
            "Return a JSON object with exactly two keys:\n"
            '  "text": (string) a realistic input for this task\n'
            '  "expected": (object) the correct extracted fields as a JSON object\n\n'
            "Make the example varied — mix different values, formats, and edge cases."
        )
    else:
        body = (
            f"Output type: {output_type}\n\n"
            "Generate ONE realistic and varied example.\n"
            "Return a JSON object with exactly two keys:\n"
            '  "text": (string) a realistic input for this task\n'
            '  "expected": (string) the correct output\n\n'
            "Make the example varied — mix different values and phrasings."
        )

    if seed:
        body += f"\n\nSeed guidance: {seed}"

    body += "\n\nReturn ONLY valid JSON. No markdown, no explanation, no comments."
    return header + body


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def _generate_one(
    client: Any,
    prompt: str,
    idx: int,
    total: int,
) -> Optional[Dict[str, Any]]:
    try:
        response = await client.chat.completions.create(
            model=_GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        # Strip markdown code fences if the model wrapped the JSON
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        result = json.loads(content)
        print(f"\rGenerated {idx}/{total}...", end="", flush=True)
        return result
    except json.JSONDecodeError as exc:
        print(f"\nWarning: example {idx} had invalid JSON: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"\nWarning: example {idx} failed: {exc}", file=sys.stderr)
        return None


async def _generate_all(prompt: str, count: int) -> List[Dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for data generation. "
            "Install it with: pip install openai"
        ) from exc

    api_key = os.environ.get(_GENERATOR_API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"Data generation requires {_GENERATOR_API_KEY_ENV} to be set. "
            "Add it to your .env file: TOGETHER_API_KEY=<your-key>"
        )

    client = AsyncOpenAI(api_key=api_key, base_url=_GENERATOR_BASE_URL)
    tasks = [_generate_one(client, prompt, i + 1, count) for i in range(count)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------

def _validate_sample(sample: Dict[str, Any], fields: List[Dict[str, Any]]) -> bool:
    """Return True if sample has the required structure."""
    if not isinstance(sample, dict):
        return False
    has_input = "text" in sample or "input" in sample
    if not has_input:
        return False
    if "expected" not in sample:
        return False
    if fields:
        expected = sample.get("expected")
        if not isinstance(expected, dict):
            return False
        # All required fields must be present
        for f in fields:
            if f.get("required", True) and f["name"] not in expected:
                return False
    return True


def _normalize(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "id": str(idx),
        "text": sample.get("text") or sample.get("input", ""),
        "expected": sample.get("expected", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_data(
    benchmark_yaml_path: str = "benchmark.yaml",
    count: int = 30,
    output_dir: Optional[str] = None,
) -> str:
    """Generate synthetic benchmark examples from a benchmark.yaml task spec.

    Args:
        benchmark_yaml_path: Path to benchmark.yaml
        count: Number of examples to generate (default: 30)
        output_dir: Override output directory. Default: .data/<benchmark_name>/

    Returns:
        Path to the written JSONL file.

    Raises:
        FileNotFoundError: If benchmark.yaml does not exist.
        ValueError: If the task uses image/document input (cannot synthesize images).
        ValueError: If generation produces no valid examples.
    """
    p = Path(benchmark_yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"benchmark.yaml not found: {p.resolve()}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    spec = raw.get("benchmark", raw)
    benchmark_name = spec.get("name", "benchmark")
    task_section = spec.get("task", {})

    input_type = task_section.get("input", {}).get("type", "text").lower()
    if input_type in ("image", "document", "pdf"):
        raise ValueError(
            "Cannot synthesize data for image/document input tasks. "
            "I can generate expected outputs but not the images themselves. "
            "You need real images — use setup-data to format them instead."
        )

    fields: List[Dict[str, Any]] = task_section.get("output", {}).get("fields", [])
    prompt = _build_prompt(spec)

    print(f"Generating {count} examples for '{benchmark_name}'...")
    raw_samples = asyncio.run(_generate_all(prompt, count))
    print()  # newline after progress

    valid = [s for s in raw_samples if _validate_sample(s, fields)]
    dropped = len(raw_samples) - len(valid)
    if dropped:
        print(f"  Dropped {dropped} invalid example(s) (missing required fields).")

    if not valid:
        raise ValueError(
            "No valid examples were generated. "
            "Check your task spec and that TOGETHER_API_KEY is set and valid."
        )

    samples = [_normalize(s, i) for i, s in enumerate(valid)]

    out_dir = Path(output_dir) if output_dir else Path(".data") / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "train.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} examples → {out_file}")
    return str(out_file)
