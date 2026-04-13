"""Interactive wizard that produces benchmark.yaml.

Invoked via ``benchy create``. Guides the user through four questions and
writes benchmark.yaml at the project root. Zero benchy internals exposed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: Optional[str] = None) -> str:
    """Read a non-empty line from stdin."""
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if value:
            return value
        if default is not None:
            return default
        print("  (required, please enter a value)")


def _choose(prompt: str, options: List[str], default: int = 1) -> int:
    """Present a numbered menu and return the 1-based selection."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if i == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        try:
            raw = input(f"\nEnter choice [1-{len(options)}] (default {default}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not raw:
            return default
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _box(lines: List[str], title: str = "") -> None:
    width = max(len(l) for l in lines + [title]) + 4
    top = "╭" + "─" * (width - 2) + "╮"
    bot = "╰" + "─" * (width - 2) + "╯"
    print(f"\n{top}")
    if title:
        print(f"│  {title:<{width - 4}}  │")
        print(f"│  {'':─<{width - 4}}  │")
    for line in lines:
        print(f"│  {line:<{width - 4}}  │")
    print(bot)


# ---------------------------------------------------------------------------
# Section collectors
# ---------------------------------------------------------------------------

def _collect_task() -> Dict[str, Any]:
    """Stage 1: what does the AI do?"""
    print("\n── Stage 1: Define the task ──────────────────────────────────")
    description = _ask("What does your AI do? (plain English)")

    input_choice = _choose(
        "What type of input does it take?",
        [
            "Text (a document, question, or passage)",
            "Image or PDF",
            "Text + image",
        ],
    )
    input_type_map = {1: "text", 2: "image", 3: "text"}
    input_type = input_type_map[input_choice]
    multimodal = input_choice in (2, 3)

    output_choice = _choose(
        "What does it produce?",
        [
            "Extracts specific fields (name, amount, date…)",
            "Classifies into one of several categories",
            "Answers a question in free text",
            "Translates text",
            "Other — I'll describe it",
        ],
    )
    output_map = {
        1: ("extraction", "structured"),
        2: ("classification", "label"),
        3: ("qa", "text"),
        4: ("translation", "text"),
        5: ("freeform", "text"),
    }
    task_type, output_type = output_map[output_choice]

    fields: List[Dict[str, Any]] = []
    labels: List[str] = []

    if output_choice == 1:
        # Extraction — collect fields
        print(
            "\nList the fields your AI should extract."
            "\nFormat: field_name: description (one per line, blank line to finish)"
        )
        while True:
            try:
                line = input("  › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                if fields:
                    break
                print("  (add at least one field)")
                continue
            if ":" in line:
                name, _, desc = line.partition(":")
                fields.append({
                    "name": name.strip(),
                    "type": "string",
                    "description": desc.strip(),
                    "required": True,
                })
            else:
                fields.append({"name": line.strip(), "type": "string", "required": True})
        print(f"  ✓ {len(fields)} field(s) defined")

    elif output_choice == 2:
        # Classification — collect labels
        print(
            "\nList the possible categories (labels), one per line. Blank line to finish."
        )
        while True:
            try:
                line = input("  › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                if labels:
                    break
                print("  (add at least one label)")
                continue
            labels.append(line)
        print(f"  ✓ {len(labels)} label(s) defined")

    task: Dict[str, Any] = {
        "type": task_type,
        "input": {"type": input_type, "description": description},
        "output": {"type": output_type},
    }
    if multimodal and input_type == "text":
        task["input"]["secondary_type"] = "image"
    if fields:
        task["output"]["fields"] = fields
    if labels:
        task["output"]["labels"] = labels

    return task


def _collect_scoring(task: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2: how to grade the output?"""
    print("\n── Stage 2: Define scoring ───────────────────────────────────")
    task_type = task.get("type", "freeform")
    output_type = task.get("output", {}).get("type", "text")

    if task_type == "classification" or output_type == "label":
        # Classification always uses exact match
        print("  Using exact match scoring for classification tasks.")
        return {"type": "binary", "case_sensitive": False}

    options = [
        "Each correct field earns a point  (score = correct / total)  — best for extraction",
        "All fields must be correct to earn a point  (pass/fail)",
        "Approximate matches count  (fuzzy — good for QA, translation, summaries)",
    ]
    defaults = {
        "extraction": 1,
        "qa": 3,
        "translation": 3,
        "freeform": 3,
        "summarization": 3,
    }
    default_choice = defaults.get(task_type, 1)
    choice = _choose("How should a correct output be scored?", options, default=default_choice)

    scoring: Dict[str, Any] = {}
    if choice == 1:
        scoring["type"] = "per_field"
        scoring["partial_credit"] = True
        tol_raw = input(
            "\n  Numeric tolerance (e.g. 0.01 means 1% difference counts as correct).\n"
            "  Press Enter to skip: "
        ).strip()
        if tol_raw:
            try:
                scoring["numeric_tolerance"] = float(tol_raw)
            except ValueError:
                pass
        ci_raw = input("  Case-sensitive comparison? [y/N]: ").strip().lower()
        scoring["case_sensitive"] = ci_raw in ("y", "yes")
    elif choice == 2:
        scoring["type"] = "binary"
        ci_raw = input("\n  Case-sensitive comparison? [y/N]: ").strip().lower()
        scoring["case_sensitive"] = ci_raw in ("y", "yes")
    else:
        scoring["type"] = "semantic"

    return scoring


def _collect_data(benchmark_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3b: supply the data."""
    print("\n── Stage 3b: Set up data ─────────────────────────────────────")
    input_type = task.get("input", {}).get("type", "text").lower()
    is_image = input_type in ("image", "document", "pdf")

    options = [
        "I have a file  (CSV or JSONL)",
        "I have data but need to reformat it  (I'll describe the columns)",
        "No data — generate synthetic examples",
        "Skip — I'll set this up later",
    ]
    choice = _choose("Do you have test data?", options)

    data: Dict[str, Any] = {}

    if choice == 1:
        # Direct file
        path = _ask("Path to your file (CSV or JSONL)")
        data["source"] = "local"
        data["path"] = path

    elif choice == 2:
        # Adapt format — produce a mapping note
        print(
            "\n  The required columns are: id (or auto-generated), text (input), expected (ground truth)."
        )
        print("  You'll need to rename or map your columns to these names.")
        path = _ask("Path to your reformatted file (save it as JSONL first)")
        data["source"] = "local"
        data["path"] = path

    elif choice == 3:
        # Synthesize
        if is_image:
            print(
                "\n  Note: I can generate expected outputs but not the images themselves."
                "\n  You need real images — use option 1 or 2 to supply them."
            )
            # Fall back to skip
            data["source"] = "generate"
            data["count"] = 0
            data["_image_warning"] = True
        else:
            count_raw = input("\n  How many examples? [30]: ").strip()
            count = int(count_raw) if count_raw.isdigit() else 30
            seed = input("  Describe the kind of examples to generate (optional): ").strip()
            data["source"] = "generate"
            data["count"] = count
            if seed:
                data["seed_description"] = seed
    else:
        data["source"] = "local"
        data["path"] = f".data/{benchmark_name}/train.jsonl"

    return data


def _collect_target(benchmark_name: str) -> Dict[str, Any]:
    """Stage 3a: which AI system?"""
    print("\n── Stage 3a: Configure target ────────────────────────────────")
    options = [
        "My own API endpoint  (HTTP URL)",
        "A specific model  (OpenAI, Anthropic, Together, etc.)",
        "A local model  (OpenAI-compatible server, e.g. vLLM)",
        "Skip — I'll set this up later",
    ]
    choice = _choose("What AI system are you benchmarking?", options)

    target: Dict[str, Any] = {}

    if choice == 1:
        target["type"] = "api"
        target["url"] = _ask("Endpoint URL (e.g. https://api.example.com/extract)")
        template = input(
            "  Body template with {{field}} placeholders [leave blank for default]: "
        ).strip()
        target["body_template"] = template or '{"input": "{{text}}"}'
        response_path = input(
            "  Response path (dot notation to extract answer, e.g. 'data'): "
        ).strip()
        if response_path:
            target["response_path"] = response_path
        target["name"] = benchmark_name

    elif choice == 2:
        target["type"] = "model"
        provider_choice = _choose(
            "Which provider?",
            ["Together", "OpenAI", "Anthropic", "Google", "Other (type manually)"],
            default=1,
        )
        provider_map = {1: "together", 2: "openai", 3: "anthropic", 4: "google"}
        model_defaults = {
            "together": "moonshotai/Kimi-K2.5",
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-5",
            "google": "gemini-2.0-flash",
        }
        if provider_choice in provider_map:
            target["provider"] = provider_map[provider_choice]
        else:
            target["provider"] = _ask("Provider name")
        model_hint = model_defaults.get(target["provider"], "moonshotai/Kimi-K2.5")
        target["model"] = _ask(f"Model name", default=model_hint)
        sp = input("  System prompt (optional, press Enter to skip): ").strip()
        if sp:
            target["system_prompt"] = sp

    elif choice == 3:
        target["type"] = "local"
        target["url"] = _ask("Server URL (e.g. http://localhost:8000/v1)")
        target["model"] = _ask("Model name (as declared by the server)")

    else:
        target["type"] = "model"
        target["provider"] = "together"
        target["model"] = "moonshotai/Kimi-K2.5"
        target["_placeholder"] = True

    return target


# ---------------------------------------------------------------------------
# Wizard entry point
# ---------------------------------------------------------------------------

def run_create_wizard(output_path: Optional[str] = None) -> int:
    """Interactive wizard that produces a benchmark spec file.

    Args:
        output_path: Where to write the spec. Defaults to benchmarks/<name>.yaml
                     after the wizard collects the benchmark name.

    Returns:
        0 on success, 1 on error.
    """
    _box(
        ["Define a benchmark for your AI system", "", "Four steps. No code required."],
        title="benchy create",
    )

    name_raw = _ask("\nBenchmark name (used for output folders, no spaces)", default="my-benchmark")
    name = name_raw.strip().replace(" ", "-").lower()

    # Resolve output path now that we have the name
    if output_path is None:
        output_path = f"benchmarks/{name}.yaml"

    description = _ask("Short description (one sentence)")

    task = _collect_task()
    scoring = _collect_scoring(task)
    target = _collect_target(name)
    data = _collect_data(name, task)

    spec = {
        "benchmark": {
            "name": name,
            "description": description,
            "task": task,
            "scoring": scoring,
            "data": data,
            "target": target,
        }
    }

    # Write the spec file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        overwrite = input(f"\n  {output_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite not in ("y", "yes"):
            print("  Aborted.")
            return 1

    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(spec, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Synthesize data now if requested
    generated_path = ""
    if data.get("source") == "generate" and data.get("count", 0) > 0:
        try:
            from .data_generator import generate_data
            count = data.get("count", 30)
            print(f"\n  Generating {count} examples...")
            generated_path = generate_data(output_path, count=count)
            # Update benchmark.yaml to point to generated data
            spec["benchmark"]["data"]["path"] = generated_path
            spec["benchmark"]["data"]["source"] = "local"
            with open(out, "w", encoding="utf-8") as f:
                yaml.dump(spec, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            print(f"\n  Warning: Data generation failed: {exc}")
            print("  You can run it later with: benchy synthesize-data")

    # Summary box
    fields = task.get("output", {}).get("fields", [])
    field_summary = f"{len(fields)} fields" if fields else task.get("output", {}).get("type", "")
    scoring_label = {
        "per_field": "per-field",
        "binary": "pass/fail",
        "semantic": "fuzzy (semantic)",
    }.get(scoring.get("type", ""), scoring.get("type", ""))
    data_label = (
        generated_path or data.get("path", "not set yet")
    )
    target_label = (
        target.get("url")
        or f"{target.get('provider', '')} / {target.get('model', '')}"
        or "not set"
    )

    summary_lines = [
        f"Name:     {name}",
        f"Task:     {task.get('type')} ({field_summary})",
        f"Scoring:  {scoring_label}",
        f"Data:     {data_label}",
        f"Target:   {target_label}",
        "",
        f"File:     {output_path}",
        "",
        "Next — smoke test (5 examples):",
        f"  benchy eval --benchmark {output_path} --limit 5 --exit-policy smoke",
    ]
    _box(summary_lines, title="✓ Benchmark defined")
    return 0
