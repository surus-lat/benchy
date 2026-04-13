"""Benchy CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .config_loader import load_config
from .config_manager import ConfigManager


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=False))


def _cmd_tasks(args: argparse.Namespace) -> int:
    config_manager = ConfigManager()
    central_config = load_config("configs/config.yaml")

    task_groups: Dict[str, Any] = central_config.get("task_groups", {}) or {}
    tasks: List[str] = config_manager.list_available_tasks()

    if args.json:
        payload = {
            "task_groups": task_groups,
            "tasks": tasks,
        }
        _print_json(payload)
        return 0

    if task_groups:
        print("Task groups:")
        for name, group in sorted(task_groups.items()):
            description = (group or {}).get("description", "")
            group_tasks = (group or {}).get("tasks", []) or []
            suffix = f" - {description}" if description else ""
            print(f"  - {name}{suffix}")
            if args.verbose:
                for task in group_tasks:
                    print(f"      - {task}")
        print("")

    print("Tasks:")
    discover_task_group = None
    if args.verbose:
        from .tasks.registry import discover_task_group

    for task in tasks:
        print(f"  - {task}")
        if args.verbose and discover_task_group:
            group_info = discover_task_group(task)
            if group_info:
                for subtask in group_info.subtasks:
                    print(f"      - {task}.{subtask.name}")
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    config_manager = ConfigManager()
    providers = config_manager.list_available_providers()
    if args.json:
        _print_json({"providers": providers})
        return 0
    for provider in providers:
        print(provider)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    config_manager = ConfigManager()
    models = config_manager.list_available_models()
    if args.json:
        _print_json({"models": models})
        return 0
    for model in models:
        print(model)
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    from .tasks.common.dataset_adapters import list_data_datasets

    datasets = list_data_datasets()
    if not datasets:
        print("No datasets found in .data/")
        return 0

    if args.json:
        _print_json({"datasets": datasets})
        return 0

    for ds in datasets:
        name = ds["name"]
        desc = ds.get("description", "")
        schema_tag = " [schema]" if ds.get("has_schema") else ""
        # Summarise splits
        splits = ds.get("splits", {})
        if splits:
            split_parts = [f"{s}={info['num_rows']}" for s, info in splits.items() if isinstance(info, dict) and "num_rows" in info]
            split_str = ", ".join(split_parts)
        else:
            split_str = ""

        print(f"  {name}{schema_tag}")
        if desc and args.verbose:
            # Truncate long descriptions
            if len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"    {desc}")
        if split_str:
            print(f"    splits: {split_str}")

        labels = ds.get("label_distribution")
        if labels and args.verbose:
            label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
            print(f"    labels: {label_str}")

        if ds.get("features") and args.verbose:
            print(f"    features: {', '.join(ds['features'])}")
        print()

    return 0


def _discover_benchmark_candidates() -> List[str]:
    """Scan for benchmark spec files in standard locations."""
    from pathlib import Path
    candidates = []
    if Path("benchmark.yaml").exists():
        candidates.append("benchmark.yaml")
    if Path("benchmarks").exists():
        candidates.extend(sorted(str(p) for p in Path("benchmarks").glob("*.yaml")))
    return candidates


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    import yaml
    from pathlib import Path

    candidates = _discover_benchmark_candidates()
    if not candidates:
        print("No benchmark specs found.")
        print("Run 'benchy create' to create one.")
        return 0

    results = []
    for p in candidates:
        try:
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
            spec = raw.get("benchmark", raw)
            results.append({
                "path": p,
                "name": spec.get("name", "(no name)"),
                "description": spec.get("description", ""),
            })
        except Exception:
            results.append({"path": p, "name": "(unreadable)", "description": ""})

    if args.json:
        _print_json({"benchmarks": results})
        return 0

    for r in results:
        desc = f"  {r['description']}" if r.get("description") else ""
        print(f"  {r['name']}  ({r['path']}){desc}")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    from .benchy_create import run_create_wizard
    return run_create_wizard(output_path=args.output)


def _cmd_validate(args: argparse.Namespace) -> int:
    from .benchmark_compiler import validate_benchmark_yaml

    path = args.benchmark
    if path is None:
        candidates = _discover_benchmark_candidates()
        if len(candidates) == 1:
            path = candidates[0]
        elif len(candidates) > 1:
            print("Multiple benchmark specs found. Pass --benchmark to choose one:\n")
            for c in candidates:
                print(f"  benchy validate --benchmark {c}")
            return 1
        else:
            print("No benchmark spec found. Run 'benchy create' or pass --benchmark <path>.")
            return 1

    errors = validate_benchmark_yaml(path)
    if not errors:
        print(f"✓ {path} is valid and ready to run.")
        print(f"\n  Run a smoke test:")
        print(f"  benchy eval --benchmark {path} --limit 5 --exit-policy smoke")
        return 0
    print(f"✗ {path} has {len(errors)} error(s):\n")
    for err in errors:
        print(f"  • {err}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchy",
        description="Benchy - vLLM-powered ML benchmarking suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval
    from .benchy_cli_eval import add_eval_arguments, run_eval

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation (current eval.py flow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  benchy eval --model-name gpt-4o-mini --tasks spanish --limit 2\n"
            "  benchy eval --provider together --model-name meta-llama/Llama-3.1-8B-Instruct --tasks spanish --limit 2\n"
            "  benchy eval --base-url http://host:8000/v1 --model-name mymodel --tasks spanish --limit 2\n"
            "  benchy eval --config openai_gpt-4o-mini.yaml --limit 10\n"
            "  benchy eval --config configs/tests/spanish-gptoss.yaml --limit 2\n"
            "  benchy eval --provider vllm --model-name unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit --vllm-config vllm_two_cards_mm --tasks latam_board --limit 10\n"
            "  benchy eval --model-path /models/my-sft --model-name my-sft --vllm-config vllm_two_cards_mm --tasks latam_board\n"
            "  benchy eval --model-path /models/my-sft --output-path model --tasks latam_board\n"
            "  benchy eval --model-path /models/my-sft --limit 5 --tasks spanish,portuguese\n"
        ),
    )
    add_eval_arguments(eval_parser)
    eval_parser.set_defaults(_handler=run_eval)

    # probe
    from .benchy_cli_probe import add_probe_arguments, run_probe

    probe_parser = subparsers.add_parser(
        "probe",
        help="Probe model capabilities and compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  benchy probe --provider vllm --base-url http://localhost:8000/v1 --model-name mymodel\n"
            "  benchy probe --provider openai --model-name gpt-4o-mini\n"
            "  benchy probe --base-url http://localhost:8000/v1 --model-name meta-llama/Llama-3.1-8B-Instruct\n"
        ),
    )
    add_probe_arguments(probe_parser)
    probe_parser.set_defaults(_handler=run_probe)

    # tasks
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="List available tasks and task groups",
    )
    tasks_parser.add_argument("--json", action="store_true", help="Output as JSON")
    tasks_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include group task expansion",
    )
    tasks_parser.set_defaults(_handler=_cmd_tasks)

    # providers
    providers_parser = subparsers.add_parser(
        "providers",
        help="List available providers (from configs/providers)",
    )
    providers_parser.add_argument("--json", action="store_true", help="Output as JSON")
    providers_parser.set_defaults(_handler=_cmd_providers)

    # models
    models_parser = subparsers.add_parser(
        "models",
        help="List available models (from configs/models)",
    )
    models_parser.add_argument("--json", action="store_true", help="Output as JSON")
    models_parser.set_defaults(_handler=_cmd_models)

    # datasets
    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List datasets discovered in .data/",
    )
    datasets_parser.add_argument("--json", action="store_true", help="Output as JSON")
    datasets_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show descriptions, features, and label distributions",
    )
    datasets_parser.set_defaults(_handler=_cmd_datasets)

    # benchmarks — list all benchmark specs in this project
    benchmarks_parser = subparsers.add_parser(
        "benchmarks",
        help="List benchmark specs found in this project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scans benchmark.yaml at the project root and benchmarks/*.yaml.\n\n"
            "Examples:\n"
            "  benchy benchmarks\n"
            "  benchy benchmarks --json\n"
        ),
    )
    benchmarks_parser.add_argument("--json", action="store_true", help="Output as JSON")
    benchmarks_parser.set_defaults(_handler=_cmd_benchmarks)

    # create — interactive wizard that produces a benchmark spec
    create_parser = subparsers.add_parser(
        "create",
        help="Create a benchmark spec with the interactive wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "By default writes to benchmarks/<name>.yaml.\n\n"
            "Examples:\n"
            "  benchy create\n"
            "  benchy create --output benchmarks/my-benchmark.yaml\n"
            "  benchy create --output benchmark.yaml  # single-file layout\n"
        ),
    )
    create_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for the benchmark spec. "
            "Defaults to benchmarks/<name>.yaml after the wizard collects the name."
        ),
    )
    create_parser.set_defaults(_handler=_cmd_create)

    # validate — pre-flight check for a benchmark spec
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a benchmark spec before running",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Auto-discovers the spec if --benchmark is omitted and exactly one exists.\n\n"
            "Examples:\n"
            "  benchy validate\n"
            "  benchy validate --benchmark benchmarks/my-benchmark.yaml\n"
        ),
    )
    validate_parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        metavar="YAML_PATH",
        help="Path to benchmark spec to validate (auto-discovers if omitted)",
    )
    validate_parser.set_defaults(_handler=_cmd_validate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error("No command handler registered")
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
