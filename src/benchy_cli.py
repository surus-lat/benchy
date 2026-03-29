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
