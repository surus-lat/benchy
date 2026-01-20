"""Implementation of `benchy eval`."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .config_loader import load_config, resolve_config_path
from .config_manager import ConfigManager
from .gpu_config import load_gpu_config
from .logging_utils import setup_file_logging
from .run_id_manager import (
    generate_run_id,
    get_prefect_flow_name,
    get_run_paths,
    setup_run_directories,
)
from .signal_utils import register_signal_handlers
from .inference.vllm_config import VLLMServerConfig


logger = logging.getLogger(__name__)


PROVIDER_SPECS = {
    "vllm": {"config_key": "vllm"},
    "openai": {
        "config_key": "openai",
        "log": "Using OpenAI cloud provider for model: {model_name}",
    },
    "anthropic": {
        "config_key": "anthropic",
        "log": "Using Anthropic cloud provider for model: {model_name}",
    },
    "surus": {
        "config_key": "surus",
        "log": "Using SURUS AI provider for extraction tasks",
    },
    "surus_ocr": {
        "config_key": "surus_ocr",
        "log": "Using SURUS AI OCR provider for image extraction tasks",
    },
    "surus_factura": {
        "config_key": "surus_factura",
        "log": "Using SURUS AI Factura provider for image extraction tasks",
    },
    "together": {
        "config_key": "together",
        "log": "Using Together AI cloud provider for model: {model_name}",
    },
}


def _parse_tasks_arg(value: Optional[str]) -> list:
    if not value:
        return []
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _load_tasks_file(path: str) -> list:
    tasks = []
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tasks.append(line)
    return tasks


def _dedupe_tasks(tasks: list) -> list:
    seen = set()
    ordered = []
    for task_name in tasks:
        if task_name in seen:
            continue
        seen.add(task_name)
        ordered.append(task_name)
    return ordered


def resolve_provider_config(
    config: dict,
    provider_type: str,
    model_name: str,
    gpu_manager,
):
    """Resolve provider config and vLLM server config for a provider type."""
    provider_spec = PROVIDER_SPECS.get(provider_type)
    if not provider_spec:
        raise ValueError(f"Unknown provider type: {provider_type}")

    provider_config = config.get(provider_spec["config_key"], {})
    log_message = provider_spec.get("log")
    if log_message:
        logger.info(log_message.format(model_name=model_name))

    vllm_server_config = None
    if provider_type == "vllm":
        cuda_devices = provider_config.get("cuda_devices", gpu_manager.get_vllm_cuda_devices())
        vllm_server_config = VLLMServerConfig.from_config(
            provider_config,
            cuda_devices=cuda_devices,
        )

    return provider_config, vllm_server_config


def _normalize_provider_name(name: str) -> str:
    raw = (name or "").strip()
    if raw.endswith((".yaml", ".yml")):
        raw = raw.rsplit(".", 1)[0]
    return raw


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config_ref",
        nargs="?",
        default=None,
        help=(
            "Optional config name/path (same as --config). If it's just a name, benchy searches "
            "configs/models, configs/systems, configs/tests, and configs/."
        ),
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config name or path to YAML file (default: BENCHY_CONFIG env var)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run test pipeline (only start and test vLLM server, no evaluation)",
    )
    parser.add_argument(
        "--register",
        "-r",
        action="store_true",
        help="Register flows with Prefect server for dashboard visibility",
    )
    parser.add_argument(
        "--prefect-url",
        type=str,
        default="http://localhost:4200/api",
        help="Prefect API URL (default: http://localhost:4200/api)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--log-samples", action="store_true", help="Enable sample logging for all tasks")
    parser.add_argument("--no-log-samples", action="store_true", help="Disable sample logging for all tasks")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for organizing outputs")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks or task groups (overrides config tasks)",
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="Path to a task list file (one task per line, overrides config tasks)",
    )
    parser.add_argument(
        "--task-group",
        action="append",
        default=None,
        help="Task group name(s) from configs/config.yaml (can be repeated)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model directory to load via vLLM (e.g. merged safetensors + tokenizer assets)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Model identifier. With --model-path, this is the run alias used for outputs/requests; "
            "without --model-path, this is treated as a Hugging Face model ID to load via vLLM "
            "(e.g. unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit)."
        ),
    )
    parser.add_argument(
        "--vllm-config",
        type=str,
        default=None,
        help="Provider config name under configs/providers (e.g. vllm_two_cards_mm)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Override the benchmark output base directory (default: configs/config.yaml paths.benchmark_outputs). "
            "If set to 'model' and --model-path is provided, outputs go under <model-path>/benchy_outputs."
        ),
    )


def _load_or_build_config(args: argparse.Namespace) -> tuple[dict, Optional[str]]:
    """Return (merged_config, model_path_override)."""
    config_manager = ConfigManager()

    config_ref = args.config or args.config_ref or os.environ.get("BENCHY_CONFIG")
    config: Optional[dict] = None
    if config_ref:
        resolved_config_path = str(resolve_config_path(config_ref))
        try:
            config = config_manager.load_model_config(resolved_config_path)
            logger.info("Loaded configuration from %s using ConfigManager", resolved_config_path)
        except (FileNotFoundError, KeyError) as exc:
            logger.info("Falling back to legacy config loading: %s", exc)
            config = load_config(resolved_config_path)
            logger.info("Loaded legacy configuration from %s", resolved_config_path)

    model_path_override = args.model_path
    if model_path_override:
        model_path = Path(model_path_override).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"--model-path does not exist: {model_path}")
        model_path_override = str(model_path)

        config = dict(config or {})
        model_section = dict((config.get("model") or {}))

        if args.model_name:
            model_section["name"] = args.model_name
        elif model_section.get("name"):
            pass
        else:
            model_section["name"] = model_path.name

        config["model"] = model_section
        config["provider_type"] = "vllm"

        vllm_provider_name = args.vllm_config
        if not vllm_provider_name and "vllm" not in config:
            vllm_provider_name = os.environ.get("BENCHY_VLLM_CONFIG", "vllm_two_cards_mm")

        if vllm_provider_name:
            provider_config = config_manager.get_provider_config(_normalize_provider_name(vllm_provider_name))
            config["vllm"] = provider_config
    elif args.model_name and not config_ref:
        # Shortcut path: model id + provider config, without a model YAML.
        config = dict(config or {})
        config["model"] = {"name": args.model_name}
        config["provider_type"] = "vllm"

        vllm_provider_name = args.vllm_config or os.environ.get("BENCHY_VLLM_CONFIG", "vllm_two_cards_mm")
        provider_config = config_manager.get_provider_config(_normalize_provider_name(vllm_provider_name))
        config["vllm"] = provider_config

    if not config:
        raise ValueError(
            "No model configuration provided. Pass `--config <model.yaml>` or "
            "`--model-path <local_model_dir>`, or `--model-name <hf_model_id>`."
        )

    # Optional vLLM provider override even when using a model config.
    if args.vllm_config and not args.model_path:
        existing_provider_type = config.get("provider_type")
        if existing_provider_type and existing_provider_type != "vllm":
            raise ValueError("--vllm-config is only valid for vLLM runs (or when using --model-path).")
        provider_config = config_manager.get_provider_config(_normalize_provider_name(args.vllm_config))
        config["vllm"] = provider_config
        config["provider_type"] = "vllm"

    model_name = (config.get("model") or {}).get("name")
    if not model_name:
        raise KeyError("Missing required config key: model.name")

    return config, model_path_override


def run_eval(args: argparse.Namespace) -> int:
    load_dotenv()

    if args.log_samples and args.no_log_samples:
        raise SystemExit("Cannot specify both --log-samples and --no-log-samples")

    prefect_enabled = args.register or os.environ.get("BENCHY_ENABLE_PREFECT", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if prefect_enabled:
        os.environ.pop("BENCHY_DISABLE_PREFECT", None)
        if args.prefect_url:
            os.environ["PREFECT_API_URL"] = args.prefect_url
        elif "PREFECT_API_URL" not in os.environ:
            os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    else:
        os.environ.setdefault("BENCHY_DISABLE_PREFECT", "1")

    # Prefect-dependent imports must happen after BENCHY_DISABLE_PREFECT is set.
    from .prefect_compat import PREFECT_AVAILABLE, serve
    from .pipeline import benchmark_pipeline, test_vllm_server

    register_signal_handlers()

    logger.info("Starting benchy eval")

    if args.verbose:
        logger.info("Command line arguments: %s", args)
        logger.info("Working directory: %s", os.getcwd())

    config, model_path_override = _load_or_build_config(args)
    config_manager = ConfigManager()

    central_config = load_config("configs/config.yaml")
    gpu_manager = load_gpu_config(central_config)

    base_output_path = central_config["paths"]["benchmark_outputs"]
    if args.output_path:
        if args.output_path.strip().lower() == "model":
            if not model_path_override:
                raise SystemExit("--output-path model requires --model-path")
            base_output_path = str(Path(model_path_override) / "benchy_outputs")
        else:
            base_output_path = os.path.expanduser(os.path.expandvars(args.output_path))

    run_id = generate_run_id(
        custom_run_id=args.run_id,
        is_test=args.test,
        is_limited=args.limit is not None,
    )

    run_paths = get_run_paths(
        run_id,
        base_output_path,
        central_config["logging"]["log_dir"],
    )
    setup_run_directories(run_paths)

    log_setup = setup_file_logging(config, central_config["logging"]["log_dir"], run_id)
    log_setup.log_config()

    model_config = config.get("model", {}) or {}
    model_name = model_config["name"]
    organization = model_config.get("organization")
    url = model_config.get("url")
    provider_type = config.get("provider_type", "vllm")

    provider_config, vllm_server_config = resolve_provider_config(
        config=config,
        provider_type=provider_type,
        model_name=model_name,
        gpu_manager=gpu_manager,
    )

    api_endpoint = provider_config.get("api_endpoint", config.get("api_endpoint", "completions"))

    task_defaults_overrides = {}
    if args.log_samples:
        task_defaults_overrides["log_samples"] = True
    elif args.no_log_samples:
        task_defaults_overrides["log_samples"] = False

    config_task_defaults = config.get("task_defaults", {}) or {}
    if config_task_defaults:
        task_defaults_overrides = {**config_task_defaults, **task_defaults_overrides}

    config_tasks = config.get("tasks", ["spanish", "portuguese"])
    tasks_override = []
    if args.tasks:
        tasks_override.extend(_parse_tasks_arg(args.tasks))
    if args.tasks_file:
        tasks_override.extend(_load_tasks_file(args.tasks_file))
    if args.task_group:
        for group_entry in args.task_group:
            tasks_override.extend(_parse_tasks_arg(group_entry))
    tasks_override = _dedupe_tasks(tasks_override)

    model_provider_types = {"vllm", "openai", "anthropic", "together"}
    is_system_provider = provider_type not in model_provider_types

    if tasks_override:
        if is_system_provider and config_tasks:
            allowed = []
            disallowed = []
            for task in tasks_override:
                if task in config_tasks:
                    allowed.append(task)
                elif "." in task and task.split(".")[0] in config_tasks:
                    allowed.append(task)
                else:
                    disallowed.append(task)
            if disallowed:
                logger.warning("Ignoring tasks not declared in system config: %s", disallowed)
            tasks_to_run = allowed
        else:
            tasks_to_run = tasks_override
    else:
        tasks_to_run = config_tasks

    if not tasks_to_run:
        raise SystemExit("No tasks to run after applying task overrides.")

    tasks_to_run = config_manager.expand_task_groups(tasks_to_run, central_config)

    if args.register:
        if not PREFECT_AVAILABLE:
            raise SystemExit(
                "Prefect is disabled or not installed; install with `pip install .[prefect]` "
                "and unset BENCHY_DISABLE_PREFECT to register flows."
            )
        logger.info("Registering flows with Prefect server for dashboard visibility...")
        benchmark_pipeline.name = get_prefect_flow_name("benchmark_pipeline", run_id)
        test_vllm_server.name = get_prefect_flow_name("test_vllm_server", run_id)
        serve(
            benchmark_pipeline.to_deployment(name=benchmark_pipeline.name),
            test_vllm_server.to_deployment(name=test_vllm_server.name),
            limit=1,
            print_starting_message=True,
        )
        return 0

    if args.test:
        if provider_type != "vllm":
            raise SystemExit(f"Test mode is only supported for vLLM provider, not {provider_type}")
        if vllm_server_config is None:
            vllm_server_config = VLLMServerConfig()
        test_vllm_server(
            model_name=model_name,
            model_path=model_path_override,
            run_id=run_id,
            vllm_config=vllm_server_config,
        )
        return 0

    benchmark_pipeline(
        model_name=model_name,
        model_path=model_path_override,
        tasks=tasks_to_run,
        output_path=base_output_path,
        limit=args.limit,
        api_endpoint=api_endpoint,
        task_defaults_overrides=task_defaults_overrides or None,
        log_setup=log_setup,
        run_id=run_id,
        provider_type=provider_type,
        provider_config=provider_config,
        organization=organization,
        url=url,
        vllm_config=vllm_server_config,
    )
    return 0
