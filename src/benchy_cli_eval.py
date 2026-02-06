"""Implementation of `benchy eval`."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

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
    "surus_classify": {
        "config_key": "surus_classify",
        "log": "Using SURUS AI classify provider for classification tasks",
    },
    "surus_remove_background": {
        "config_key": "surus_remove_background",
        "log": "Using SURUS AI remove-background provider for image manipulation tasks",
    },
    "together": {
        "config_key": "together",
        "log": "Using Together AI cloud provider for model: {model_name}",
    },
    "google": {
        "config_key": "google",
        "log": "Using Google cloud provider for model: {model_name}",
    },
}

MODEL_PROVIDER_TYPES = {"vllm", "openai", "anthropic", "together", "google"}
CLI_PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout": 120,
        "max_retries": 3,
        "max_concurrent": 3,
        "temperature": 1.0,
        "max_tokens": 2048,
        "max_tokens_param_name": "max_tokens",
        "api_endpoint": "auto",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "timeout": 120,
        "max_retries": 3,
        "max_concurrent": 3,
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_tokens_param_name": "max_tokens",
        "api_endpoint": "auto",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "timeout": 120,
        "max_retries": 3,
        "max_concurrent": 3,
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_tokens_param_name": "max_tokens",
        "api_endpoint": "chat",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "api_key_env": "GOOGLE_API_KEY",
        "timeout": 60,
        "max_retries": 3,
        "max_concurrent": 3,
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_tokens_param_name": "max_tokens",
        "api_endpoint": "auto",
    },
}


def _parse_tasks_arg(value: Optional[Any]) -> list:
    if not value:
        return []

    if isinstance(value, str):
        tokens = [value]
    elif isinstance(value, list):
        tokens = [str(token) for token in value if token is not None]
    else:
        tokens = [str(value)]

    parsed = []
    for token in tokens:
        parsed.extend(entry.strip() for entry in token.split(",") if entry.strip())
    return parsed


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


def _redact_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a log-safe copy of CLI args."""
    payload = dict(vars(args))
    if payload.get("api_key"):
        payload["api_key"] = "***"
    return payload


def _default_api_endpoint(provider_type: str) -> str:
    if provider_type == "anthropic":
        return "chat"
    if provider_type in {"openai", "together"}:
        return "auto"
    return "completions"


def _apply_cli_provider_overrides(
    provider_config: Dict[str, Any],
    args: argparse.Namespace,
    *,
    allow_base_url: bool,
    allow_api_key: bool,
) -> Dict[str, Any]:
    merged = dict(provider_config or {})

    if args.base_url is not None:
        if not allow_base_url:
            raise ValueError("--base-url is only valid for OpenAI-compatible providers")
        merged["base_url"] = args.base_url
    if args.api_key_env is not None:
        if not allow_api_key:
            raise ValueError("--api-key-env is only valid for OpenAI-compatible providers")
        merged["api_key_env"] = args.api_key_env
    if args.api_key is not None:
        if not allow_api_key:
            raise ValueError("--api-key is only valid for OpenAI-compatible providers")
        merged["api_key"] = args.api_key

    if args.timeout is not None:
        merged["timeout"] = args.timeout
    if args.max_retries is not None:
        merged["max_retries"] = args.max_retries
    if args.max_concurrent is not None:
        merged["max_concurrent"] = args.max_concurrent
    if args.temperature is not None:
        merged["temperature"] = args.temperature
    if args.max_tokens is not None:
        merged["max_tokens"] = args.max_tokens
    if args.max_tokens_param_name is not None:
        merged["max_tokens_param_name"] = args.max_tokens_param_name
    if args.api_endpoint is not None:
        merged["api_endpoint"] = args.api_endpoint
    if args.image_max_edge is not None:
        if args.image_max_edge <= 0:
            raise ValueError("--image-max-edge must be a positive integer")
        merged["image_max_edge"] = args.image_max_edge

    return merged


def _build_cli_provider_config(provider_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    defaults = dict(CLI_PROVIDER_DEFAULTS.get(provider_type, {}))
    merged = _apply_cli_provider_overrides(
        defaults,
        args,
        allow_base_url=True,
        allow_api_key=True,
    )
    if "api_endpoint" not in merged:
        merged["api_endpoint"] = _default_api_endpoint(provider_type)
    return merged


def _apply_model_metadata(config: Dict[str, Any], args: argparse.Namespace) -> None:
    model_section = dict(config.get("model") or {})
    if args.organization is not None:
        model_section["organization"] = args.organization
    if args.url is not None:
        model_section["url"] = args.url
    config["model"] = model_section


def _build_cli_only_config(
    args: argparse.Namespace,
    *,
    config_manager: ConfigManager,
) -> tuple[Dict[str, Any], Optional[str]]:
    if args.provider:
        provider_type = args.provider
    elif args.model_path or args.vllm_config:
        provider_type = "vllm"
    elif args.base_url:
        provider_type = "openai"
    else:
        provider_type = "openai"

    if args.model_path and provider_type != "vllm":
        raise ValueError("--model-path can only be used with the vLLM provider")
    if args.model_path and args.base_url:
        raise ValueError("--model-path cannot be combined with --base-url")
    if provider_type == "vllm" and args.base_url:
        raise ValueError("--base-url is not valid when running a local vLLM server")
    if provider_type != "vllm" and args.vllm_config:
        raise ValueError("--vllm-config is only valid for vLLM runs")

    model_path_override = None
    config: Dict[str, Any] = {}

    if provider_type == "vllm":
        model_path_override = args.model_path
        if model_path_override:
            model_path = Path(model_path_override).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"--model-path does not exist: {model_path}")
            model_path_override = str(model_path)

        model_name = args.model_name
        if not model_name:
            if model_path_override:
                model_name = Path(model_path_override).name
            else:
                raise ValueError("Missing required --model-name for vLLM runs without --model-path")

        config["model"] = {"name": model_name}
        config["provider_type"] = "vllm"

        vllm_provider_name = args.vllm_config
        if not vllm_provider_name:
            vllm_provider_name = os.environ.get("BENCHY_VLLM_CONFIG", "vllm_two_cards_mm")

        provider_config: Dict[str, Any] = {}
        if vllm_provider_name:
            provider_config = config_manager.get_provider_config(
                _normalize_provider_name(vllm_provider_name)
            )
        provider_config = _apply_cli_provider_overrides(
            provider_config,
            args,
            allow_base_url=False,
            allow_api_key=False,
        )
        if "api_endpoint" not in provider_config:
            provider_config["api_endpoint"] = _default_api_endpoint("vllm")
        config["vllm"] = provider_config
    else:
        if not args.model_name:
            raise ValueError("Missing required --model-name for OpenAI-compatible runs")
        config["model"] = {"name": args.model_name}
        config["provider_type"] = provider_type
        config[provider_type] = _build_cli_provider_config(provider_type, args)

    _apply_model_metadata(config, args)
    return config, model_path_override


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for task runners (default: task default, usually 20).",
    )
    parser.add_argument("--log-samples", action="store_true", help="Enable sample logging for all tasks")
    parser.add_argument("--no-log-samples", action="store_true", help="Disable sample logging for all tasks")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for organizing outputs")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Task/task-group overrides (space-separated and/or comma-separated). "
            "Examples: --tasks spanish portuguese OR --tasks spanish,portuguese"
        ),
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
            "Model identifier used in requests and outputs. With --model-path or --provider vllm, "
            "this is the served model name/alias. For OpenAI-compatible endpoints, this is the "
            "model name sent in API requests (e.g. gpt-4o-mini)."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["vllm", "openai", "together", "anthropic", "google"],
        default=None,
        help="Provider type to use when no model config is provided (default: inferred).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (e.g. https://api.openai.com/v1 or http://host:8000/v1).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable name to read API key from (e.g. OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key value (discouraged; not written to run artifacts).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retry attempts for API requests.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent API requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (overrides provider default).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (overrides provider default).",
    )
    parser.add_argument(
        "--max-tokens-param-name",
        type=str,
        default=None,
        help="Parameter name for max tokens (e.g. max_tokens or max_completion_tokens).",
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        choices=["auto", "chat", "completions"],
        default=None,
        help="Request mode for OpenAI-compatible providers (default: auto).",
    )
    parser.add_argument(
        "--image-max-edge",
        type=int,
        default=None,
        help=(
            "Optionally downscale input images before sending requests. "
            "Sets maximum image width/height in pixels while preserving aspect ratio."
        ),
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Optional organization metadata for run artifacts.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional URL metadata for run artifacts.",
    )
    parser.add_argument(
        "--vllm-config",
        type=str,
        default=None,
        help="Provider config name under configs/providers (e.g. vllm_two_cards_mm)",
    )
    parser.add_argument(
        "--compatibility",
        type=str,
        choices=["warn", "skip", "error"],
        default=None,
        help="Compatibility handling for incompatible tasks (default: warn for CLI-only, skip for config).",
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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to use for tasks (e.g., ICM57, kapaxia). Task must support dataset configuration.",
    )


def _load_or_build_config(args: argparse.Namespace) -> tuple[dict, Optional[str], bool]:
    """Return (merged_config, model_path_override, used_config_file)."""
    config_manager = ConfigManager()

    config_ref = args.config or args.config_ref or os.environ.get("BENCHY_CONFIG")
    config: Optional[dict] = None
    used_config_file = False
    if config_ref:
        resolved_config_path = str(resolve_config_path(config_ref))
        try:
            config = config_manager.load_model_config(resolved_config_path)
            logger.info("Loaded configuration from %s using ConfigManager", resolved_config_path)
        except (FileNotFoundError, KeyError) as exc:
            logger.info("Falling back to legacy config loading: %s", exc)
            config = load_config(resolved_config_path)
            logger.info("Loaded legacy configuration from %s", resolved_config_path)
        used_config_file = True

    if not config:
        config, model_path_override = _build_cli_only_config(args, config_manager=config_manager)
        return config, model_path_override, used_config_file

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

    if not config:
        raise ValueError(
            "No model configuration provided. Pass `--config <model.yaml>`, `--model-path <local_model_dir>`, "
            "or `--model-name <model_id>` for OpenAI-compatible runs."
        )

    # Optional vLLM provider override even when using a model config.
    if args.vllm_config and not args.model_path:
        existing_provider_type = config.get("provider_type")
        if existing_provider_type and existing_provider_type != "vllm":
            raise ValueError("--vllm-config is only valid for vLLM runs (or when using --model-path).")
        provider_config = config_manager.get_provider_config(_normalize_provider_name(args.vllm_config))
        config["vllm"] = provider_config
        config["provider_type"] = "vllm"

    if args.provider:
        existing_provider_type = config.get("provider_type")
        if existing_provider_type and existing_provider_type != args.provider:
            raise ValueError(
                f"--provider {args.provider} conflicts with config provider_type {existing_provider_type}."
            )
        config["provider_type"] = args.provider

    provider_type = config.get("provider_type", "vllm")
    provider_section = dict(config.get(provider_type) or {})
    allow_base_url = provider_type in {"openai", "together", "anthropic", "google"}
    allow_api_key = allow_base_url
    config[provider_type] = _apply_cli_provider_overrides(
        provider_section,
        args,
        allow_base_url=allow_base_url,
        allow_api_key=allow_api_key,
    )

    _apply_model_metadata(config, args)

    model_name = (config.get("model") or {}).get("name")
    if not model_name:
        raise KeyError("Missing required config key: model.name")

    return config, model_path_override, used_config_file


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
        logger.info("Command line arguments: %s", _redact_args(args))
        logger.info("Working directory: %s", os.getcwd())

    config, model_path_override, used_config_file = _load_or_build_config(args)
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
    # Handle log_samples flag
    if args.log_samples:
        # Explicitly enabled via CLI
        task_defaults_overrides["log_samples"] = True
    elif args.no_log_samples:
        # Explicitly disabled via CLI
        task_defaults_overrides["log_samples"] = False
    elif not used_config_file:
        # Default to False when running from CLI without a config file
        # (configs can still override this in their task_defaults)
        task_defaults_overrides["log_samples"] = False
    # If used_config_file and no explicit CLI flag, let config decide
    
    if args.batch_size is not None:
        task_defaults_overrides["batch_size"] = args.batch_size
    
    # Handle dataset override
    if args.dataset is not None:
        if "dataset" not in task_defaults_overrides:
            task_defaults_overrides["dataset"] = {}
        task_defaults_overrides["dataset"]["name"] = args.dataset

    provider_task_defaults = {}
    if isinstance(provider_config, dict):
        provider_task_defaults = provider_config.get("task_defaults", {}) or {}

    config_task_defaults = config.get("task_defaults", {}) or {}
    if config_task_defaults:
        task_defaults_overrides = {**provider_task_defaults, **config_task_defaults, **task_defaults_overrides}
    elif provider_task_defaults:
        task_defaults_overrides = {**provider_task_defaults, **task_defaults_overrides}

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

    is_system_provider = provider_type not in MODEL_PROVIDER_TYPES

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

    compatibility_mode = args.compatibility
    if compatibility_mode is None:
        compatibility_mode = "skip" if used_config_file else "warn"

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
        compatibility_mode=compatibility_mode,
        organization=organization,
        url=url,
        vllm_config=vllm_server_config,
    )
    return 0
