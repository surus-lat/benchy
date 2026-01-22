"""Shared runner for grouped benchmark tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..engine import BenchmarkRunner, build_connection_info, get_interface_for_provider, mark_task_complete, save_results
from ..engine.protocols import BaseTask, check_compatibility

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskGroupSpec:
    """Configuration for running a task group.

    Use this in each task's run wrapper to declare how subtasks are built and
    how results are aggregated. Provide either:
    - prepare_task (default runner), or
    - run_subtask (custom runner).

    Optional setup/teardown hooks let you load shared resources once and reuse
    them across subtasks via TaskGroupContext.shared.
    """

    name: str
    display_name: str
    output_subdir: Optional[str] = None
    supports_subtasks: bool = True
    default_subtasks: Optional[List[str]] = None
    prepare_task: Optional[Callable[["SubtaskContext"], Optional[BaseTask]]] = None
    run_subtask: Optional[Callable[["SubtaskContext"], Dict[str, Any]]] = None
    aggregate_metrics: Optional[Callable[[Dict[str, Dict[str, Any]], List[str]], Dict[str, Any]]] = None
    write_summary: Optional[
        Callable[[Dict[str, Any], Dict[str, Dict[str, Any]], Path, str, List[str]], None]
    ] = None
    setup: Optional[Callable[["TaskGroupContext"], Any]] = None
    teardown: Optional[Callable[["TaskGroupContext", Any], None]] = None


@dataclass
class TaskGroupContext:
    """Shared context for a task group run.

    This is passed to setup/teardown hooks to preload shared resources.
    """
    task_config: Dict[str, Any]
    defaults: Dict[str, Any]
    prompts: Dict[str, Any]
    model_name: str
    provider_type: str
    connection_info: Dict[str, Any]
    output_dir: Path
    limit: Optional[int]
    compatibility_mode: str
    shared: Any


@dataclass
class SubtaskContext:
    """Context for a single subtask execution."""
    subtask_name: str
    subtask_config: Dict[str, Any]
    task_config: Dict[str, Any]
    defaults: Dict[str, Any]
    prompts: Dict[str, Any]
    model_name: str
    provider_type: str
    connection_info: Dict[str, Any]
    output_dir: Path
    subtask_output_dir: Path
    limit: Optional[int]
    compatibility_mode: str
    shared: Any = None


def ensure_task_interface_compatibility(task: BaseTask, interface: Any):
    """Return a compatibility report for a task/interface pair."""
    report = check_compatibility(task, interface)
    for warning in report.warnings:
        logger.warning(warning)
    return report


def run_task_group(
    *,
    spec: TaskGroupSpec,
    model_name: str,
    output_path: str,
    server_info: Optional[Dict[str, Any]],
    task_config: Dict[str, Any],
    limit: Optional[int] = None,
    provider_config: Optional[Dict[str, Any]] = None,
    compatibility_mode: str = "skip",
) -> Dict[str, Any]:
    """Run a task group using a shared execution flow.

    Returns a consistent shape for all tasks:
    - metrics: aggregated task metrics
    - subtask_metrics: per-subtask aggregate metrics
    """
    logger.info(f"Starting {spec.display_name} evaluation for model: {model_name}")

    provider_type = "vllm"
    if provider_config:
        provider_type = provider_config.get("provider_type", "vllm")

    connection_info = build_connection_info(
        provider_type=provider_type,
        provider_config=provider_config or {},
        server_info=server_info,
        model_config=task_config.get("defaults", {}),
    )

    logger.info(f"Provider: {provider_type}")
    logger.info(f"Base URL: {connection_info.get('base_url')}")
    logger.info(f"Capabilities: {connection_info.get('capabilities')}")

    output_subdir = task_config.get("output", {}).get("subdirectory", spec.output_subdir or spec.name)
    task_output_path = Path(output_path) / output_subdir
    task_output_path.mkdir(parents=True, exist_ok=True)

    probe_report = _probe_and_configure_openai_interface(
        connection_info=connection_info,
        model_name=model_name,
        provider_type=provider_type,
        output_dir=task_output_path,
    )

    defaults = task_config.get("defaults", {})
    prompts = task_config.get("prompts", {})
    subtask_configs = task_config.get("task_configs", {})

    if spec.supports_subtasks:
        subtasks_to_run = task_config.get("tasks", None)
        if subtasks_to_run is None:
            subtasks_to_run = spec.default_subtasks or []
    else:
        subtasks_to_run = [spec.name]

    if not subtasks_to_run:
        raise ValueError(
            f"No subtasks configured for {spec.name}. Set tasks in the task config or default_subtasks in the spec."
        )

    shared_state: Any = {}
    group_context = TaskGroupContext(
        task_config=task_config,
        defaults=defaults,
        prompts=prompts,
        model_name=model_name,
        provider_type=provider_type,
        connection_info=connection_info,
        output_dir=task_output_path,
        limit=limit,
        compatibility_mode=compatibility_mode,
        shared=shared_state,
    )
    if probe_report:
        group_context.shared = {"compatibility_report": probe_report, **(shared_state or {})}
        shared_state = group_context.shared

    if spec.setup:
        setup_result = spec.setup(group_context)
        if setup_result is not None:
            shared_state = setup_result
            group_context.shared = shared_state
        else:
            shared_state = group_context.shared

    all_metrics: Dict[str, Dict[str, Any]] = {}
    skipped_subtasks: List[Dict[str, str]] = []

    async def _run_default_subtasks() -> None:
        """Run default subtask flow with a single event loop."""
        for subtask_name in subtasks_to_run:
            if spec.supports_subtasks:
                logger.info(f"Running subtask: {subtask_name}")
                subtask_config = subtask_configs.get(subtask_name, {})
                subtask_output_dir = task_output_path / subtask_name
            else:
                subtask_config = task_config
                subtask_output_dir = task_output_path

            context = SubtaskContext(
                subtask_name=subtask_name,
                subtask_config=subtask_config,
                task_config=task_config,
                defaults=defaults,
                prompts=prompts,
                model_name=model_name,
                provider_type=provider_type,
                connection_info=connection_info,
                output_dir=task_output_path,
                subtask_output_dir=subtask_output_dir,
                limit=limit,
                compatibility_mode=compatibility_mode,
                shared=shared_state,
            )

            subtask_results = await _run_default_subtask_async(spec, context)

            if subtask_results:
                if subtask_results.get("skipped"):
                    skipped_subtasks.append(
                        {
                            "subtask": subtask_name,
                            "reason": subtask_results.get("skip_reason", "incompatible"),
                        }
                    )
                all_metrics[subtask_name] = subtask_results.get("aggregate_metrics", {})

            logger.info(f"Subtask {subtask_name} completed")

    try:
        if spec.run_subtask:
            for subtask_name in subtasks_to_run:
                if spec.supports_subtasks:
                    logger.info(f"Running subtask: {subtask_name}")
                    subtask_config = subtask_configs.get(subtask_name, {})
                    subtask_output_dir = task_output_path / subtask_name
                else:
                    subtask_config = task_config
                    subtask_output_dir = task_output_path

                context = SubtaskContext(
                    subtask_name=subtask_name,
                    subtask_config=subtask_config,
                    task_config=task_config,
                    defaults=defaults,
                    prompts=prompts,
                    model_name=model_name,
                    provider_type=provider_type,
                    connection_info=connection_info,
                    output_dir=task_output_path,
                    subtask_output_dir=subtask_output_dir,
                    limit=limit,
                    compatibility_mode=compatibility_mode,
                    shared=shared_state,
                )

                subtask_results = spec.run_subtask(context)

                if subtask_results:
                    if subtask_results.get("skipped"):
                        skipped_subtasks.append(
                            {
                                "subtask": subtask_name,
                                "reason": subtask_results.get("skip_reason", "incompatible"),
                            }
                        )
                    all_metrics[subtask_name] = subtask_results.get("aggregate_metrics", {})

                logger.info(f"Subtask {subtask_name} completed")
        else:
            if spec.prepare_task is None:
                raise ValueError(f"No task factory provided for {spec.name}")
            asyncio.run(_run_default_subtasks())

        aggregated: Dict[str, Any] = {}
        if spec.aggregate_metrics:
            aggregated = spec.aggregate_metrics(all_metrics, subtasks_to_run)
        elif len(subtasks_to_run) == 1:
            aggregated = all_metrics.get(subtasks_to_run[0], {})

        if spec.write_summary:
            spec.write_summary(
                aggregated,
                all_metrics,
                task_output_path,
                model_name,
                subtasks_to_run,
            )

        mark_task_complete(task_output_path)

        logger.info(f"{spec.display_name} evaluation completed successfully")

        return {
            "model_name": model_name,
            "task": spec.name,
            "output_path": str(task_output_path),
            "metrics": aggregated,
            "subtask_metrics": all_metrics,
            "skipped_subtasks": skipped_subtasks,
        }

    except ConnectionError as exc:
        logger.error(f"Connection failed: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Error running {spec.name}: {type(exc).__name__}: {exc}")
        raise
    finally:
        if spec.teardown:
            try:
                spec.teardown(group_context, shared_state)
            except Exception as exc:
                logger.warning(f"Teardown failed for {spec.name}: {type(exc).__name__}: {exc}")


async def _run_default_subtask_async(spec: TaskGroupSpec, context: SubtaskContext) -> Dict[str, Any]:
    """Run one subtask using the default BenchmarkRunner flow."""
    task_instance = spec.prepare_task(context)
    if task_instance is None:
        return {}

    interface = get_interface_for_provider(
        provider_type=context.provider_type,
        connection_info=context.connection_info,
        model_name=context.model_name,
    )
    report = ensure_task_interface_compatibility(task_instance, interface)
    if not report.compatible:
        reason = ", ".join(report.errors) if report.errors else "incompatible capabilities"
        compatibility_mode = (context.compatibility_mode or "skip").lower()
        if compatibility_mode == "warn":
            logger.warning(
                f"Continuing despite incompatibility for {task_instance.get_task_name()}: {reason}"
            )
        elif compatibility_mode == "error":
            raise ValueError(
                f"Incompatible task {task_instance.get_task_name()}: {reason}"
            )
        else:
            if compatibility_mode not in {"skip", "warn", "error"}:
                logger.warning(
                    "Unknown compatibility_mode '%s'; defaulting to skip", compatibility_mode
                )
            logger.warning(
                f"Skipping {task_instance.get_task_name()} due to incompatibility: {reason}"
            )
            return {
                "skipped": True,
                "skip_reason": reason,
                "aggregate_metrics": {},
                "per_sample_metrics": [],
                "samples": [],
            }

    runner_config = {
        "model_name": context.model_name,
        "batch_size": context.defaults.get("batch_size", 20),
        "output_dir": str(context.subtask_output_dir),
        "log_samples": context.defaults.get("log_samples", False),
    }

    runner = BenchmarkRunner(task_instance, interface, runner_config)
    results = await runner.run(limit=context.limit, no_resume=False)

    save_results(
        results=results,
        output_dir=context.subtask_output_dir,
        model_name=context.model_name,
        task_name=task_instance.get_task_name(),
        log_samples=context.defaults.get("log_samples", False),
        mark_complete=False,
    )

    return results


def _probe_and_configure_openai_interface(
    *,
    connection_info: Dict[str, Any],
    model_name: str,
    provider_type: str,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Probe OpenAI-style interfaces for supported request modes once per task group."""
    if provider_type not in {"vllm", "openai", "anthropic", "together"}:
        return None

    request_modes = (connection_info.get("capabilities") or {}).get("request_modes") or []
    if not request_modes or "raw_payload" in request_modes:
        return None

    # Probe chat/completions/logprobs once and update connection_info for the run.
    report = asyncio.run(_probe_openai_interface(connection_info, model_name, request_modes))
    _apply_probe_results(connection_info, report)
    _write_probe_report(output_dir, report)
    _log_probe_report(report)
    return report


async def _probe_openai_interface(
    connection_info: Dict[str, Any],
    model_name: str,
    request_modes: List[str],
) -> Dict[str, Any]:
    """Probe OpenAI-style endpoints (chat/completions/logprobs)."""
    report: Dict[str, Any] = {
        "model_name": model_name,
        "provider_type": connection_info.get("provider_type"),
        "base_url": connection_info.get("base_url"),
        "api_endpoint_requested": connection_info.get("api_endpoint") or "auto",
        "modes": {},
        "selected_api_endpoint": None,
    }

    if "chat" in request_modes:
        report["modes"]["chat"] = await _probe_openai_mode(
            connection_info,
            model_name,
            api_endpoint="chat",
            use_logprobs=False,
        )

    if "completions" in request_modes:
        report["modes"]["completions"] = await _probe_openai_mode(
            connection_info,
            model_name,
            api_endpoint="completions",
            use_logprobs=False,
        )

    supports_logprobs = bool((connection_info.get("capabilities") or {}).get("supports_logprobs"))
    if supports_logprobs:
        report["modes"]["logprobs"] = await _probe_openai_mode(
            connection_info,
            model_name,
            api_endpoint="completions",
            use_logprobs=True,
        )

    report["selected_api_endpoint"] = _select_api_endpoint(
        report["api_endpoint_requested"],
        report["modes"],
    )
    return report


async def _probe_openai_mode(
    connection_info: Dict[str, Any],
    model_name: str,
    *,
    api_endpoint: str,
    use_logprobs: bool,
) -> Dict[str, Any]:
    """Run a single probe call for a request mode."""
    from ..interfaces.openai_interface import OpenAIInterface

    probe_info = deepcopy(connection_info)
    probe_info["api_endpoint"] = api_endpoint
    probe_info["max_tokens"] = min(int(probe_info.get("max_tokens", 16)), 16)

    interface = OpenAIInterface(probe_info, model_name)
    request = _build_probe_request(use_logprobs=use_logprobs)
    try:
        results = await interface.generate_batch([request])
    finally:
        close_fn = getattr(interface, "close", None)
        if close_fn:
            await close_fn()

    result = results[0] if results else {}
    ok = _probe_success(result)
    return {
        "ok": ok,
        "error": result.get("error"),
        "error_type": result.get("error_type"),
    }


def _build_probe_request(*, use_logprobs: bool) -> Dict[str, Any]:
    """Build a minimal probe request for mode detection."""
    if use_logprobs:
        return {
            "system_prompt": "",
            "user_prompt": "Choose A or B. Answer with a single letter.",
            "schema": None,
            "sample_id": "probe_logprobs",
            "use_logprobs": True,
            "choices": ["A", "B"],
        }
    return {
        "system_prompt": "",
        "user_prompt": "Reply with OK.",
        "schema": None,
        "sample_id": "probe_chat",
        "use_logprobs": False,
        "choices": None,
    }


def _probe_success(result: Dict[str, Any]) -> bool:
    """Return True if the probe produced a usable output."""
    if result.get("error"):
        return False
    output = result.get("output")
    if output is None:
        return False
    if isinstance(output, str) and not output.strip():
        return False
    return True


def _select_api_endpoint(requested: str, modes: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Select the best available endpoint based on probe results."""
    preferred = requested or "auto"
    if preferred in ("chat", "completions"):
        if modes.get(preferred, {}).get("ok"):
            return preferred
        fallback = "completions" if preferred == "chat" else "chat"
        if modes.get(fallback, {}).get("ok"):
            return fallback
        return preferred

    if modes.get("chat", {}).get("ok"):
        return "chat"
    if modes.get("completions", {}).get("ok"):
        return "completions"
    return None


def _apply_probe_results(connection_info: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Apply probe-derived endpoint/logprobs support to connection_info."""
    selected = report.get("selected_api_endpoint")
    if selected:
        connection_info["api_endpoint"] = selected

    logprobs = report.get("modes", {}).get("logprobs")
    if logprobs is not None:
        supports_logprobs = bool(logprobs.get("ok"))
        connection_info["supports_logprobs"] = supports_logprobs
        capabilities = connection_info.get("capabilities") or {}
        capabilities["supports_logprobs"] = supports_logprobs
        connection_info["capabilities"] = capabilities


def _write_probe_report(output_dir: Path, report: Dict[str, Any]) -> None:
    """Persist probe report to the task output directory."""
    report_path = output_dir / "compatibility_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _log_probe_report(report: Dict[str, Any]) -> None:
    """Log probe summary for visibility in run logs."""
    requested = report.get("api_endpoint_requested")
    selected = report.get("selected_api_endpoint")
    logger.info(f"Request mode probe selected api_endpoint={selected} (requested={requested})")
    for mode, result in (report.get("modes") or {}).items():
        status = "ok" if result.get("ok") else "failed"
        error = result.get("error")
        suffix = f" ({error})" if error else ""
        logger.info(f"Probe {mode}: {status}{suffix}")
