"""Probe runner - orchestrates capability checks and generates reports."""

import asyncio
import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tunable thresholds for repetition detection
WHITESPACE_RUN_THRESHOLD = 50  # chars
TOKEN_REPETITION_WINDOW = 10   # tokens to scan
TOKEN_REPETITION_MIN_COUNT = 5  # repetitions to flag

# Explicit check definitions - no dynamic discovery
QUICK_PROFILE_CHECKS = [
    "request_modes",      # chat/completions/logprobs
    "schema_transports",  # structured_outputs/response_format
    "multimodal",         # image + text
    "truncation",         # repetition/finish_reason
    "param_support",      # max_tokens param detection
]

FULL_PROFILE_CHECKS = QUICK_PROFILE_CHECKS + [
    # Future: context_length, batch_support, etc.
]

# Per-check timeout budgets (seconds)
CHECK_TIMEOUTS = {
    "request_modes": 30,
    "schema_transports": 30,
    "multimodal": 45,
    "truncation": 20,
    "param_support": 20,
}

# Fail-open behavior per check
CHECK_FAIL_MODES = {
    "request_modes": "failed",      # Critical for eval
    "schema_transports": "failed",  # Critical for eval
    "multimodal": "degraded",       # Non-critical
    "truncation": "degraded",       # Warning only
    "param_support": "degraded",    # Warning only
}


async def run_probe(
    connection_info: Dict[str, Any],
    model_name: str,
    run_id: str,
    output_path: Path,
    profile: str = "quick",
    global_timeout: int = 180,
) -> Dict[str, Any]:
    """Run full probe suite and generate report.
    
    Args:
        connection_info: Connection configuration dict
        model_name: Model name to probe
        run_id: Unique run identifier
        output_path: Directory to write probe artifacts
        profile: Probe profile ("quick" or "full")
        global_timeout: Global timeout for all checks (seconds)
    
    Returns:
        Probe report dict
    """
    started_at = datetime.now(timezone.utc)
    provider_type = connection_info.get("provider_type", "unknown")
    base_url = connection_info.get("base_url", "")
    
    logger.info(f"Starting probe: model={model_name} provider={provider_type} profile={profile}")
    
    # Initialize report structure with compatibility fields
    report = {
        "schema_version": "1.0",
        "benchy_version": _get_benchy_version(),
        "model": model_name,
        "provider_type": provider_type,
        "base_url": base_url,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "status": "passed",
        
        # COMPATIBILITY: Required by group_runner.py
        "modes": {
            "chat": {"ok": False, "error": None, "error_type": None},
            "completions": {"ok": False, "error": None, "error_type": None},
            "logprobs": {"ok": False, "error": None, "error_type": None},
        },
        "schema_transports": {
            "structured_outputs": {"ok": False, "error": None, "schema_transport": "structured_outputs"},
            "response_format": {"ok": False, "error": None, "schema_transport": "response_format"},
        },
        "selected_api_endpoint": "chat",  # Default, will be updated
        "selected_schema_transport": "structured_outputs",  # Default, will be updated
        
        # NEW: Additional checks
        "checks": {},
        
        # Risk analysis
        "risk_flags": {
            "truncation_risk": False,
            "schema_unreliable": False,
            "repetition_risk": False,
            "multimodal_unreliable": False,
        },
        
        # Provider fingerprint
        "provider_fingerprint": {},
        
        # Global errors
        "errors": [],
    }
    
    try:
        # Collect provider fingerprint
        report["provider_fingerprint"] = await _collect_provider_fingerprint(
            connection_info, model_name
        )
        
        # Determine which checks to run
        checks_to_run = QUICK_PROFILE_CHECKS if profile == "quick" else FULL_PROFILE_CHECKS
        
        # Get request modes from capabilities
        request_modes = (connection_info.get("capabilities") or {}).get("request_modes") or []
        supports_schema = bool((connection_info.get("capabilities") or {}).get("supports_schema"))
        
        # Run core compatibility checks
        if request_modes and "raw_payload" not in request_modes:
            # Request modes check
            if "request_modes" in checks_to_run:
                modes_result = await _run_check_with_timeout(
                    "request_modes",
                    lambda: _probe_request_modes(connection_info, model_name, request_modes),
                    CHECK_TIMEOUTS["request_modes"],
                    CHECK_FAIL_MODES["request_modes"],
                )
                report["modes"].update(modes_result)
            
            # Schema transports check
            if "schema_transports" in checks_to_run and supports_schema and "chat" in request_modes:
                transports_result = await _run_check_with_timeout(
                    "schema_transports",
                    lambda: _probe_schema_transports(connection_info, model_name),
                    CHECK_TIMEOUTS["schema_transports"],
                    CHECK_FAIL_MODES["schema_transports"],
                )
                report["schema_transports"].update(transports_result)
        
        # Select best endpoint and transport
        report["selected_api_endpoint"] = _select_api_endpoint(
            connection_info.get("api_endpoint") or "auto",
            report["modes"],
        ) or "chat"
        
        report["selected_schema_transport"] = _select_schema_transport(
            "structured_outputs" if connection_info.get("use_structured_outputs") else "auto",
            report["schema_transports"],
        ) or "structured_outputs"
        
        # Run additional checks
        if "multimodal" in checks_to_run:
            report["checks"]["multimodal"] = await _run_check_with_timeout(
                "multimodal",
                lambda: _probe_multimodal(connection_info, model_name),
                CHECK_TIMEOUTS["multimodal"],
                CHECK_FAIL_MODES["multimodal"],
            )
        
        if "truncation" in checks_to_run:
            report["checks"]["truncation"] = await _run_check_with_timeout(
                "truncation",
                lambda: _probe_truncation(connection_info, model_name),
                CHECK_TIMEOUTS["truncation"],
                CHECK_FAIL_MODES["truncation"],
            )
        
        if "param_support" in checks_to_run:
            report["checks"]["param_support"] = await _run_check_with_timeout(
                "param_support",
                lambda: _probe_param_support(connection_info, model_name),
                CHECK_TIMEOUTS["param_support"],
                CHECK_FAIL_MODES["param_support"],
            )
        
        # Generate risk flags
        report["risk_flags"] = _generate_risk_flags(report)
        
        # Determine overall status
        report["status"] = _determine_overall_status(report)
        
    except Exception as e:
        logger.error(f"Probe failed with exception: {e}", exc_info=True)
        report["status"] = "failed"
        report["errors"].append(str(e))
    
    finally:
        ended_at = datetime.now(timezone.utc)
        report["ended_at"] = ended_at.isoformat()
        duration = (ended_at - started_at).total_seconds()
        
        logger.info(
            f"Probe completed: status={report['status']} duration={duration:.1f}s "
            f"endpoint={report['selected_api_endpoint']} "
            f"transport={report['selected_schema_transport']}"
        )
        
        # Write artifacts
        _write_probe_artifacts(output_path, report)
    
    return report


async def run_probe_for_eval(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Run probe for eval integration (inline, no artifacts).
    
    This is a lightweight version for group_runner.py integration.
    """
    provider_type = connection_info.get("provider_type", "unknown")
    
    logger.info(f"Running inline probe: model={model_name} provider={provider_type}")
    
    # Initialize minimal report structure
    report = {
        "model_name": model_name,
        "provider_type": provider_type,
        "base_url": connection_info.get("base_url"),
        "modes": {
            "chat": {"ok": False, "error": None, "error_type": None},
            "completions": {"ok": False, "error": None, "error_type": None},
            "logprobs": {"ok": False, "error": None, "error_type": None},
        },
        "schema_transports": {
            "structured_outputs": {"ok": False, "error": None, "schema_transport": "structured_outputs"},
            "response_format": {"ok": False, "error": None, "schema_transport": "response_format"},
        },
        "selected_api_endpoint": "chat",
        "selected_schema_transport": "structured_outputs",
    }
    
    try:
        request_modes = (connection_info.get("capabilities") or {}).get("request_modes") or []
        supports_schema = bool((connection_info.get("capabilities") or {}).get("supports_schema"))
        
        if request_modes and "raw_payload" not in request_modes:
            # Probe request modes
            modes_result = await _probe_request_modes(connection_info, model_name, request_modes)
            report["modes"].update(modes_result)
            
            # Probe schema transports
            if supports_schema and "chat" in request_modes:
                transports_result = await _probe_schema_transports(connection_info, model_name)
                report["schema_transports"].update(transports_result)
        
        # Select best options
        report["selected_api_endpoint"] = _select_api_endpoint(
            connection_info.get("api_endpoint") or "auto",
            report["modes"],
        ) or "chat"
        
        report["selected_schema_transport"] = _select_schema_transport(
            "structured_outputs" if connection_info.get("use_structured_outputs") else "auto",
            report["schema_transports"],
        ) or "structured_outputs"
        
    except Exception as e:
        logger.error(f"Inline probe failed: {e}", exc_info=True)
    
    return report


async def _run_check_with_timeout(
    check_name: str,
    check_func: Callable,
    timeout: int,
    fail_mode: str,
) -> Dict[str, Any]:
    """Run a check with timeout and failure handling."""
    start_time = datetime.now()
    
    try:
        logger.info(f"Probe check={check_name} timeout={timeout}s")
        result = await asyncio.wait_for(check_func(), timeout=timeout)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Probe result: check={check_name} status={result.get('status', 'ok')} "
            f"finish_reason={result.get('finish_reason', 'unknown')} "
            f"tokens={result.get('completion_tokens', 0)}/{result.get('prompt_tokens', 0)} "
            f"duration={duration:.2f}s"
        )
        
        return result
        
    except asyncio.TimeoutError:
        logger.warning(f"Check {check_name} timed out after {timeout}s")
        return {"status": fail_mode, "error": f"Timeout after {timeout}s", "evidence": {}}
    
    except Exception as e:
        logger.error(f"Check {check_name} failed: {e}")
        return {"status": fail_mode, "error": str(e), "evidence": {}}


# ============================================================================
# Core Probe Functions (migrated from group_runner.py)
# ============================================================================

async def _probe_request_modes(
    connection_info: Dict[str, Any],
    model_name: str,
    request_modes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Probe chat/completions/logprobs support.
    
    Returns dict matching group_runner.py format for compatibility.
    """
    results = {}
    
    if "chat" in request_modes:
        results["chat"] = await _probe_openai_mode(
            connection_info,
            model_name,
            api_endpoint="chat",
            use_logprobs=False,
        )
    
    if "completions" in request_modes:
        results["completions"] = await _probe_openai_mode(
            connection_info,
            model_name,
            api_endpoint="completions",
            use_logprobs=False,
        )
    
    supports_logprobs = bool((connection_info.get("capabilities") or {}).get("supports_logprobs"))
    if supports_logprobs:
        results["logprobs"] = await _probe_openai_mode(
            connection_info,
            model_name,
            api_endpoint="completions",
            use_logprobs=True,
        )
    
    return results


async def _probe_schema_transports(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Probe structured_outputs vs response_format support."""
    results = {}
    
    results["structured_outputs"] = await _probe_openai_mode(
        connection_info,
        model_name,
        api_endpoint="chat",
        use_logprobs=False,
        schema_transport="structured_outputs",
    )
    
    results["response_format"] = await _probe_openai_mode(
        connection_info,
        model_name,
        api_endpoint="chat",
        use_logprobs=False,
        schema_transport="response_format",
    )
    
    return results


async def _probe_openai_mode(
    connection_info: Dict[str, Any],
    model_name: str,
    *,
    api_endpoint: str,
    use_logprobs: bool,
    schema_transport: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single probe call for a request mode.
    
    Migrated from group_runner.py with enhanced logging.
    """
    from ..interfaces.openai_interface import OpenAIInterface
    
    probe_info = deepcopy(connection_info)
    probe_info["api_endpoint"] = api_endpoint
    
    # Use 100 tokens for probes to ensure models produce usable output
    # Some models (like gpt-5-mini) have high token overhead and need more tokens
    probe_info["max_tokens"] = min(int(probe_info.get("max_tokens", 100)), 100)
    
    if schema_transport == "structured_outputs":
        probe_info["use_structured_outputs"] = True
    elif schema_transport == "response_format":
        probe_info["use_structured_outputs"] = False
    
    interface = OpenAIInterface(probe_info, model_name)
    request = _build_probe_request(use_logprobs=use_logprobs, schema_transport=schema_transport)
    
    try:
        results = await interface.generate_batch([request])
    finally:
        close_fn = getattr(interface, "close", None)
        if close_fn:
            await close_fn()
    
    result = results[0] if results else {}
    ok = _probe_schema_transport_success(result) if schema_transport else _probe_success(result)
    
    return {
        "ok": ok,
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "schema_transport": schema_transport,
    }


def _build_probe_request(*, use_logprobs: bool, schema_transport: Optional[str] = None) -> Dict[str, Any]:
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
    
    if schema_transport:
        return {
            "system_prompt": "",
            "user_prompt": "Return JSON with field `ok` and value `yes`.",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "string"}},
                "required": ["ok"],
            },
            "sample_id": f"probe_schema_{schema_transport}",
            "use_logprobs": False,
            "choices": None,
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


def _probe_schema_transport_success(result: Dict[str, Any]) -> bool:
    """Return True when schema transport appears accepted by the endpoint."""
    if result.get("output") is not None:
        return True
    raw = result.get("raw")
    if isinstance(raw, str) and raw.strip():
        return True
    return False


# ============================================================================
# Selection Logic (migrated from group_runner.py)
# ============================================================================

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
    
    # Auto mode: prefer chat, fallback to completions
    if modes.get("chat", {}).get("ok"):
        return "chat"
    if modes.get("completions", {}).get("ok"):
        return "completions"
    return None


def _select_schema_transport(
    requested: str,
    schema_transports: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Select best schema transport based on probe results."""
    preferred = requested or "auto"
    
    if preferred in ("structured_outputs", "response_format"):
        if schema_transports.get(preferred, {}).get("ok"):
            return preferred
        fallback = "response_format" if preferred == "structured_outputs" else "structured_outputs"
        if schema_transports.get(fallback, {}).get("ok"):
            return fallback
        return preferred
    
    # Auto mode preference: structured_outputs first, then response_format
    if schema_transports.get("structured_outputs", {}).get("ok"):
        return "structured_outputs"
    if schema_transports.get("response_format", {}).get("ok"):
        return "response_format"
    return None


# ============================================================================
# New Probe Checks
# ============================================================================

async def _probe_multimodal(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Test multimodal (image + text) capability."""
    from .assets import get_test_image_path
    from ..interfaces.openai_interface import OpenAIInterface
    
    try:
        image_path = get_test_image_path()
        
        probe_info = deepcopy(connection_info)
        probe_info["api_endpoint"] = "chat"
        probe_info["max_tokens"] = 16
        
        interface = OpenAIInterface(probe_info, model_name)
        request = {
            "system_prompt": "",
            "user_prompt": "Reply with OK if you can see an image.",
            "schema": None,
            "sample_id": "probe_multimodal",
            "use_logprobs": False,
            "choices": None,
            "image_path": str(image_path),
        }
        
        try:
            results = await interface.generate_batch([request])
        finally:
            close_fn = getattr(interface, "close", None)
            if close_fn:
                await close_fn()
        
        result = results[0] if results else {}
        
        return {
            "status": "ok" if _probe_success(result) else "failed",
            "evidence": {
                "image_accepted": not result.get("error_type") in ["api_error", "connectivity_error"],
                "response_received": bool(result.get("output")),
                "finish_reason": result.get("finish_reason"),
            },
            "error": result.get("error"),
            "finish_reason": result.get("finish_reason"),
            "completion_tokens": result.get("completion_tokens"),
            "prompt_tokens": result.get("prompt_tokens"),
        }
    
    except Exception as e:
        logger.error(f"Multimodal probe failed: {e}")
        return {
            "status": "failed",
            "evidence": {},
            "error": str(e),
            "finish_reason": None,
            "completion_tokens": None,
            "prompt_tokens": None,
        }


async def _probe_truncation(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Test for truncation/repetition issues.
    
    This probe intentionally forces truncation to test if the model:
    1. Properly reports finish_reason=length (good behavior)
    2. Produces repetition patterns when truncated (bad behavior)
    
    A model that reports finish_reason=length WITHOUT repetition is OK.
    A model that produces repetition when truncated is DEGRADED.
    """
    from ..interfaces.openai_interface import OpenAIInterface
    
    try:
        probe_info = deepcopy(connection_info)
        probe_info["api_endpoint"] = "chat"
        probe_info["max_tokens"] = 16  # Force truncation to test behavior
        
        interface = OpenAIInterface(probe_info, model_name)
        request = {
            "system_prompt": "",
            "user_prompt": "Write a 500-word essay about artificial intelligence.",
            "schema": None,
            "sample_id": "probe_truncation",
            "use_logprobs": False,
            "choices": None,
        }
        
        try:
            results = await interface.generate_batch([request])
        finally:
            close_fn = getattr(interface, "close", None)
            if close_fn:
                await close_fn()
        
        result = results[0] if results else {}
        output_text = result.get("output", "")
        finish_reason = result.get("finish_reason")
        completion_tokens = result.get("completion_tokens", 0)
        
        # The key issue: does the model produce repetition when truncated?
        repetition_detected = _detect_repetition(output_text)
        truncated = finish_reason == "length" or completion_tokens >= 16
        reports_truncation_correctly = finish_reason == "length"
        
        # ONLY flag as degraded if repetition is detected
        # Truncation itself is expected and OK if handled cleanly
        issues_found = repetition_detected
        
        return {
            "status": "degraded" if issues_found else "ok",
            "evidence": {
                "finish_reason": finish_reason,
                "repetition_detected": repetition_detected,
                "reports_truncation_correctly": reports_truncation_correctly,
                "completion_tokens": completion_tokens,
                "max_tokens_hit": completion_tokens >= 16,
            },
            "error": "Repetition detected when truncated" if repetition_detected else None,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "prompt_tokens": result.get("prompt_tokens"),
        }
    
    except Exception as e:
        logger.error(f"Truncation probe failed: {e}")
        return {
            "status": "degraded",
            "evidence": {},
            "error": str(e),
            "finish_reason": None,
            "completion_tokens": None,
            "prompt_tokens": None,
        }


def _detect_repetition(text: str) -> bool:
    """Detect repetition patterns in model output."""
    if not text:
        return False
    
    # Whitespace runs > threshold
    if re.search(rf'\s{{{WHITESPACE_RUN_THRESHOLD},}}', text):
        return True
    
    # Token repetition (same 2-token pattern repeated 5+ times in window)
    tokens = text.split()
    if len(tokens) < TOKEN_REPETITION_WINDOW:
        return False
    
    for i in range(len(tokens) - TOKEN_REPETITION_WINDOW):
        window = tokens[i:i+TOKEN_REPETITION_WINDOW]
        if len(window) < 2:
            continue
        
        pattern = (tokens[i], tokens[i+1])
        count = sum(1 for j in range(len(window)-1) 
                   if j+1 < len(window) and (window[j], window[j+1]) == pattern)
        
        if count >= TOKEN_REPETITION_MIN_COUNT:
            return True
    
    return False


async def _probe_param_support(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Test which max_tokens parameter variant is accepted.
    
    OpenAI's newer models (gpt-5, o1, o3) require max_completion_tokens.
    Older models and vLLM use max_tokens.
    """
    from ..interfaces.openai_interface import OpenAIInterface
    
    try:
        # Test 1: Try max_tokens (most common)
        # Explicitly set max_tokens_param_name to disable auto-detection
        probe_info1 = deepcopy(connection_info)
        probe_info1["api_endpoint"] = "chat"
        probe_info1["max_tokens_param_name"] = "max_tokens"  # Force explicit parameter
        probe_info1["max_tokens"] = 100  # High budget to ensure completion for models with overhead
        
        interface1 = OpenAIInterface(probe_info1, model_name)
        request = {
            "system_prompt": "",
            "user_prompt": "Reply with OK.",
            "schema": None,
            "sample_id": "probe_param_max_tokens",
            "use_logprobs": False,
            "choices": None,
        }
        
        try:
            results1 = await interface1.generate_batch([request])
        finally:
            close_fn = getattr(interface1, "close", None)
            if close_fn:
                await close_fn()
        
        result1 = results1[0] if results1 else {}
        max_tokens_works = _probe_success(result1)
        
        # Test 2: Always try max_completion_tokens if max_tokens didn't work
        # (not just on API errors - could be empty output)
        result2 = None
        max_completion_tokens_works = False
        
        if not max_tokens_works:
            probe_info2 = deepcopy(connection_info)
            probe_info2["api_endpoint"] = "chat"
            probe_info2["max_tokens_param_name"] = "max_completion_tokens"
            probe_info2["max_tokens"] = 100  # High budget to ensure completion for models with overhead
            
            interface2 = OpenAIInterface(probe_info2, model_name)
            
            try:
                results2 = await interface2.generate_batch([request])
            finally:
                close_fn = getattr(interface2, "close", None)
                if close_fn:
                    await close_fn()
            
            result2 = results2[0] if results2 else {}
            max_completion_tokens_works = _probe_success(result2)
        
        # Determine accepted parameter
        if max_tokens_works:
            accepted_param = "max_tokens"
        elif max_completion_tokens_works:
            accepted_param = "max_completion_tokens"
        else:
            accepted_param = "unknown"
        
        final_result = result2 if result2 else result1
        
        return {
            "status": "ok" if accepted_param != "unknown" else "degraded",
            "evidence": {
                "max_tokens_param": accepted_param,
                "accepts_max_tokens": max_tokens_works,
                "accepts_max_completion_tokens": max_completion_tokens_works,
            },
            "error": None if accepted_param != "unknown" else "Could not determine max_tokens parameter",
            "finish_reason": final_result.get("finish_reason"),
            "completion_tokens": final_result.get("completion_tokens"),
            "prompt_tokens": final_result.get("prompt_tokens"),
        }
    
    except Exception as e:
        logger.error(f"Parameter support probe failed: {e}")
        return {
            "status": "degraded",
            "evidence": {"max_tokens_param": "unknown"},
            "error": str(e),
            "finish_reason": None,
            "completion_tokens": None,
            "prompt_tokens": None,
        }


# ============================================================================
# Provider Fingerprinting
# ============================================================================

async def _collect_provider_fingerprint(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Collect provider metadata for reproducibility tracking."""
    fingerprint = {
        "models_endpoint_available": False,
        "reported_models": [],
        "server_headers": {},
    }
    
    try:
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available, skipping provider fingerprint")
            return fingerprint
        
        base_url = connection_info.get("base_url")
        if not base_url:
            return fingerprint
        
        api_key = connection_info.get("api_key") or "EMPTY"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/models",
                timeout=10,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code == 200:
                fingerprint["models_endpoint_available"] = True
                data = response.json()
                fingerprint["reported_models"] = [
                    m.get("id") for m in data.get("data", [])
                    if isinstance(m, dict) and m.get("id")
                ]
            
            # Capture server headers
            fingerprint["server_headers"] = {
                "server": response.headers.get("server"),
                "x-request-id": response.headers.get("x-request-id"),
            }
    
    except Exception as e:
        logger.debug(f"Could not collect provider fingerprint: {e}")
    
    return fingerprint


# ============================================================================
# Risk Analysis and Status
# ============================================================================

def _generate_risk_flags(report: Dict[str, Any]) -> Dict[str, bool]:
    """Generate risk flags based on check results.
    
    IMPORTANT: truncation_risk should only be set if the model produces
    repetition/broken output when truncated, NOT just because it reports
    finish_reason=length (which is correct behavior).
    """
    flags = {
        "truncation_risk": False,
        "schema_unreliable": False,
        "repetition_risk": False,
        "multimodal_unreliable": False,
    }
    
    # Repetition risk - the actual problem we care about
    truncation_check = report.get("checks", {}).get("truncation", {})
    if truncation_check.get("status") == "degraded":
        evidence = truncation_check.get("evidence", {})
        if evidence.get("repetition_detected"):
            flags["repetition_risk"] = True
            # If repetition occurs during truncation, this is a real truncation risk
            flags["truncation_risk"] = True
    
    # Schema unreliable
    transports = report.get("schema_transports", {})
    if not transports.get("structured_outputs", {}).get("ok") and \
       not transports.get("response_format", {}).get("ok"):
        flags["schema_unreliable"] = True
    
    # Multimodal unreliable
    multimodal_check = report.get("checks", {}).get("multimodal", {})
    if multimodal_check.get("status") == "failed":
        flags["multimodal_unreliable"] = True
    
    return flags


def _determine_overall_status(report: Dict[str, Any]) -> str:
    """Determine overall probe status."""
    # Check for critical failures
    modes = report.get("modes", {})
    if not any(isinstance(m, dict) and m.get("ok") for m in modes.values()):
        return "failed"
    
    transports = report.get("schema_transports", {})
    if not any(isinstance(t, dict) and t.get("ok") for t in transports.values()):
        return "failed"
    
    # Check for degraded state
    checks = report.get("checks", {})
    if any(isinstance(c, dict) and c.get("status") == "degraded" for c in checks.values()):
        return "degraded"
    
    if any(isinstance(c, dict) and c.get("status") == "failed" for c in checks.values()):
        return "degraded"
    
    return "passed"


# ============================================================================
# Artifact Writing
# ============================================================================

def _write_probe_artifacts(output_path: Path, report: Dict[str, Any]) -> None:
    """Write probe report and summary to disk."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write JSON report
    report_path = output_path / "probe_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Wrote probe report: {report_path}")
    
    # Write text summary
    summary_path = output_path / "probe_summary.txt"
    with open(summary_path, "w") as f:
        f.write(_generate_summary_text(report))
    
    logger.info(f"Wrote probe summary: {summary_path}")


def _generate_summary_text(report: Dict[str, Any]) -> str:
    """Generate human-readable summary text."""
    lines = []
    
    lines.append(f"Probe Report: {report['model']}")
    lines.append(f"Provider: {report['provider_type']} @ {report['base_url']}")
    lines.append(f"Status: {report['status'].upper()}")
    
    started = datetime.fromisoformat(report['started_at'])
    ended = datetime.fromisoformat(report['ended_at'])
    duration = (ended - started).total_seconds()
    lines.append(f"Duration: {duration:.1f} seconds")
    lines.append("")
    
    lines.append("Capabilities:")
    
    # Modes
    modes = report.get("modes", {})
    for mode, result in modes.items():
        if isinstance(result, dict):
            status = "[OK]" if result.get("ok") else "[FAIL]"
        else:
            status = "[FAIL]"
        lines.append(f"  {status} {mode.capitalize()} endpoint")
    
    # Schema transports
    transports = report.get("schema_transports", {})
    for transport, result in transports.items():
        if isinstance(result, dict) and result.get("ok"):
            lines.append(f"  [OK] {transport.replace('_', ' ').title()}")
    
    # Additional checks
    checks = report.get("checks", {})
    for check_name, result in checks.items():
        if isinstance(result, dict):
            status_map = {"ok": "[OK]", "degraded": "[WARN]", "failed": "[FAIL]"}
            status = status_map.get(result.get("status"), "[?]")
        else:
            status = "[FAIL]"
        lines.append(f"  {status} {check_name.replace('_', ' ').title()}")
    
    lines.append("")
    lines.append("Selected Configuration:")
    lines.append(f"  API endpoint: {report['selected_api_endpoint']}")
    lines.append(f"  Schema transport: {report['selected_schema_transport']}")
    
    # Risk flags
    risk_flags = report.get("risk_flags", {})
    if any(risk_flags.values()):
        lines.append("")
        lines.append("Risk Flags:")
        if risk_flags.get("truncation_risk"):
            lines.append("  [!] Truncation risk: Model responses may be cut off at token limit")
        if risk_flags.get("repetition_risk"):
            lines.append("  [!] Repetition risk: Model may produce repetitive output")
        if risk_flags.get("schema_unreliable"):
            lines.append("  [!] Schema unreliable: Structured output may not work correctly")
        if risk_flags.get("multimodal_unreliable"):
            lines.append("  [!] Multimodal unreliable: Image inputs may not be supported")
    
    return "\n".join(lines)


def _get_benchy_version() -> str:
    """Get benchy version string."""
    try:
        from .. import __version__
        return __version__
    except:
        return "0.1.0"
