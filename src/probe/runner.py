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
    "access_readiness",   # auth/model/quota preflight
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
    "access_readiness": 20,
    "request_modes": 30,
    "schema_transports": 30,
    "multimodal": 45,
    "truncation": 20,
    "param_support": 20,
}

# Fail-open behavior per check
CHECK_FAIL_MODES = {
    "access_readiness": "failed",    # Critical preflight
    "request_modes": "failed",      # Critical for eval
    "schema_transports": "failed",  # Critical for eval
    "multimodal": "degraded",       # Non-critical
    "truncation": "degraded",       # Warning only
    "param_support": "degraded",    # Warning only
}

CHECK_DISPLAY_NAMES = {
    "access_readiness": "Access Readiness",
    "request_modes": "Request Modes",
    "schema_transports": "Schema Transports",
    "multimodal": "Multimodal",
    "truncation": "Truncation Behavior",
    "param_support": "Max Tokens Parameter",
}


def _empty_mode_result(*, schema_transport: Optional[str] = None) -> Dict[str, Any]:
    return {
        "tested": False,
        "ok": None,
        "accepted_by_api": None,
        "reliable_for_eval": None,
        "error": None,
        "error_type": None,
        "schema_transport": schema_transport,
        "skip_reason": "not_tested",
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
            "chat": _empty_mode_result(),
            "completions": _empty_mode_result(),
            "logprobs": _empty_mode_result(),
        },
        "schema_transports": {
            "structured_outputs": _empty_mode_result(schema_transport="structured_outputs"),
            "response_format": _empty_mode_result(schema_transport="response_format"),
        },
        "selected_api_endpoint": "chat",  # Default, will be updated
        "selected_schema_transport": None,
        "api_endpoint_requested": connection_info.get("api_endpoint") or "auto",
        "schema_transport_requested": (
            "structured_outputs" if connection_info.get("use_structured_outputs") else "auto"
        ),
        "schema_transport_forced": bool(connection_info.get("use_structured_outputs_explicit")),
        
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
        # Source-of-truth metadata for what probe tested
        "test_plan": _build_test_plan(profile),
        "known_blindspots": _known_blindspots(),
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
        access_allows_probing = True
        if "access_readiness" in checks_to_run:
            report["checks"]["access_readiness"] = await _run_check_with_timeout(
                "access_readiness",
                lambda: _probe_access_readiness(connection_info, model_name),
                CHECK_TIMEOUTS["access_readiness"],
                CHECK_FAIL_MODES["access_readiness"],
            )
            access_allows_probing = report["checks"]["access_readiness"].get("status") != "failed"

        if access_allows_probing and request_modes and "raw_payload" not in request_modes:
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
        )
        
        # Run additional checks
        if access_allows_probing and "multimodal" in checks_to_run:
            report["checks"]["multimodal"] = await _run_check_with_timeout(
                "multimodal",
                lambda: _probe_multimodal(connection_info, model_name),
                CHECK_TIMEOUTS["multimodal"],
                CHECK_FAIL_MODES["multimodal"],
            )
        
        if access_allows_probing and "truncation" in checks_to_run:
            report["checks"]["truncation"] = await _run_check_with_timeout(
                "truncation",
                lambda: _probe_truncation(connection_info, model_name),
                CHECK_TIMEOUTS["truncation"],
                CHECK_FAIL_MODES["truncation"],
            )
        
        if access_allows_probing and "param_support" in checks_to_run:
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
        "api_endpoint_requested": connection_info.get("api_endpoint") or "auto",
        "schema_transport_requested": (
            "structured_outputs" if connection_info.get("use_structured_outputs") else "auto"
        ),
        "schema_transport_forced": bool(connection_info.get("use_structured_outputs_explicit")),
        "modes": {
            "chat": _empty_mode_result(),
            "completions": _empty_mode_result(),
            "logprobs": _empty_mode_result(),
        },
        "schema_transports": {
            "structured_outputs": _empty_mode_result(schema_transport="structured_outputs"),
            "response_format": _empty_mode_result(schema_transport="response_format"),
        },
        "selected_api_endpoint": "chat",
        "selected_schema_transport": None,
        "checks": {},
        "test_plan": _build_test_plan("quick"),
        "known_blindspots": _known_blindspots(),
    }
    
    try:
        request_modes = (connection_info.get("capabilities") or {}).get("request_modes") or []
        supports_schema = bool((connection_info.get("capabilities") or {}).get("supports_schema"))
        
        report["checks"]["access_readiness"] = await _run_check_with_timeout(
            "access_readiness",
            lambda: _probe_access_readiness(connection_info, model_name),
            CHECK_TIMEOUTS["access_readiness"],
            CHECK_FAIL_MODES["access_readiness"],
        )
        access_allows_probing = report["checks"]["access_readiness"].get("status") != "failed"

        if access_allows_probing and request_modes and "raw_payload" not in request_modes:
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
        )
        
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
    probe_info["temperature"] = 0.0
    
    # Keep probes deterministic and with enough budget to avoid false negatives.
    probe_info["max_tokens"] = max(128, min(int(probe_info.get("max_tokens", 256)), 512))
    
    if schema_transport == "structured_outputs":
        probe_info["use_structured_outputs"] = True
    elif schema_transport == "response_format":
        probe_info["use_structured_outputs"] = False
    
    interface = OpenAIInterface(probe_info, model_name)
    request = _build_probe_request(
        use_logprobs=use_logprobs,
        schema_transport=schema_transport,
        stress=False,
    )
    
    try:
        results = await interface.generate_batch([request])

        result = results[0] if results else {}
        ok = _probe_schema_transport_success(result) if schema_transport else _probe_success(result)

        stress_result: Dict[str, Any] = {}
        stress_ok: Optional[bool] = None
        stress_executed = False
        if schema_transport and ok:
            stress_executed = True
            stress_request = _build_probe_request(
                use_logprobs=False,
                schema_transport=schema_transport,
                stress=True,
            )
            supports_multimodal = bool(
                (connection_info.get("capabilities") or {}).get("supports_multimodal", True)
            )
            if supports_multimodal:
                from .assets import get_test_image_path

                stress_request["image_path"] = str(get_test_image_path())
            stress_results = await interface.generate_batch([stress_request])
            stress_result = stress_results[0] if stress_results else {}
            stress_ok = _probe_schema_transport_success(stress_result)
            ok = ok and stress_ok
    finally:
        close_fn = getattr(interface, "close", None)
        if close_fn:
            await close_fn()

    selected_result = result if ok or not stress_result else stress_result
    error = selected_result.get("error")
    error_type = selected_result.get("error_type")
    if not ok and stress_result and not error:
        error = stress_result.get("error") or "Schema transport failed stress probe"
        error_type = stress_result.get("error_type")

    accepted_by_api = None
    reliable_for_eval = None
    if schema_transport:
        basic_accepted = _probe_schema_transport_accepted(result)
        basic_reliable = _probe_schema_transport_success(result)
        stress_accepted = _probe_schema_transport_accepted(stress_result) if stress_executed else None
        stress_reliable = _probe_schema_transport_success(stress_result) if stress_executed else None
        effective_stress_accepted = stress_accepted if stress_accepted is not None else True
        effective_stress_reliable = stress_reliable if stress_reliable is not None else True
        accepted_by_api = bool(basic_accepted and effective_stress_accepted)
        reliable_for_eval = bool(basic_reliable and effective_stress_reliable)
        ok = reliable_for_eval
        if accepted_by_api and not reliable_for_eval and not error:
            error = "Transport accepted by API but produced unreliable structured output"
            error_type = "unreliable_schema_transport"

    return {
        "tested": True,
        "ok": ok,
        "accepted_by_api": accepted_by_api,
        "reliable_for_eval": reliable_for_eval,
        "error": error,
        "error_type": error_type,
        "schema_transport": schema_transport,
        "skip_reason": None,
        "finish_reason": selected_result.get("finish_reason"),
        "completion_tokens": selected_result.get("completion_tokens"),
        "prompt_tokens": selected_result.get("prompt_tokens"),
        "evidence": {
            "basic_ok": bool(_probe_schema_transport_success(result) if schema_transport else _probe_success(result)),
            "stress_ok": stress_ok if schema_transport else None,
            "stress_executed": stress_executed if schema_transport else None,
            "basic_accepted": _probe_schema_transport_accepted(result) if schema_transport else None,
            "stress_accepted": _probe_schema_transport_accepted(stress_result) if schema_transport and stress_executed else None,
        },
    }


def _build_probe_request(
    *,
    use_logprobs: bool,
    schema_transport: Optional[str] = None,
    stress: bool = False,
) -> Dict[str, Any]:
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
    
    if schema_transport and stress:
        return {
            "system_prompt": "You are a strict JSON extraction engine.",
            "user_prompt": (
                "Extract the fields and return JSON only. "
                "Do not add prose. Keep values concise."
            ),
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "string", "enum": ["yes"]},
                    "doc_id": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "qty": {"type": "integer"},
                                "price": {"type": "number"},
                            },
                            "required": ["name", "qty", "price"],
                        },
                    },
                    "totals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "subtotal": {"type": "number"},
                            "tax": {"type": "number"},
                            "grand_total": {"type": "number"},
                        },
                        "required": ["subtotal", "tax", "grand_total"],
                    },
                },
                "required": ["ok", "doc_id", "items", "totals"],
            },
            "sample_id": f"probe_schema_{schema_transport}_stress",
            "use_logprobs": False,
            "choices": None,
        }

    if schema_transport:
        return {
            "system_prompt": "",
            "user_prompt": "Return JSON exactly as {\"ok\":\"yes\"}.",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "string", "enum": ["yes"]}},
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
    if result.get("error"):
        return False
    finish_reason = str(result.get("finish_reason") or "").lower()
    if finish_reason == "length":
        return False
    output = result.get("output")
    if not isinstance(output, dict):
        return False
    if output.get("ok") not in {"yes", None}:
        return False
    raw = result.get("raw")
    if isinstance(raw, str) and _detect_repetition(raw):
        return False
    return True


def _probe_schema_transport_accepted(result: Dict[str, Any]) -> bool:
    """Return True when the transport parameter is accepted at API level.

    This is intentionally looser than reliability checks:
    - accepted_by_api=True can still be unreliable_for_eval=False.
    """
    if not isinstance(result, dict) or not result:
        return False
    if result.get("output") is not None:
        return True
    if result.get("finish_reason") is not None:
        return True
    if result.get("completion_tokens") is not None or result.get("prompt_tokens") is not None:
        return True
    raw = result.get("raw")
    if isinstance(raw, str) and raw.strip():
        return True
    err_type = str(result.get("error_type") or "")
    if err_type == "invalid_response":
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
        probe_info["temperature"] = 0.0
        probe_info["max_tokens"] = max(64, min(int(probe_info.get("max_tokens", 128)), 256))
        
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
        probe_info["temperature"] = 0.0
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
        probe_info1["temperature"] = 0.0
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
            probe_info2["temperature"] = 0.0
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
                "tested_parameters": ["max_tokens", "max_completion_tokens"],
                "temperature_tested": False,
                "temperature_note": (
                    "Temperature is not part of this check. "
                    "OpenAI gpt-5/o* families are handled by interface routing logic."
                ),
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
            "evidence": {
                "max_tokens_param": "unknown",
                "tested_parameters": ["max_tokens", "max_completion_tokens"],
                "temperature_tested": False,
            },
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


def _parse_error_status_code(error: Optional[str]) -> Optional[int]:
    if not error:
        return None
    match = re.search(r"(?:status code|error code)[:\s]+(\d{3})", str(error).lower())
    if match:
        return int(match.group(1))
    return None


def _classify_access_issue(
    error: Optional[str],
    error_type: Optional[str],
) -> Dict[str, Optional[str]]:
    msg = (error or "").lower()
    status_code = _parse_error_status_code(error)

    if status_code == 401 or "invalid api key" in msg or "unauthorized" in msg:
        return {"issue_code": "invalid_api_key", "severity": "failed", "status_code": str(status_code or 401)}
    if status_code == 403 or "forbidden" in msg:
        return {"issue_code": "forbidden", "severity": "failed", "status_code": str(status_code or 403)}
    if status_code == 404 or ("model" in msg and "not found" in msg):
        return {"issue_code": "model_not_found", "severity": "failed", "status_code": str(status_code or 404)}
    if "insufficient_quota" in msg or "insufficient quota" in msg or "credits" in msg or "billing" in msg:
        return {"issue_code": "insufficient_credits", "severity": "failed", "status_code": str(status_code or 429)}
    if status_code == 429 or error_type == "rate_limit" or "rate limit" in msg:
        return {"issue_code": "rate_limited", "severity": "degraded", "status_code": str(status_code or 429)}
    if error:
        return {"issue_code": "access_error", "severity": "failed", "status_code": str(status_code) if status_code else None}
    return {"issue_code": None, "severity": None, "status_code": None}


def _remediation_for_issue(issue_code: Optional[str]) -> List[str]:
    mapping = {
        "invalid_api_key": [
            "Set a valid API key (`--api-key` or provider env var).",
            "Verify key has access to the configured base URL/provider.",
        ],
        "forbidden": [
            "Check account/project permissions for this endpoint/model.",
            "Verify organization/project scoping for the API key.",
        ],
        "model_not_found": [
            "Verify `--model-name` exactly matches provider model ID.",
            "Check `/models` output for available model IDs.",
        ],
        "insufficient_credits": [
            "Check account billing/quota and add credits.",
            "Retry probe after quota replenishment.",
        ],
        "rate_limited": [
            "Retry probe after cooldown or lower concurrency.",
            "Check provider rate limits for your key/project.",
        ],
        "access_error": [
            "Check provider response body/error details in probe report.",
            "Validate base URL and provider compatibility.",
        ],
    }
    return mapping.get(issue_code or "", ["Inspect provider error details in probe_report.json."])


async def _probe_access_readiness(
    connection_info: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Preflight access check for auth/model/quota readiness."""
    from ..interfaces.openai_interface import OpenAIInterface

    probe_info = deepcopy(connection_info)
    probe_info["api_endpoint"] = "chat"
    probe_info["temperature"] = 0.0
    probe_info["max_tokens"] = 16

    interface = OpenAIInterface(probe_info, model_name)
    request = {
        "system_prompt": "",
        "user_prompt": "Reply with OK.",
        "schema": None,
        "sample_id": "probe_access_readiness",
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
    error = result.get("error")
    error_type = result.get("error_type")
    classification = _classify_access_issue(error, error_type)
    issue_code = classification["issue_code"]
    severity = classification["severity"]
    status_code = classification["status_code"]

    status = "ok"
    if severity == "failed":
        status = "failed"
    elif severity == "degraded":
        status = "degraded"

    model_listed = None
    models_endpoint_status = None
    try:
        base_url = connection_info.get("base_url")
        if base_url:
            import httpx

            api_key = connection_info.get("api_key") or "EMPTY"
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{base_url}/models",
                    timeout=10,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                models_endpoint_status = resp.status_code
                if resp.status_code == 200:
                    payload = resp.json()
                    ids = [
                        m.get("id")
                        for m in payload.get("data", [])
                        if isinstance(m, dict) and m.get("id")
                    ]
                    model_listed = model_name in ids
    except Exception:
        pass

    return {
        "status": status,
        "issue_code": issue_code,
        "error": error,
        "error_type": error_type,
        "evidence": {
            "status_code": int(status_code) if status_code else None,
            "models_endpoint_status": models_endpoint_status,
            "model_listed": model_listed,
            "response_received": bool(result.get("output") is not None),
            "finish_reason": result.get("finish_reason"),
        },
        "remediation": _remediation_for_issue(issue_code),
        "finish_reason": result.get("finish_reason"),
        "completion_tokens": result.get("completion_tokens"),
        "prompt_tokens": result.get("prompt_tokens"),
    }


def _build_test_plan(profile: str) -> Dict[str, Any]:
    checks = QUICK_PROFILE_CHECKS if profile == "quick" else FULL_PROFILE_CHECKS
    timeout_map = {name: CHECK_TIMEOUTS.get(name) for name in checks}
    return {
        "profile": profile,
        "checks": checks,
        "timeouts_s": timeout_map,
        "definitions": {
            "access_readiness": {
                "goal": "Fail fast on auth/model/quota blockers before benchmark runs",
                "tests": [
                    "minimal chat request to target model",
                    "best-effort /models lookup for listing/context",
                ],
                "pass_criteria": (
                    "No auth/quota/model-not-found blockers detected; "
                    "request returns usable output"
                ),
            },
            "request_modes": {
                "goal": "Detect chat/completions/logprobs endpoint behavior",
                "tests": [
                    "chat: plain text response",
                    "completions: plain text response",
                    "logprobs: multiple-choice via completions logprobs",
                ],
                "pass_criteria": "Request returns usable output and no API/parse errors",
            },
            "schema_transports": {
                "goal": "Verify schema transport reliability",
                "tests": [
                    "structured_outputs basic schema request",
                    "response_format basic schema request",
                    "stress schema request (nested fields, optional multimodal input)",
                ],
                "pass_criteria": (
                    "No errors, no truncation finish_reason=length, parsed JSON object, "
                    "no repetition pattern, stress check passes when run"
                ),
            },
            "multimodal": {
                "goal": "Check image+text inference path",
                "tests": ["chat request with bundled test image"],
                "pass_criteria": "Model accepts image input and returns non-empty output",
            },
            "truncation": {
                "goal": "Detect degenerate repetition under truncation",
                "tests": ["chat request with low max_tokens to induce truncation"],
                "pass_criteria": "No repetition patterns when truncated",
            },
            "param_support": {
                "goal": "Detect max output token parameter name",
                "tests": [
                    "chat request with max_tokens",
                    "fallback chat request with max_completion_tokens",
                ],
                "pass_criteria": "At least one max token parameter works",
            },
        },
    }


def _known_blindspots() -> List[str]:
    return [
        "Probe uses synthetic prompts/schemas; dataset-specific schemas can still fail.",
        "Schema reliability is sampled with a small number of requests, not a distribution.",
        "Prompt-token pressure in real tasks may be much higher than probe scenarios.",
        "Provider-side transient load can cause occasional false negatives.",
        "Parameter check validates max token key only; it does not validate every request option.",
        "Access readiness relies on provider error strings/status shape for issue classification.",
    ]


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
    checks = report.get("checks", {})
    access_check = checks.get("access_readiness", {})
    if isinstance(access_check, dict) and access_check.get("status") == "failed":
        return "failed"

    modes = report.get("modes", {})
    tested_modes = [m for m in modes.values() if isinstance(m, dict) and m.get("tested")]
    if tested_modes and not any(m.get("ok") for m in tested_modes):
        return "failed"

    transports = report.get("schema_transports", {})
    tested_transports = [t for t in transports.values() if isinstance(t, dict) and t.get("tested")]

    if tested_transports and not any(t.get("ok") for t in tested_transports):
        return "degraded"

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
        status = _format_probe_status(result if isinstance(result, dict) else {})
        lines.append(f"  {status} {mode.capitalize()} endpoint")
    
    # Schema transports
    transports = report.get("schema_transports", {})
    has_transport_warn = False
    for transport, result in transports.items():
        status = _format_schema_transport_status(result if isinstance(result, dict) else {})
        if status == "[WARN]":
            has_transport_warn = True
        lines.append(
            f"  {status} {transport.replace('_', ' ').title()} "
            f"(schema parameter format)"
        )
    
    # Additional checks
    checks = report.get("checks", {})
    for check_name, result in checks.items():
        if isinstance(result, dict):
            status_map = {"ok": "[OK]", "degraded": "[WARN]", "failed": "[FAIL]"}
            status = status_map.get(result.get("status"), "[?]")
        else:
            status = "[FAIL]"
        display_name = CHECK_DISPLAY_NAMES.get(check_name, check_name.replace("_", " ").title())
        if check_name == "param_support" and isinstance(result, dict):
            accepted = (result.get("evidence") or {}).get("max_tokens_param")
            suffix = f" (selected: {accepted})" if accepted else ""
            lines.append(f"  {status} {display_name}{suffix}")
        else:
            lines.append(f"  {status} {display_name}")

    if has_transport_warn:
        lines.append("")
        lines.append("Notes:")
        lines.append("  [WARN] schema parameter format: accepted by API but unreliable for eval")
    
    lines.append("")
    lines.append("Selected Configuration:")
    lines.append(f"  API endpoint: {report['selected_api_endpoint']}")
    lines.append(f"  Schema transport: {report.get('selected_schema_transport') or 'none'}")
    lines.append("  Schema transport options:")
    for transport_name, transport_result in transports.items():
        option_status = _schema_transport_option_label(transport_result if isinstance(transport_result, dict) else {})
        option_error = ""
        if isinstance(transport_result, dict):
            err = transport_result.get("error")
            if err:
                option_error = f" ({err})"
        lines.append(f"    - {transport_name}: {option_status}{option_error}")
    
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

    details = _collect_failure_details(report)
    if details:
        lines.append("")
        lines.append("Failure Details:")
        for detail in details:
            lines.append(f"  - {detail}")

    lines.append("")
    lines.append("What Was Tested:")
    lines.append("  See probe_report.json -> test_plan for exact probes and pass criteria.")
    lines.append("  Param check validates max token parameter only (not temperature).")
    lines.append("  Transport = schema parameter format sent to the provider API.")
    
    return "\n".join(lines)


def _get_benchy_version() -> str:
    """Get benchy version string."""
    try:
        from .. import __version__
        return __version__
    except:
        return "0.1.0"


def _format_probe_status(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "[FAIL]"
    if not result.get("tested"):
        return "[SKIP]"
    return "[OK]" if result.get("ok") else "[FAIL]"


def _format_schema_transport_status(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "[FAIL]"
    if not result.get("tested"):
        return "[SKIP]"
    if result.get("ok"):
        return "[OK]"
    if result.get("accepted_by_api") and not result.get("reliable_for_eval"):
        return "[WARN]"
    return "[FAIL]"


def _collect_failure_details(report: Dict[str, Any]) -> List[str]:
    details: List[str] = []

    for mode, result in (report.get("modes") or {}).items():
        if not isinstance(result, dict) or not result.get("tested") or result.get("ok"):
            continue
        err = result.get("error") or "unknown error"
        err_type = result.get("error_type")
        details.append(f"mode.{mode} failed: {err}" + (f" [{err_type}]" if err_type else ""))

    for transport, result in (report.get("schema_transports") or {}).items():
        if not isinstance(result, dict) or not result.get("tested") or result.get("ok"):
            continue
        err = result.get("error") or "unknown error"
        err_type = result.get("error_type")
        evidence = result.get("evidence") or {}
        stress = evidence.get("stress_ok")
        stress_note = f", stress_ok={stress}" if stress is not None else ""
        accepted = result.get("accepted_by_api")
        reliable = result.get("reliable_for_eval")
        state_note = ""
        if accepted is not None or reliable is not None:
            state_note = f", accepted_by_api={accepted}, reliable_for_eval={reliable}"
        state = "accepted_but_unreliable" if accepted and not reliable else "failed"
        details.append(
            f"schema.{transport} {state}: {err}"
            + (f" [{err_type}]" if err_type else "")
            + stress_note
            + state_note
        )

    for check_name, result in (report.get("checks") or {}).items():
        if not isinstance(result, dict):
            continue
        status = result.get("status")
        if status not in {"failed", "degraded"}:
            continue
        err = result.get("error") or "warning/no error message"
        issue_code = result.get("issue_code")
        issue_suffix = f" issue={issue_code}" if issue_code else ""
        details.append(f"check.{check_name} {status}:{issue_suffix} {err}".rstrip())
        remediation = result.get("remediation") or []
        if isinstance(remediation, list):
            for step in remediation[:2]:
                details.append(f"  remediation: {step}")

    return details


def _schema_transport_option_label(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict) or not result.get("tested"):
        return "not_tested"
    if result.get("ok"):
        return "usable"
    if result.get("accepted_by_api") and not result.get("reliable_for_eval"):
        return "accepted_but_unreliable"
    return "unsupported_or_failed"
