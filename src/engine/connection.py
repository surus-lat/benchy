"""Connection info utilities for the benchmark engine.

This module provides functions to build connection_info dicts from
various provider configurations. The connection_info is a standard
format that interfaces use - they don't need to know about providers.
"""

from dataclasses import asdict, replace
from typing import Dict, Any, Optional
import logging

from .protocols import InterfaceCapabilities, parse_interface_capabilities

logger = logging.getLogger(__name__)

PROVIDER_CAPABILITY_DEFAULTS = {
    "vllm": InterfaceCapabilities(
        supports_multimodal=False,
        supports_logprobs=True,
        supports_schema=True,
        supports_files=False,
        supports_streaming=False,
        request_modes=["chat", "completions"],
    ),
    "openai": InterfaceCapabilities(
        supports_multimodal=True,
        supports_logprobs=False,
        supports_schema=True,
        supports_files=True,
        supports_streaming=False,
        request_modes=["chat", "completions"],
    ),
    "anthropic": InterfaceCapabilities(
        supports_multimodal=False,
        supports_logprobs=False,
        supports_schema=True,
        supports_files=False,
        supports_streaming=False,
        request_modes=["chat"],
    ),
    "together": InterfaceCapabilities(
        # Together's API is OpenAI-compatible; vision support is model-dependent.
        # Treat the provider as capable and restrict per-model via model_capabilities.
        supports_multimodal=True,
        supports_logprobs=False,
        supports_schema=True,
        supports_files=True,
        supports_streaming=False,
        request_modes=["chat", "completions"],
    ),
    "surus": InterfaceCapabilities(
        supports_multimodal=False,
        supports_logprobs=False,
        supports_schema=True,
        supports_files=False,
        supports_streaming=False,
        request_modes=["raw_payload"],
    ),
    "surus_ocr": InterfaceCapabilities(
        supports_multimodal=True,
        supports_logprobs=False,
        supports_schema=True,
        supports_files=True,
        supports_streaming=False,
        request_modes=["raw_payload"],
    ),
    "surus_factura": InterfaceCapabilities(
        supports_multimodal=True,
        supports_logprobs=False,
        supports_schema=False,
        supports_files=True,
        supports_streaming=False,
        request_modes=["raw_payload"],
    ),
    "surus_classify": InterfaceCapabilities(
        supports_multimodal=False,
        supports_logprobs=False,
        supports_schema=False,
        supports_files=False,
        supports_streaming=False,
        request_modes=["raw_payload"],
    ),
    "http": InterfaceCapabilities(
        supports_multimodal=False,
        supports_logprobs=False,
        supports_schema=True,
        supports_files=False,
        supports_streaming=False,
        request_modes=["raw_payload"],
    ),
    "google": InterfaceCapabilities(
        supports_multimodal=True,
        supports_logprobs=False,
        supports_schema=False,
        supports_files=True,
        supports_streaming=False,
        request_modes=["raw_payload"],
    ),
}

_CAPABILITY_FIELDS = [
    "supports_multimodal",
    "supports_logprobs",
    "supports_schema",
    "supports_files",
    "supports_streaming",
    "supports_batch",
]


def _merge_capabilities(
    provider_caps: InterfaceCapabilities,
    model_caps: Optional[Dict[str, Any]],
) -> InterfaceCapabilities:
    """Combine provider (stack) and model capabilities.

    Model capabilities can only restrict provider capabilities. They never
    enable a feature that the provider does not support.
    """
    if not model_caps:
        return provider_caps

    effective = provider_caps
    for field in _CAPABILITY_FIELDS:
        if field in model_caps:
            effective = replace(
                effective,
                **{field: getattr(provider_caps, field) and bool(model_caps[field])},
            )

    if "request_modes" in model_caps:
        model_modes = model_caps.get("request_modes") or []
        provider_modes = provider_caps.request_modes or []
        if provider_modes:
            effective = replace(
                effective,
                request_modes=[mode for mode in provider_modes if mode in model_modes],
            )
        else:
            effective = replace(effective, request_modes=model_modes)

    return effective

def build_connection_info(
    provider_type: str,
    provider_config: Optional[Dict[str, Any]] = None,
    server_info: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standardized connection_info dict from provider configuration.
    
    This function resolves the various ways connection info can be provided
    (vLLM server_info, OpenAI config, Anthropic config, etc.) into a single
    standard format that interfaces can use.
    
    Args:
        provider_type: Type of provider ('vllm', 'openai', 'anthropic', 'surus')
        provider_config: Provider-specific configuration dict
        server_info: vLLM server info (for vllm provider)
        model_config: Model-level config overrides
        
    Returns:
        Standardized connection_info dict with:
        - base_url: API endpoint URL
        - api_key: API key (or None to read from env)
        - api_key_env: Environment variable for API key
        - timeout: Request timeout
        - max_retries: Max retry attempts
        - temperature: Generation temperature
        - max_tokens: Max tokens to generate
        - use_structured_outputs: Whether to use vLLM's structured outputs (vllm only)
    """
    provider_config = provider_config or {}
    model_config = model_config or {}
    
    # Build base connection info
    # Priority: provider_config (with model overrides) > task defaults (model_config)
    # This ensures model-specific requirements (like gpt-5-mini temperature) take precedence
    connection_info = {
        "provider_type": provider_type,
        "timeout": provider_config.get("timeout", 120),
        "max_retries": provider_config.get("max_retries", 3),
        "temperature": provider_config.get("temperature", model_config.get("temperature", 0.0)),
        "max_tokens": provider_config.get("max_tokens", model_config.get("max_tokens", 2048)),
        "max_tokens_param_name": provider_config.get("max_tokens_param_name", model_config.get("max_tokens_param_name", "max_tokens")),
        "max_concurrent": provider_config.get("max_concurrent", model_config.get("max_concurrent")),
        "api_endpoint": provider_config.get("api_endpoint", model_config.get("api_endpoint")),
        "logprobs_top_k": provider_config.get("logprobs_top_k", model_config.get("logprobs_top_k")),
        "problematic_models": provider_config.get("problematic_models", model_config.get("problematic_models")),
        # Image artifact generation (OpenAI-compatible images endpoints).
        "image_response_format": provider_config.get("image_response_format", model_config.get("image_response_format")),
        "image_size": provider_config.get("image_size", model_config.get("image_size")),
        "image_max_edge": provider_config.get("image_max_edge", model_config.get("image_max_edge")),
        "image_artifact_fallback_to_chat": provider_config.get(
            "image_artifact_fallback_to_chat",
            model_config.get("image_artifact_fallback_to_chat", True),
        ),
    }
    # Preserve explicit API keys (e.g. --api-key on CLI). Interfaces will
    # still fall back to api_key_env if this is missing/empty.
    if provider_config.get("api_key") is not None:
        connection_info["api_key"] = provider_config.get("api_key")

    base_capabilities = PROVIDER_CAPABILITY_DEFAULTS.get(provider_type, InterfaceCapabilities())
    provider_capabilities = parse_interface_capabilities(
        provider_config.get("capabilities"),
        default=base_capabilities,
    )
    model_capabilities = provider_config.get("model_capabilities") or {}
    legacy_overrides: Dict[str, Any] = {}
    if "supports_logprobs" in provider_config:
        legacy_overrides["supports_logprobs"] = provider_config.get("supports_logprobs")
    if legacy_overrides:
        provider_capabilities = parse_interface_capabilities(legacy_overrides, default=provider_capabilities)
    
    if provider_type == "vllm":
        # vLLM: Get URL from server_info
        if server_info:
            connection_info["base_url"] = server_info["url"] + "/v1"
        else:
            # Fallback to config-based URL
            host = provider_config.get("host", "localhost")
            port = provider_config.get("port", 8000)
            connection_info["base_url"] = f"http://{host}:{port}/v1"
        
        connection_info["api_key"] = "EMPTY"  # vLLM doesn't need real key
        # vLLM v0.12.0+ supports schema-guided structured outputs, but allow opting out
        # for debugging/perf (falls back to prompt-only schema enforcement + validation).
        connection_info["use_structured_outputs"] = bool(provider_config.get("use_structured_outputs", True))
    elif provider_type == "openai":
        connection_info["base_url"] = provider_config.get("base_url", "https://api.openai.com/v1")
        connection_info["api_key_env"] = provider_config.get("api_key_env", "OPENAI_API_KEY")
        connection_info["use_structured_outputs"] = False
        
    elif provider_type == "anthropic":
        connection_info["base_url"] = provider_config.get("base_url", "https://api.anthropic.com/v1")
        connection_info["api_key_env"] = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
        connection_info["use_structured_outputs"] = False
    
    elif provider_type == "together":
        connection_info["base_url"] = provider_config.get("base_url", "https://api.together.xyz/v1")
        connection_info["api_key_env"] = provider_config.get("api_key_env", "TOGETHER_API_KEY")
        connection_info["use_structured_outputs"] = False
        
    elif provider_type == "surus":
        connection_info["base_url"] = provider_config.get("endpoint", provider_config.get("base_url"))
        connection_info["api_key_env"] = provider_config.get("api_key_env", "SURUS_API_KEY")
        connection_info["use_structured_outputs"] = False
        
    elif provider_type == "surus_ocr":
        connection_info["base_url"] = provider_config.get("endpoint", provider_config.get("base_url"))
        connection_info["api_key_env"] = provider_config.get("api_key_env", "SURUS_API_KEY")
        connection_info["use_structured_outputs"] = False
        
    elif provider_type == "surus_factura":
        connection_info["base_url"] = provider_config.get("endpoint", provider_config.get("base_url"))
        connection_info["api_key_env"] = provider_config.get("api_key_env", "SURUS_API_KEY")
        connection_info["use_structured_outputs"] = False

    elif provider_type == "surus_classify":
        connection_info["base_url"] = provider_config.get("endpoint", provider_config.get("base_url"))
        connection_info["api_key_env"] = provider_config.get("api_key_env", "SURUS_API_KEY")
        connection_info["use_structured_outputs"] = False

    elif provider_type == "surus_remove_background":
        connection_info["base_url"] = provider_config.get("endpoint", provider_config.get("base_url"))
        connection_info["api_key_env"] = provider_config.get("api_key_env", "SURUS_API_KEY")
        connection_info["use_structured_outputs"] = False
    
    elif provider_type == "google":
        connection_info["base_url"] = provider_config.get("endpoint", provider_config.get("base_url", "https://generativelanguage.googleapis.com/v1"))
        connection_info["api_key_env"] = provider_config.get("api_key_env", "GOOGLE_API_KEY")
        connection_info["use_structured_outputs"] = False
        
    else:
        # Generic HTTP provider
        connection_info["base_url"] = provider_config.get("base_url", provider_config.get("endpoint"))
        connection_info["api_key_env"] = provider_config.get("api_key_env")
        connection_info["use_structured_outputs"] = False

    capabilities = _merge_capabilities(provider_capabilities, model_capabilities)
    connection_info["capabilities"] = asdict(capabilities)
    connection_info["supports_logprobs"] = capabilities.supports_logprobs
    
    logger.debug(f"Built connection_info for {provider_type}: base_url={connection_info.get('base_url')}")
    return connection_info


def get_interface_for_provider(
    provider_type: str,
    connection_info: Dict[str, Any],
    model_name: str,
):
    """Get the appropriate interface instance for a provider type.
    
    Args:
        provider_type: Type of provider
        connection_info: Connection info dict from build_connection_info()
        model_name: Name of the model
        
    Returns:
        Interface instance
    """
    if provider_type == "surus":
        from ..interfaces.surus.surus_interface import SurusInterface
        # SurusInterface has its own config format, adapt connection_info
        surus_config = {
            "surus": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 30),
                "max_retries": connection_info.get("max_retries", 3),
                "image_max_edge": connection_info.get("image_max_edge"),
                "capabilities": connection_info.get("capabilities"),
            }
        }
        return SurusInterface(surus_config, model_name, "surus")
    
    elif provider_type == "surus_ocr":
        from ..interfaces.surus.surus_ocr_interface import SurusOCRInterface
        # SurusOCRInterface has its own config format, adapt connection_info
        surus_ocr_config = {
            "surus_ocr": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 60),
                "max_retries": connection_info.get("max_retries", 3),
                "image_max_edge": connection_info.get("image_max_edge"),
                "capabilities": connection_info.get("capabilities"),
            }
        }
        return SurusOCRInterface(surus_ocr_config, model_name, "surus_ocr")
    
    elif provider_type == "surus_factura":
        from ..interfaces.surus.surus_factura_interface import SurusFacturaInterface
        # SurusFacturaInterface has its own config format, adapt connection_info
        surus_factura_config = {
            "surus_factura": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 60),
                "max_retries": connection_info.get("max_retries", 3),
                "image_max_edge": connection_info.get("image_max_edge"),
                "capabilities": connection_info.get("capabilities"),
            }
        }
        return SurusFacturaInterface(surus_factura_config, model_name, "surus_factura")

    elif provider_type == "surus_classify":
        from ..interfaces.surus.surus_classify_interface import SurusClassifyInterface
        surus_classify_config = {
            "surus_classify": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 30),
                "max_retries": connection_info.get("max_retries", 3),
                "image_max_edge": connection_info.get("image_max_edge"),
                "capabilities": connection_info.get("capabilities"),
            }
        }
        return SurusClassifyInterface(surus_classify_config, model_name, "surus_classify")

    elif provider_type == "surus_remove_background":
        from ..interfaces.surus.surus_remove_background_interface import SurusRemoveBackgroundInterface
        surus_remove_background_config = {
            "surus_remove_background": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 60),
                "max_retries": connection_info.get("max_retries", 3),
                "image_max_edge": connection_info.get("image_max_edge"),
                "capabilities": connection_info.get("capabilities"),
            }
        }
        return SurusRemoveBackgroundInterface(surus_remove_background_config, model_name, "surus_remove_background")
    
    elif provider_type == "google":
        # Use generic GoogleInterface (similar to OpenAIInterface pattern)
        from ..interfaces.google.google_interface import GoogleInterface
        return GoogleInterface(connection_info, model_name)
    
    else:
        # Use OpenAIInterface for vllm, openai, anthropic, together
        from ..interfaces.openai_interface import OpenAIInterface
        return OpenAIInterface(connection_info, model_name)
