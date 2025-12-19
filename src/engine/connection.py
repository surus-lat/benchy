"""Connection info utilities for the benchmark engine.

This module provides functions to build connection_info dicts from
various provider configurations. The connection_info is a standard
format that interfaces use - they don't need to know about providers.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


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
        "timeout": provider_config.get("timeout", 120),
        "max_retries": provider_config.get("max_retries", 3),
        "temperature": provider_config.get("temperature", model_config.get("temperature", 0.0)),
        "max_tokens": provider_config.get("max_tokens", model_config.get("max_tokens", 2048)),
        "max_tokens_param_name": provider_config.get("max_tokens_param_name", model_config.get("max_tokens_param_name", "max_tokens")),
    }
    
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
        connection_info["use_structured_outputs"] = True  # vLLM v0.12.0+ structured outputs
        
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
        
    else:
        # Generic HTTP provider
        connection_info["base_url"] = provider_config.get("base_url", provider_config.get("endpoint"))
        connection_info["api_key_env"] = provider_config.get("api_key_env")
        connection_info["use_structured_outputs"] = False
    
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
        from ..interfaces.surus_interface import SurusInterface
        # SurusInterface has its own config format, adapt connection_info
        surus_config = {
            "surus": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 30),
                "max_retries": connection_info.get("max_retries", 3),
            }
        }
        return SurusInterface(surus_config, model_name, "surus")
    
    elif provider_type == "surus_ocr":
        from ..interfaces.surus_ocr_interface import SurusOCRInterface
        # SurusOCRInterface has its own config format, adapt connection_info
        surus_ocr_config = {
            "surus_ocr": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 60),
                "max_retries": connection_info.get("max_retries", 3),
            }
        }
        return SurusOCRInterface(surus_ocr_config, model_name, "surus_ocr")
    
    elif provider_type == "surus_factura":
        from ..interfaces.surus_factura_interface import SurusFacturaInterface
        # SurusFacturaInterface has its own config format, adapt connection_info
        surus_factura_config = {
            "surus_factura": {
                "endpoint": connection_info["base_url"],
                "api_key_env": connection_info.get("api_key_env", "SURUS_API_KEY"),
                "timeout": connection_info.get("timeout", 60),
                "max_retries": connection_info.get("max_retries", 3),
            }
        }
        return SurusFacturaInterface(surus_factura_config, model_name, "surus_factura")
    
    else:
        # Use ChatCompletionsInterface for vllm, openai, anthropic
        from ..interfaces.chat_completions import ChatCompletionsInterface
        return ChatCompletionsInterface(connection_info, model_name)

