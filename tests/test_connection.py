from src.engine.connection import build_connection_info


def test_openai_connection_defaults_include_expected_capabilities() -> None:
    info = build_connection_info("openai", provider_config={}, model_config={})

    assert info["base_url"] == "https://api.openai.com/v1"
    assert info["api_key_env"] == "OPENAI_API_KEY"
    assert info["use_structured_outputs"] is False
    assert info["supports_logprobs"] is False
    assert info["capabilities"]["supports_schema"] is True
    assert info["capabilities"]["request_modes"] == ["chat", "completions"]


def test_model_capabilities_only_restrict_provider_capabilities() -> None:
    info = build_connection_info(
        "openai",
        provider_config={
            "capabilities": {
                "supports_multimodal": False,
                "request_modes": ["chat", "completions"],
            },
            "model_capabilities": {
                "supports_multimodal": True,
                "request_modes": ["chat"],
            },
        },
        model_config={},
    )

    assert info["capabilities"]["supports_multimodal"] is False
    assert info["capabilities"]["request_modes"] == ["chat"]


def test_explicit_api_key_is_preserved() -> None:
    info = build_connection_info(
        "openai",
        provider_config={"api_key": "test-key"},
        model_config={},
    )

    assert info["api_key"] == "test-key"


def test_vllm_structured_outputs_default_and_override() -> None:
    default_info = build_connection_info("vllm", provider_config={}, model_config={})
    assert default_info["use_structured_outputs"] is True

    overridden_info = build_connection_info(
        "vllm",
        provider_config={"use_structured_outputs": False},
        model_config={},
    )
    assert overridden_info["use_structured_outputs"] is False


def test_alibaba_connection_defaults_include_expected_capabilities() -> None:
    info = build_connection_info("alibaba", provider_config={}, model_config={})

    assert info["base_url"] == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    assert info["api_key_env"] == "DASHSCOPE_API_KEY"
    assert info["use_structured_outputs"] is False
    assert info["supports_logprobs"] is False
    assert info["capabilities"]["supports_multimodal"] is True
    assert info["capabilities"]["request_modes"] == ["chat", "completions"]
