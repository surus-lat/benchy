"""Tests for the audio capability dimension (supports_audio / requires_audio)."""

from src.engine.protocols import (
    InterfaceCapabilities,
    RequirementLevel,
    TaskCapabilityRequirements,
    check_compatibility,
    get_task_requirements,
    parse_interface_capabilities,
)


class _TaskBase:
    config = {}
    capability_requirements = None
    requires_multimodal = False
    requires_schema = False
    requires_files = False
    requires_logprobs = False
    requires_audio = False
    answer_type = "freeform"


class _Interface:
    def __init__(self, capabilities: InterfaceCapabilities):
        self.capabilities = capabilities


def test_parse_interface_capabilities_reads_supports_audio() -> None:
    parsed = parse_interface_capabilities({"supports_audio": True})
    assert parsed.supports_audio is True


def test_parse_interface_capabilities_supports_audio_default_false() -> None:
    parsed = parse_interface_capabilities(None)
    assert parsed.supports_audio is False


def test_task_capability_requirements_default_requires_audio_optional() -> None:
    req = TaskCapabilityRequirements()
    assert req.requires_audio == RequirementLevel.OPTIONAL


def test_get_task_requirements_promotes_requires_audio_to_required() -> None:
    class Task(_TaskBase):
        requires_audio = True

    req = get_task_requirements(Task())
    assert req.requires_audio == RequirementLevel.REQUIRED


def test_get_task_requirements_audio_override_from_config() -> None:
    class Task(_TaskBase):
        config = {"capability_requirements": {"requires_audio": "preferred"}}

    req = get_task_requirements(Task())
    assert req.requires_audio == RequirementLevel.PREFERRED


def test_check_compatibility_errors_when_audio_required_unsupported() -> None:
    class Task(_TaskBase):
        requires_audio = True

    interface = _Interface(InterfaceCapabilities(supports_audio=False))
    report = check_compatibility(Task(), interface)

    assert report.compatible is False
    assert any("requires audio inputs" in err for err in report.errors)


def test_check_compatibility_warns_when_audio_preferred_unsupported() -> None:
    class Task(_TaskBase):
        capability_requirements = {"requires_audio": "preferred"}

    interface = _Interface(InterfaceCapabilities(supports_audio=False))
    report = check_compatibility(Task(), interface)

    assert report.compatible is True
    assert any("prefers audio inputs" in w for w in report.warnings)


def test_check_compatibility_passes_when_audio_supported() -> None:
    class Task(_TaskBase):
        requires_audio = True

    interface = _Interface(InterfaceCapabilities(supports_audio=True))
    report = check_compatibility(Task(), interface)

    assert report.compatible is True
    assert not any("audio" in err for err in report.errors)


def test_openai_audio_provider_defaults_advertise_audio() -> None:
    from src.engine.connection import PROVIDER_CAPABILITY_DEFAULTS

    caps = PROVIDER_CAPABILITY_DEFAULTS["openai_audio"]
    assert caps.supports_audio is True
    assert caps.supports_multimodal is False


def test_transformers_audio_provider_defaults_advertise_audio() -> None:
    from src.engine.connection import PROVIDER_CAPABILITY_DEFAULTS

    caps = PROVIDER_CAPABILITY_DEFAULTS["transformers_audio"]
    assert caps.supports_audio is True
    assert caps.supports_multimodal is False
    assert caps.supports_schema is False


def test_transformers_audio_provider_gates_audio_required_task() -> None:
    from src.engine.connection import PROVIDER_CAPABILITY_DEFAULTS

    class Task(_TaskBase):
        requires_audio = True

    interface = _Interface(PROVIDER_CAPABILITY_DEFAULTS["transformers_audio"])
    report = check_compatibility(Task(), interface)
    assert report.compatible is True
