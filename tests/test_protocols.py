from src.engine.protocols import (
    InterfaceCapabilities,
    RequirementLevel,
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
    answer_type = "freeform"


class _Interface:
    def __init__(self, capabilities: InterfaceCapabilities):
        self.capabilities = capabilities


def test_parse_interface_capabilities_applies_defaults() -> None:
    defaults = InterfaceCapabilities(supports_multimodal=True, supports_schema=True, request_modes=["chat"])
    parsed = parse_interface_capabilities({"supports_schema": False}, default=defaults)

    assert parsed.supports_multimodal is True
    assert parsed.supports_schema is False
    assert parsed.request_modes == ["chat"]


def test_get_task_requirements_from_task_flags() -> None:
    class Task(_TaskBase):
        requires_multimodal = True
        requires_schema = True

    req = get_task_requirements(Task())

    assert req.requires_multimodal == RequirementLevel.REQUIRED
    assert req.requires_schema == RequirementLevel.REQUIRED
    assert req.requires_logprobs == RequirementLevel.OPTIONAL


def test_get_task_requirements_with_config_overrides() -> None:
    class Task(_TaskBase):
        config = {
            "capability_requirements": {
                "requires_schema": "preferred",
                "requires_files": "required",
            }
        }

    req = get_task_requirements(Task())

    assert req.requires_schema == RequirementLevel.PREFERRED
    assert req.requires_files == RequirementLevel.REQUIRED


def test_check_compatibility_reports_errors_and_warnings() -> None:
    class Task(_TaskBase):
        capability_requirements = {
            "requires_schema": "required",
            "requires_logprobs": "preferred",
        }

    interface = _Interface(
        InterfaceCapabilities(
            supports_multimodal=True,
            supports_schema=False,
            supports_logprobs=False,
            supports_files=True,
            supports_streaming=False,
        )
    )

    report = check_compatibility(Task(), interface)

    assert report.compatible is False
    assert any("requires schemas" in e for e in report.errors)
    assert any("prefers logprobs" in w for w in report.warnings)
