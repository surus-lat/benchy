"""BaseAdapter — per-model-family inference contract.

Surface mirrors BaseInterface so BenchmarkRunner consumes either
flavor unchanged. Subclasses set `capabilities` (class attribute)
and implement `generate_batch`; `prepare_request` has an identity
default so adapters that don't need it can skip it entirely.

See docs/superpowers/specs/2026-06-19-adapter-layer-design.md for
the design rationale.
"""

from __future__ import annotations

from typing import Any

from src.engine.protocols import InterfaceCapabilities


class BaseAdapter:
    """Abstract per-model-family adapter.

    Subclass contract:
      - Override the `capabilities` class attribute with a populated
        InterfaceCapabilities.
      - Implement `async generate_batch(requests) -> list[dict]`
        returning one `{output, raw, error, error_type}` dict per
        request.
      - Optionally override `prepare_request` to transform samples
        before batching.
    """

    capabilities: InterfaceCapabilities = InterfaceCapabilities()

    def __init__(self, model_name: str, config: dict[str, Any]):
        self.model_name = model_name
        self.config = config

    def prepare_request(self, sample: dict[str, Any], task: Any) -> dict[str, Any]:
        """Identity default. Adapters override only if they need to
        peek, transform, or attach metadata before generate_batch."""
        return sample

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run inference on a batch. Subclasses implement.

        Returns one result dict per request:
        ``{"output": str, "raw": Any, "error": str | None, "error_type": str | None}``.

        Partial-batch failures: failed items have non-empty `error`
        and `error_type`; successful items have non-empty `output`.
        """
        raise NotImplementedError(
            "BaseAdapter.generate_batch must be overridden by a subclass."
        )
