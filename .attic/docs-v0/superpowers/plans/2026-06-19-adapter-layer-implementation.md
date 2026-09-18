# Adapter Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-model-family Adapter layer specified in
`docs/superpowers/specs/2026-06-19-adapter-layer-design.md`, ship the
two v1 adapters (`voxtral_chat`, `qwen3_asr_chat`), and route them
through `connection.py` without touching any existing model YAML.

**Architecture:** `BaseAdapter` mirrors `BaseInterface` (one abstract
`generate_batch`, identity-default `prepare_request`, `capabilities`
class attribute reusing `InterfaceCapabilities`). Discovery is via
`pkgutil.iter_modules` + `importlib.import_module` — no central
registry file. `engine/connection.py` grows one early-return branch
when `adapter:` is present in the model config; existing routing
unchanged otherwise. Two v1 adapters load their models lazily through
`AutoModelForCausalLM` + `AutoProcessor` with `trust_remote_code=True`
and build chat-template inputs.

**Tech Stack:** Python 3.12, pytest, `transformers >= 4.45`, `torch`,
`huggingface-hub`. The `[transcription]` extras group already covers
these.

## Global Constraints

- The contract Tasks see is identical for Adapters and Interfaces:
  `prepare_request(sample, task)` + `async generate_batch(requests)`
  returning `list[{output, raw, error, error_type}]`. BenchmarkRunner
  must work without any change.
- Adapters live in `src/adapters/<name>.py`. Each file exports a class
  literally named `Adapter`. No `__all__`, no registration call.
- `BaseAdapter` has exactly one abstract method (`generate_batch`),
  one default-identity method (`prepare_request`), and one class
  attribute (`capabilities`). Nothing else in the protocol.
- `list_adapters()` excludes modules whose name starts with `_` or
  equals `base`. The filesystem listing IS the registry.
- `connection.py` adds at most a 6-line early-return branch. No
  function rename, no signature change.
- Existing model YAMLs are not edited. Adapter routing only fires when
  `"adapter"` is a top-level key in the model config.
- Tests must not download model weights. Use `unittest.mock.patch` on
  `transformers.AutoModelForCausalLM.from_pretrained` and
  `transformers.AutoProcessor.from_pretrained` with fakes that return
  predictable token tensors and decoded strings.
- All `transformers` imports inside adapters happen lazily inside
  `_load()` (called from `generate_batch`), never at module-import
  time. This keeps the modules importable in test environments
  without `torch` installed and lets tests patch the imports.
- Adapter v1 capabilities for both `voxtral_chat` and `qwen3_asr_chat`:
  `supports_audio_input=True, supports_text_output=True`, everything
  else `False`. Use exact field names from `InterfaceCapabilities` in
  `src/engine/protocols.py:14`.
- Each task ends green: `pytest -q` clean and the newly-added test
  passing. Never commit a red test.
- Commit prefix: `feat:` for new code that adds capability,
  `chore:` for YAML rewrites and small wiring, `docs:` for plan
  updates.

---

## File Map

| Path | Action | Owner task |
|---|---|---|
| `src/adapters/__init__.py` | create | Task 2 |
| `src/adapters/base.py` | create | Task 1 |
| `src/adapters/voxtral_chat.py` | create | Task 4 |
| `src/adapters/qwen3_asr_chat.py` | create | Task 5 |
| `src/engine/connection.py` | modify (+~6 lines) | Task 3 |
| `configs/models/voxtral-mini-4b-transformers.yaml` | rewrite | Task 6 |
| `configs/models/qwen3-asr-0.6b-transformers.yaml` | rewrite | Task 6 |
| `tests/test_adapters_base.py` | create | Task 1 |
| `tests/test_adapters_registry.py` | create | Task 2 |
| `tests/test_adapters_connection.py` | create | Task 3 |
| `tests/test_adapters_voxtral_chat.py` | create | Task 4 |
| `tests/test_adapters_qwen3_asr_chat.py` | create | Task 5 |

---

## Task 1: `BaseAdapter` and its test

**Files:**
- Create: `src/adapters/base.py`
- Test: `tests/test_adapters_base.py`

**Interfaces:**
- Consumes: `InterfaceCapabilities` from `src/engine/protocols.py:14`.
- Produces: `BaseAdapter` class with three members — `capabilities` (class attr, type `InterfaceCapabilities`), `prepare_request(sample, task) -> dict` (identity default), `async generate_batch(requests) -> list[dict]` (abstract).

- [ ] **Step 1: Write the failing test**

Write `tests/test_adapters_base.py`:

```python
"""Tests for BaseAdapter shape and defaults."""
import asyncio
import pytest

from src.adapters.base import BaseAdapter
from src.engine.protocols import InterfaceCapabilities


def test_base_adapter_generate_batch_is_abstract():
    """Calling generate_batch on BaseAdapter raises NotImplementedError."""
    adapter = BaseAdapter("fake-model", {})
    with pytest.raises(NotImplementedError):
        asyncio.run(adapter.generate_batch([{"id": "x"}]))


def test_base_adapter_prepare_request_is_identity():
    """Default prepare_request returns the sample unchanged."""
    adapter = BaseAdapter("fake-model", {})
    sample = {"id": "s1", "audio_path": "/tmp/f.wav"}
    assert adapter.prepare_request(sample, task=None) is sample


def test_base_adapter_stores_model_name_and_config():
    """Constructor stashes its arguments as instance attributes."""
    cfg = {"some_knob": True}
    adapter = BaseAdapter("hf/repo", cfg)
    assert adapter.model_name == "hf/repo"
    assert adapter.config is cfg


def test_subclass_can_declare_capabilities_class_attribute():
    """Subclasses set `capabilities` as a class attribute typed
    InterfaceCapabilities — same pattern existing Interfaces use."""

    class _SubAdapter(BaseAdapter):
        capabilities = InterfaceCapabilities(
            supports_audio_input=True,
            supports_text_output=True,
        )

        async def generate_batch(self, requests):
            return [{"output": "ok", "raw": "ok", "error": None, "error_type": None}]

    sub = _SubAdapter("fake", {})
    assert sub.capabilities.supports_audio_input is True
    result = asyncio.run(sub.generate_batch([{"id": "x"}]))
    assert result[0]["output"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.adapters'`.

- [ ] **Step 3: Write minimal implementation**

Write `src/adapters/base.py`:

```python
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

    # Default — subclasses override.
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
```

The package needs `__init__.py` so imports work — create an empty one for now; Task 2 fills it in:

```bash
mkdir -p src/adapters
touch src/adapters/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adapters_base.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Confirm the full suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: `370+ passed, 1 xfailed` (the 4 new tests bring it to 374).

- [ ] **Step 6: Commit**

```bash
git add src/adapters/base.py src/adapters/__init__.py tests/test_adapters_base.py
git commit -m "feat: add BaseAdapter — per-model-family inference contract

BaseAdapter mirrors BaseInterface so BenchmarkRunner consumes either
flavor unchanged. One abstract method (generate_batch), one
identity-default method (prepare_request), one class attribute
(capabilities) reusing InterfaceCapabilities."
```

---

## Task 2: Discovery — `list_adapters()` and `get_adapter()`

**Files:**
- Modify: `src/adapters/__init__.py`
- Test: `tests/test_adapters_registry.py`

**Interfaces:**
- Consumes: `BaseAdapter` from Task 1.
- Produces: `list_adapters() -> list[str]` returning adapter names found on disk (excluding `_*` and `base`); `get_adapter(name: str, model_name: str, config: dict) -> BaseAdapter` importing `src.adapters.<name>` and calling its `Adapter` class.

- [ ] **Step 1: Write the failing test**

Write `tests/test_adapters_registry.py`:

```python
"""Tests for adapter discovery via importlib + pkgutil."""
import pytest

import src.adapters
from src.adapters import list_adapters, get_adapter
from src.adapters.base import BaseAdapter


def test_list_adapters_excludes_base_and_underscore(monkeypatch, tmp_path):
    """Files starting with `_` or named `base` are not adapters."""
    (tmp_path / "voxtral_chat.py").write_text("")
    (tmp_path / "_private.py").write_text("")
    (tmp_path / "base.py").write_text("")
    (tmp_path / "qwen3_asr_chat.py").write_text("")
    (tmp_path / "__init__.py").write_text("")
    monkeypatch.setattr(src.adapters, "__path__", [str(tmp_path)])
    assert list_adapters() == ["qwen3_asr_chat", "voxtral_chat"]


def test_get_adapter_unknown_name_raises_with_available_list():
    """A typo'd adapter name surfaces a helpful error message."""
    with pytest.raises(ValueError) as exc_info:
        get_adapter("not_a_real_adapter", "model-x", {})
    msg = str(exc_info.value)
    assert "not_a_real_adapter" in msg
    assert "Available:" in msg


def test_get_adapter_error_message_lists_existing_adapters():
    """Error message must include the actual available list (not a stub)."""
    with pytest.raises(ValueError) as exc_info:
        get_adapter("definitely_not_real", "model-x", {})
    # list_adapters() is the source of truth — error and helper agree
    assert str(list_adapters()) in str(exc_info.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'list_adapters' from 'src.adapters'`.

- [ ] **Step 3: Write the implementation**

Replace `src/adapters/__init__.py` with:

```python
"""Adapter discovery and instantiation.

The filesystem is the registry: each adapter lives in
`src/adapters/<name>.py` and exports a class literally named
`Adapter`. No central registry file to keep in sync.

Names starting with `_` and the literal name `base` are reserved
for internal use and never listed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from src.adapters.base import BaseAdapter


def list_adapters() -> list[str]:
    """Return the sorted list of adapter names found on disk."""
    return sorted(
        m.name
        for m in pkgutil.iter_modules(__path__)
        if not m.name.startswith("_") and m.name != "base"
    )


def get_adapter(
    name: str, model_name: str, config: dict[str, Any]
) -> BaseAdapter:
    """Import the adapter module by name and instantiate its
    `Adapter` class.

    Raises ValueError if no module by that name exists, with the
    available adapter list in the message.
    """
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown adapter {name!r}. Available: {list_adapters()}"
        ) from exc
    return module.Adapter(model_name, config)


__all__ = ["BaseAdapter", "list_adapters", "get_adapter"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_adapters_registry.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Confirm the full suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: 376 passed (previous 374 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/adapters/__init__.py tests/test_adapters_registry.py
git commit -m "feat: adapter discovery via pkgutil + importlib

list_adapters() walks src/adapters/ and returns names excluding
_* and 'base'. get_adapter(name, ...) imports src.adapters.<name>
and instantiates its Adapter class; raises ValueError with the
available list when the name is unknown. The filesystem is the
registry — no central dict to keep in sync."
```

---

## Task 3: `connection.py` routing branch

**Files:**
- Modify: `src/engine/connection.py`
- Test: `tests/test_adapters_connection.py`

**Interfaces:**
- Consumes: `get_adapter` from Task 2.
- Produces: behavior — when `model_config["adapter"]` is set, `get_interface_for_provider(...)` returns the adapter instance instead of falling through to the existing provider routing.

- [ ] **Step 1: Find the right insertion point**

Read `src/engine/connection.py` and locate `def get_interface_for_provider(`. Note the parameter names (`connection_info`, `model_name`) — the test must call the function with the same shape.

- [ ] **Step 2: Write the failing test**

Write `tests/test_adapters_connection.py`:

```python
"""Tests for the adapter routing branch in connection.py."""
from unittest.mock import patch

from src.adapters.base import BaseAdapter
from src.engine.connection import get_interface_for_provider


class _StubAdapter(BaseAdapter):
    async def generate_batch(self, requests):
        return [{"output": "stub", "raw": "stub", "error": None, "error_type": None}]


def test_adapter_field_routes_through_get_adapter():
    """When connection_info has an 'adapter' key, get_interface_for_provider
    returns get_adapter(...)'s result and skips the legacy provider path."""
    connection_info = {
        "adapter": "voxtral_chat",
        "voxtral_chat": {"trust_remote_code": True, "torch_dtype": "float16"},
        # Existing fields are present but should be ignored when adapter: is set.
        "provider_type": "transformers_audio",
    }
    expected_instance = _StubAdapter("voxtral-x", {"trust_remote_code": True, "torch_dtype": "float16"})
    with patch("src.engine.connection.get_adapter", return_value=expected_instance) as gated:
        result = get_interface_for_provider(connection_info, "voxtral-x")
    assert result is expected_instance
    gated.assert_called_once_with(
        "voxtral_chat",
        "voxtral-x",
        {"trust_remote_code": True, "torch_dtype": "float16"},
    )


def test_no_adapter_field_falls_through_to_legacy_routing():
    """When 'adapter' is absent, get_interface_for_provider does NOT touch
    get_adapter — the legacy provider routing handles it."""
    connection_info = {
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    }
    with patch("src.engine.connection.get_adapter") as gated:
        try:
            get_interface_for_provider(connection_info, "gpt-4o-mini")
        except Exception:
            # The legacy path may raise on missing fields in a unit test;
            # that's fine — we only care that get_adapter wasn't touched.
            pass
    gated.assert_not_called()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters_connection.py -v`
Expected: FAIL — likely `ImportError: cannot import name 'get_adapter' from 'src.engine.connection'` because the patch target doesn't exist yet.

- [ ] **Step 4: Add the import + branch to `connection.py`**

At the top of `src/engine/connection.py`, alongside the other imports, add:

```python
from src.adapters import get_adapter
```

Inside `get_interface_for_provider`, at the very top of the function body (before any existing logic), add:

```python
    # Adapter routing — Task layer's view stays uniform because BaseAdapter
    # and BaseInterface expose the same prepare_request + generate_batch
    # contract. See docs/superpowers/specs/2026-06-19-adapter-layer-design.md.
    adapter_name = connection_info.get("adapter")
    if adapter_name:
        adapter_config = connection_info.get(adapter_name, {})
        return get_adapter(adapter_name, model_name, adapter_config)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_adapters_connection.py -v`
Expected: 2/2 PASS.

- [ ] **Step 6: Confirm the full suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: 378 passed.

- [ ] **Step 7: Commit**

```bash
git add src/engine/connection.py tests/test_adapters_connection.py
git commit -m "feat: route models with adapter: field through get_adapter

connection.get_interface_for_provider short-circuits to the adapter
path when connection_info['adapter'] is set, reading the adapter's
config block from connection_info[adapter_name]. Existing
provider-based routing is untouched when adapter is absent."
```

---

## Task 4: `voxtral_chat` adapter

**Files:**
- Create: `src/adapters/voxtral_chat.py`
- Test: `tests/test_adapters_voxtral_chat.py`

**Interfaces:**
- Consumes: `BaseAdapter` (Task 1), `InterfaceCapabilities`.
- Produces: `Adapter` class in `src.adapters.voxtral_chat` loading `mistralai/Voxtral-Mini-4B-Realtime-2602` (or any HF repo via `model_name`) through `AutoModelForCausalLM` + `AutoProcessor` with `trust_remote_code=True`, building a chat-template input with audio + text, decoding the generated tokens as the transcription.

- [ ] **Step 1: Write the failing test**

Write `tests/test_adapters_voxtral_chat.py`:

```python
"""Tests for the voxtral_chat adapter — mocked, no real model load."""
import asyncio
from unittest.mock import MagicMock, patch

import torch


def _fake_model():
    m = MagicMock()
    # Returns input_ids + 3 new tokens, total 6 tokens.
    m.generate.return_value = torch.tensor([[10, 20, 30, 40, 50, 60]])
    return m


def _fake_processor():
    p = MagicMock()
    p.apply_chat_template.return_value = {
        # input length 3 — generate() returned 6 — so 3 new tokens decoded.
        "input_ids": torch.tensor([[10, 20, 30]]),
    }
    p.batch_decode.return_value = ["hola mundo"]
    return p


def test_voxtral_chat_returns_standard_result_shape():
    """One sample in → one {output, raw, error, error_type} out."""
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=_fake_model(),
    ), patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=_fake_processor(),
    ):
        from src.adapters.voxtral_chat import Adapter

        adapter = Adapter(
            "mistralai/Voxtral-Mini-4B-Realtime-2602",
            {"torch_dtype": "float16", "trust_remote_code": True},
        )
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/f.wav", "language": "es"}]
            )
        )

    assert len(result) == 1
    assert result[0] == {
        "output": "hola mundo",
        "raw": "hola mundo",
        "error": None,
        "error_type": None,
    }


def test_voxtral_chat_captures_per_request_errors():
    """A generate() that raises ends up in error/error_type, not propagated."""
    bad_model = MagicMock()
    bad_model.generate.side_effect = RuntimeError("CUDA OOM")

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=bad_model,
    ), patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=_fake_processor(),
    ):
        from src.adapters.voxtral_chat import Adapter

        adapter = Adapter("mistralai/Voxtral-Mini-4B-Realtime-2602", {})
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/f.wav"}]
            )
        )

    assert result[0]["output"] == ""
    assert result[0]["error_type"] == "RuntimeError"
    assert "CUDA OOM" in result[0]["error"]


def test_voxtral_chat_loads_model_lazily():
    """from_pretrained is not called until the first generate_batch."""
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=_fake_model(),
    ) as model_load, patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=_fake_processor(),
    ) as proc_load:
        from src.adapters.voxtral_chat import Adapter

        adapter = Adapter("mistralai/Voxtral-Mini-4B-Realtime-2602", {})
        model_load.assert_not_called()
        proc_load.assert_not_called()

        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

        model_load.assert_called_once()
        proc_load.assert_called_once()


def test_voxtral_chat_capabilities_declares_audio_input():
    """Adapter capabilities mark audio_input + text_output as supported."""
    from src.adapters.voxtral_chat import Adapter

    caps = Adapter.capabilities
    assert caps.supports_audio_input is True
    assert caps.supports_text_output is True
    assert caps.supports_multimodal is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters_voxtral_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.adapters.voxtral_chat'`.

- [ ] **Step 3: Write the adapter**

Write `src/adapters/voxtral_chat.py`:

```python
"""voxtral_chat adapter — Mistral Voxtral family ASR.

Voxtral models are multimodal LLMs that accept audio via a chat
template and emit text. They cannot be loaded through
`transformers.pipeline("automatic-speech-recognition")` because
their model_type ('voxtral_realtime') is not in the pipeline's
auto-dispatch table even with trust_remote_code=True. This adapter
calls AutoModelForCausalLM + AutoProcessor directly.

Config knobs (read from `voxtral_chat:` block in the model YAML):
  trust_remote_code: bool (default True)
  torch_dtype: "float16" | "float32" | "bfloat16" (default "float16")
  device: "auto" | "cpu" | "cuda" | "mps" (default "auto")
  chat_prompt: str (default "Transcribe this audio.")
  max_new_tokens: int (default 256)
"""

from __future__ import annotations

from typing import Any

from src.adapters.base import BaseAdapter
from src.engine.protocols import InterfaceCapabilities


_DTYPE_MAP = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Adapter(BaseAdapter):
    capabilities = InterfaceCapabilities(
        supports_audio_input=True,
        supports_text_output=True,
    )

    def __init__(self, model_name: str, config: dict[str, Any]):
        super().__init__(model_name, config)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype_name = _DTYPE_MAP[self.config.get("torch_dtype", "float16")]
        torch_dtype = getattr(torch, dtype_name)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        device = _resolve_device(self.config.get("device", "auto"))

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self._device = device

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._model is None:
            self._load()

        prompt = self.config.get("chat_prompt", "Transcribe this audio.")
        max_new_tokens = int(self.config.get("max_new_tokens", 256))

        results = []
        for req in requests:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": req["audio_path"]},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(self, "_device") and self._device != "cpu":
                    inputs = {
                        k: v.to(self._device) if hasattr(v, "to") else v
                        for k, v in inputs.items()
                    }
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                input_len = inputs["input_ids"].shape[-1]
                generated = output_ids[:, input_len:]
                text = self._processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0]
                results.append(
                    {"output": text, "raw": text, "error": None, "error_type": None}
                )
            except Exception as exc:
                results.append(
                    {
                        "output": "",
                        "raw": "",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_adapters_voxtral_chat.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Confirm the full suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: 382 passed.

- [ ] **Step 6: Commit**

```bash
git add src/adapters/voxtral_chat.py tests/test_adapters_voxtral_chat.py
git commit -m "feat: add voxtral_chat adapter

Loads Voxtral-family ASR models via AutoModelForCausalLM +
AutoProcessor with trust_remote_code=True. Builds chat-template
input with audio + text prompt, decodes generated tokens past the
input length as the transcription. Mocked tests cover the standard
result shape, per-request error capture, lazy loading, and
capability declaration."
```

---

## Task 5: `qwen3_asr_chat` adapter

**Files:**
- Create: `src/adapters/qwen3_asr_chat.py`
- Test: `tests/test_adapters_qwen3_asr_chat.py`

**Interfaces:**
- Consumes: `BaseAdapter`, `InterfaceCapabilities`.
- Produces: `Adapter` class in `src.adapters.qwen3_asr_chat` — same shape and behavior as `voxtral_chat.Adapter` but parameterized for the Qwen3-ASR family. Structurally identical implementation; consolidation into a shared base is a future optimization (see best-part-is-no-part lens 5) — defer until a third such adapter shows up.

- [ ] **Step 1: Write the failing test**

Write `tests/test_adapters_qwen3_asr_chat.py`:

```python
"""Tests for the qwen3_asr_chat adapter — mocked, no real model load."""
import asyncio
from unittest.mock import MagicMock, patch

import torch


def _fake_model():
    m = MagicMock()
    m.generate.return_value = torch.tensor([[10, 20, 30, 40, 50, 60]])
    return m


def _fake_processor():
    p = MagicMock()
    p.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[10, 20, 30]]),
    }
    p.batch_decode.return_value = ["olá mundo"]
    return p


def test_qwen3_asr_chat_returns_standard_result_shape():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=_fake_model(),
    ), patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=_fake_processor(),
    ):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter(
            "Qwen/Qwen3-ASR-0.6B",
            {"torch_dtype": "float16", "trust_remote_code": True},
        )
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/f.wav", "language": "pt"}]
            )
        )

    assert result[0] == {
        "output": "olá mundo",
        "raw": "olá mundo",
        "error": None,
        "error_type": None,
    }


def test_qwen3_asr_chat_captures_per_request_errors():
    bad_model = MagicMock()
    bad_model.generate.side_effect = ValueError("audio file not found")

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=bad_model,
    ), patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=_fake_processor(),
    ):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {})
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/missing.wav"}]
            )
        )

    assert result[0]["output"] == ""
    assert result[0]["error_type"] == "ValueError"


def test_qwen3_asr_chat_loads_model_lazily():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=_fake_model(),
    ) as model_load, patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=_fake_processor(),
    ) as proc_load:
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {})
        model_load.assert_not_called()
        proc_load.assert_not_called()

        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

        model_load.assert_called_once()
        proc_load.assert_called_once()


def test_qwen3_asr_chat_capabilities_declares_audio_input():
    from src.adapters.qwen3_asr_chat import Adapter

    caps = Adapter.capabilities
    assert caps.supports_audio_input is True
    assert caps.supports_text_output is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters_qwen3_asr_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.adapters.qwen3_asr_chat'`.

- [ ] **Step 3: Write the adapter**

Write `src/adapters/qwen3_asr_chat.py` — structurally identical to `voxtral_chat.py` with the only differences being the docstring, the default `chat_prompt` (Qwen tends to prefer a slightly more imperative prompt), and the class is its own type (not a subclass of voxtral). Body:

```python
"""qwen3_asr_chat adapter — Qwen3-ASR family ASR.

Qwen3-ASR is an LLM-style ASR that accepts audio via the Qwen
processor's chat template and emits text. Its model_type
('qwen3_asr') is not in transformers' pipeline auto-dispatch even
with trust_remote_code=True, so this adapter calls
AutoModelForCausalLM + AutoProcessor directly.

Config knobs (read from `qwen3_asr_chat:` block in the model YAML):
  trust_remote_code: bool (default True)
  torch_dtype: "float16" | "float32" | "bfloat16" (default "float16")
  device: "auto" | "cpu" | "cuda" | "mps" (default "auto")
  chat_prompt: str (default "Please transcribe the audio.")
  max_new_tokens: int (default 256)
"""

from __future__ import annotations

from typing import Any

from src.adapters.base import BaseAdapter
from src.engine.protocols import InterfaceCapabilities


_DTYPE_MAP = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Adapter(BaseAdapter):
    capabilities = InterfaceCapabilities(
        supports_audio_input=True,
        supports_text_output=True,
    )

    def __init__(self, model_name: str, config: dict[str, Any]):
        super().__init__(model_name, config)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype_name = _DTYPE_MAP[self.config.get("torch_dtype", "float16")]
        torch_dtype = getattr(torch, dtype_name)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        device = _resolve_device(self.config.get("device", "auto"))

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self._device = device

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._model is None:
            self._load()

        prompt = self.config.get("chat_prompt", "Please transcribe the audio.")
        max_new_tokens = int(self.config.get("max_new_tokens", 256))

        results = []
        for req in requests:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": req["audio_path"]},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(self, "_device") and self._device != "cpu":
                    inputs = {
                        k: v.to(self._device) if hasattr(v, "to") else v
                        for k, v in inputs.items()
                    }
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                input_len = inputs["input_ids"].shape[-1]
                generated = output_ids[:, input_len:]
                text = self._processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0]
                results.append(
                    {"output": text, "raw": text, "error": None, "error_type": None}
                )
            except Exception as exc:
                results.append(
                    {
                        "output": "",
                        "raw": "",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_adapters_qwen3_asr_chat.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Confirm the full suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: 386 passed.

- [ ] **Step 6: Commit**

```bash
git add src/adapters/qwen3_asr_chat.py tests/test_adapters_qwen3_asr_chat.py
git commit -m "feat: add qwen3_asr_chat adapter

Same shape as voxtral_chat: AutoModelForCausalLM + AutoProcessor
with trust_remote_code, chat-template input, generated-token
decode. Defaults match Qwen3-ASR's documented prompt style.
Consolidating with voxtral_chat into a shared base deferred per
best-part-is-no-part lens 5 — wait for a third instance."
```

---

## Task 6: Migrate the two model YAMLs to use `adapter:`

**Files:**
- Modify: `configs/models/voxtral-mini-4b-transformers.yaml`
- Modify: `configs/models/qwen3-asr-0.6b-transformers.yaml`

**Interfaces:**
- Consumes: the `adapter:` routing in `connection.py` (Task 3) and both adapters (Tasks 4, 5).
- Produces: model configs that resolve through the adapter path. Loading the YAML and instantiating the inference object should return an `Adapter` instance, not a `TransformersAudioInterface`.

- [ ] **Step 1: Write the verification test**

Add a single test that exercises end-to-end YAML → connection → adapter wiring without downloading a model. Append to `tests/test_adapters_connection.py`:

```python
def test_voxtral_yaml_resolves_through_adapter_path(monkeypatch):
    """Loading the voxtral YAML wires connection_info to the voxtral_chat adapter."""
    import yaml
    from pathlib import Path

    cfg_path = Path("configs/models/voxtral-mini-4b-transformers.yaml")
    model_config = yaml.safe_load(cfg_path.read_text())

    assert model_config.get("adapter") == "voxtral_chat", (
        "Task 6 should have set adapter: voxtral_chat in the YAML"
    )

    from src.adapters.voxtral_chat import Adapter as VoxtralAdapter
    from src.engine.connection import get_interface_for_provider

    # Build a connection_info dict the way ConfigManager would.
    connection_info = dict(model_config)
    result = get_interface_for_provider(connection_info, model_config["model"]["name"])
    assert isinstance(result, VoxtralAdapter)


def test_qwen3_asr_yaml_resolves_through_adapter_path():
    """Same shape for Qwen3-ASR."""
    import yaml
    from pathlib import Path

    cfg_path = Path("configs/models/qwen3-asr-0.6b-transformers.yaml")
    model_config = yaml.safe_load(cfg_path.read_text())

    assert model_config.get("adapter") == "qwen3_asr_chat"

    from src.adapters.qwen3_asr_chat import Adapter as Qwen3Adapter
    from src.engine.connection import get_interface_for_provider

    connection_info = dict(model_config)
    result = get_interface_for_provider(connection_info, model_config["model"]["name"])
    assert isinstance(result, Qwen3Adapter)
```

Run: `.venv/bin/python -m pytest tests/test_adapters_connection.py::test_voxtral_yaml_resolves_through_adapter_path -v`
Expected: FAIL — the YAML still has the old `transformers_audio:` shape; assertion on `adapter` key fails.

- [ ] **Step 2: Rewrite `configs/models/voxtral-mini-4b-transformers.yaml`**

Replace the entire file with:

```yaml
model:
  name: mistralai/Voxtral-Mini-4B-Realtime-2602
adapter: voxtral_chat
voxtral_chat:
  trust_remote_code: true
  torch_dtype: float16
  chat_prompt: "Transcribe this audio. Output only the transcription, no commentary."
  max_new_tokens: 256
  device: auto
task_defaults:
  log_samples: true
tasks:
- transcription
metadata:
  provider: huggingface
  model_type: voxtral
  is_cloud: false
  description: "mistralai/Voxtral-Mini-4B-Realtime-2602 via voxtral_chat adapter (~4B params; en/fr/es/de/ru/zh/ja/it). fp16 required on <=16GB Macs."
```

- [ ] **Step 3: Rewrite `configs/models/qwen3-asr-0.6b-transformers.yaml`**

Replace the entire file with:

```yaml
model:
  name: Qwen/Qwen3-ASR-0.6B
adapter: qwen3_asr_chat
qwen3_asr_chat:
  trust_remote_code: true
  torch_dtype: float16
  chat_prompt: "Please transcribe the audio."
  max_new_tokens: 256
  device: auto
task_defaults:
  log_samples: true
tasks:
- transcription
metadata:
  provider: huggingface
  model_type: qwen3_asr
  is_cloud: false
  description: "Qwen/Qwen3-ASR-0.6B via qwen3_asr_chat adapter (~600M params, multilingual)."
```

- [ ] **Step 4: Run the verification tests**

Run: `.venv/bin/python -m pytest tests/test_adapters_connection.py -v`
Expected: all 4 PASS (2 from Task 3, 2 added in this task).

- [ ] **Step 5: Confirm the full suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: 388 passed.

- [ ] **Step 6: Verify existing Whisper paths still work**

The non-adapter model YAMLs must keep routing through the legacy
provider path. Run an existing-path smoke to prove it:

```bash
set -a; source .env; set +a
.venv/bin/benchy eval \
  -c whisper-tiny-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 1 --run-id adapter_layer_legacy_smoke \
  --exit-policy smoke 2>&1 | tail -5
echo "exit=$?"
```

Expected: `exit=0` and `Run status: passed`. If this fails, the
`connection.py` branch has wrongly affected the legacy path — fix
before continuing.

- [ ] **Step 7: Commit**

```bash
git add configs/models/voxtral-mini-4b-transformers.yaml \
        configs/models/qwen3-asr-0.6b-transformers.yaml \
        tests/test_adapters_connection.py
git commit -m "chore: migrate voxtral + qwen3-asr YAMLs to adapter: path

Drop the transformers_audio: block and provider routing for these
two custom-code models. The adapter: field plus the matching
adapter-name block carry every knob the old shape did, and the new
verification tests confirm the YAMLs resolve to the right Adapter
class. Whisper-tiny legacy smoke verifies the existing provider
path is unaffected."
```

---

## Task 7: End-to-end smoke against real models + skill update

**Files:**
- Possibly modify: `configs/models/voxtral-mini-4b-transformers.yaml` or `configs/models/qwen3-asr-0.6b-transformers.yaml` if a config knob needs tuning based on smoke findings
- Modify: `.agent/skills/whisper-benchmark/SKILL.md` and `docs/how-to-transcription-benchmark.md` — add a brief note that Voxtral / Qwen3-ASR / Canary use the adapter path (with current support status)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: documented evidence that the adapter layer works end-to-end against real Voxtral and Qwen3-ASR weights, OR a BLOCKED note with the specific failure mode (per the spec — a blocked adapter does not block the layer).

- [ ] **Step 1: Disk + memory pre-flight**

```bash
df -h ~ | tail -1
sysctl -n hw.memsize | awk '{printf "%.0f GB total RAM\n", $1/1024/1024/1024}'
du -sh ~/.cache/huggingface/hub 2>/dev/null | head -1
```

Voxtral weights are ~8 GB on disk. If free space is below ~15 GB, stop and surface to the user; do not auto-clean cache.

- [ ] **Step 2: Smoke voxtral_chat end-to-end (2 samples)**

```bash
set -a; source .env; set +a
timeout 1200 .venv/bin/benchy eval \
  -c voxtral-mini-4b-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 2 --log-samples \
  --run-id adapter_smoke_voxtral \
  --exit-policy smoke 2>&1 | tail -25
echo "exit=$?"
```

Three possible outcomes:

  **PASS** — `exit=0`, `Run status: passed`, predictions look like Spanish.
  Verify: parse `outputs/benchmark_outputs/adapter_smoke_voxtral_LIMITED/<model>/transcription/fleurs_es_latam/*_metrics.json` for `metrics.wer < 1.0`. Record the WER in the skill update at Step 4.

  **PARTIAL** — `exit=0` but `metrics.wer == 1.0` and `error_rate > 0`.
  Read the log under `logs/<run>/` to find the per-sample error. Common cause: the model card uses a different chat-template key or processor method name than the adapter assumed. Fix in `src/adapters/voxtral_chat.py`, re-test. If still partial after a single targeted fix, record as BLOCKED and continue to Step 3.

  **BLOCKED** — load fails outright. Record the specific error in Step 4 and move on. The spec explicitly allows this.

- [ ] **Step 3: Smoke qwen3_asr_chat end-to-end (2 samples)**

```bash
timeout 600 .venv/bin/benchy eval \
  -c qwen3-asr-0.6b-transformers \
  --tasks transcription.fleurs_es_latam \
  --limit 2 --log-samples \
  --run-id adapter_smoke_qwen3_asr \
  --exit-policy smoke 2>&1 | tail -25
echo "exit=$?"
```

Apply the same three-outcome triage as Step 2.

- [ ] **Step 4: Update the transcription skill + doc with current status**

Open `.agent/skills/whisper-benchmark/SKILL.md` and find the
"Extending" section that mentions Voxtral/Qwen3-ASR/Canary as needing
"per-family interface" code. Replace that paragraph with a current
status block, e.g.:

```markdown
**Non-Whisper architectures (Voxtral, Qwen3-ASR, Canary)** now have
dedicated adapters under `src/adapters/`. As of 2026-06-19:

- `voxtral_chat`: [PASS / BLOCKED — write outcome here, with WER if PASS or error class if BLOCKED]
- `qwen3_asr_chat`: [PASS / BLOCKED — same]
- `canary_nemo`: not yet implemented; needs `nemo-toolkit[asr]` install path

Use the adapter path by setting `adapter: <name>` in the model YAML.
See `docs/superpowers/specs/2026-06-19-adapter-layer-design.md` for
the layer design.
```

Make the equivalent edit in `docs/how-to-transcription-benchmark.md`.

- [ ] **Step 5: Confirm the full unit-test suite is still green**

Run: `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`
Expected: 388 passed (Task 6 totals).

- [ ] **Step 6: Commit**

```bash
git add .agent/skills/whisper-benchmark/SKILL.md \
        docs/how-to-transcription-benchmark.md \
        src/adapters/  # in case Step 2 required a targeted fix
git commit -m "docs: document adapter-layer ASR support status

Records 2026-06-19 end-to-end smoke outcomes for voxtral_chat and
qwen3_asr_chat. Updates the whisper-benchmark skill and the
how-to-transcription-benchmark doc to point at src/adapters/ for
non-Whisper architectures and to flag the current PASS/BLOCKED
state per model."
```

---

## Post-implementation

After Task 7:

1. **Update the multi-architecture ASR benchmark plan** — open
   `docs/superpowers/plans/2026-06-19-multi-architecture-asr-benchmark.md`
   and update the Phase 1 status section: replace the BLOCKED rows
   for Voxtral and Qwen3-ASR with the new outcome. Canary remains
   BLOCKED behind the NeMo decision.
2. **Decide whether to run Phase 2 (panel)** now that adapters work.
   That's a fresh user decision — surface the question rather than
   auto-running a multi-hour panel.
3. **Future spec — adapter consolidation.** `voxtral_chat` and
   `qwen3_asr_chat` are structurally identical (lens 5). When a
   third LLM-style ASR adapter shows up, do not add a third
   duplicate file — extract a `BaseLLMChatASRAdapter` and have
   `voxtral_chat`, `qwen3_asr_chat`, and the third one each be a
   ~10-line subclass declaring model-specific prompt/dtype defaults.
   Not now; record the trigger condition.
4. **Future spec — adapter docs**. The skill index in
   `.agent/skills/README.md` should grow a row for `add-adapter`
   once anyone other than the original implementer adds one.

## Risks and rollback

- **An adapter's chat-template assumption is wrong.** Each Task 4/5/7
  step has a fallback ("record as BLOCKED, move on"). One bad
  adapter does not block the layer per the spec.
- **`connection.py` branch accidentally captures legacy YAMLs.** Task
  6 Step 6 is the gate that catches this — a Whisper smoke through
  the legacy path. Revert Task 3 if it fails.
- **Test mocks drift from real `transformers` API.** The mocks
  exercise the public functions only (`from_pretrained`,
  `apply_chat_template`, `generate`, `batch_decode`). If a real
  smoke (Tasks 7) reveals a different call shape, fix the adapter
  *and* the test in the same commit so they evolve together.
- **Rollback.** Each task is one commit. `git revert <sha>` undoes
  one task. Tasks 1–5 are pure additions; reverting them removes
  the adapter layer entirely without touching legacy code. Task 6
  is the only one that changes existing files (two YAMLs); reverting
  it restores their original shape.
