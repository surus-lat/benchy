# Adapter Layer — Design Spec

**Status:** Design approved 2026-06-19, ready for implementation plan.

**Scope:** Introduce a per-model-family **Adapter** abstraction that
absorbs architecture variance, transport variance, and dispatch quirks
behind a single contract. Ship two adapters that unblock the
multi-architecture ASR benchmark (Voxtral, Qwen3-ASR). Leave every
existing model YAML and code path working unchanged.

This spec is the tactical first move toward a mid-term redesign where
Tasks, Benchmarks, AI Systems, and Adapters are first-class peers. The
mid-term redesign is **not** in this spec's scope.

## Problem

Benchy's current `Interface` layer is provider-shaped:
`OpenAIInterface` covers everything OpenAI-API-compatible,
`TransformersAudioInterface` covers everything that loads via
`transformers.pipeline("automatic-speech-recognition")`. This works for
models that fit the dispatch shape, and fails immediately for models
that don't.

Phase 1 of the multi-architecture ASR benchmark
(`docs/superpowers/plans/2026-06-19-multi-architecture-asr-benchmark.md`)
confirmed this empirically. Three custom-code models all blocked with
the same root cause:

| Model | Failure |
|---|---|
| `nvidia/canary-1b-flash` | `model type 'fastconformer' not recognized` — NeMo-specific arch |
| `Qwen/Qwen3-ASR-0.6B` | `model type 'qwen3_asr' not recognized` — LLM-style chat-template ASR |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | `model type 'voxtral_realtime' not recognized` — LLM-style chat-template ASR |

`pipeline()` calls `AutoConfig.from_pretrained` without passing
`trust_remote_code=True`, and even with the flag the models need
chat-template input building, not `pipeline()`-style auto-dispatch.

The fix shape is per-family code: load the model with the right
`AutoModel*` class, build the right input via `AutoProcessor`, decode
the right output. Today there's nowhere to put that code that doesn't
contaminate `TransformersAudioInterface`. The Adapter layer is that
home.

## Goals

1. A model YAML can name an adapter that owns "how to ask this family
   for inference." Models that need bespoke loaders or input shapes
   work without changes to the existing Interface code.
2. Adding a new adapter is one new file under `src/adapters/` and one
   YAML edit. No central registry to keep in sync.
3. Every existing model YAML keeps working with zero edits. The
   adapter layer coexists with the Interface layer until a future spec
   migrates the rest.
4. The Task layer doesn't change. Adapters expose the same result
   shape as Interfaces (`{output, raw, error, error_type}`), so
   `BenchmarkRunner`, handlers, metrics, retry, and observability work
   unchanged.

## Non-goals

- Migrating existing model YAMLs to declare adapters. Mechanical work
  for a future spec; not needed for this one.
- Splitting `Task` from `Benchmark` (the dataset+metric pair). Future
  spec.
- Treating "AI Systems" as a first-class collection. Future spec.
- Supporting Canary in v1. Needs `nemo-toolkit[asr]` (~2 GB, conflicts
  with pinned torch). Separate decision.
- New CLI flags, new public JSON schemas, new dataset spec. Public
  surface stays frozen.

## Design

### The principle behind every choice

**Best part is no part — but not at the cost of capability.** Each
"part" in this spec exists only when no Python language feature,
existing project pattern, or filesystem affordance can carry its
weight. When in doubt, the section explicitly names what the part
*isn't* doing and why a lighter alternative would have failed.

### Concept

An **Adapter** owns three things for one model family:

- **Architecture** — what `AutoModel*` to call, what processor to
  load, what input shape it expects.
- **Transport** — in-process via `transformers`, HTTP via SDK, or vendor
  client. The Adapter hides this from the Task.
- **Dispatch quirks** — `trust_remote_code` toggles, chat templates,
  `generate_kwargs.language`, custom decoder calls.

Two adapters for the same family should not exist. If two model
families differ only in HF repo id, they share an Adapter (parameterize
on `model_name`).

### Protocol

`BaseAdapter` exposes the same surface BenchmarkRunner already uses on
Interfaces — two methods and one class attribute — so the runner
treats Adapters and Interfaces interchangeably without a single line
of change in `BenchmarkRunner`. The Adapter contract is *symmetric to*
the Interface contract, not a new one.

```python
# src/adapters/base.py — ~30 lines

from typing import Any
from src.engine.protocols import InterfaceCapabilities

class BaseAdapter:
    """Abstract per-model-family adapter.

    Surface intentionally mirrors `BaseInterface` so BenchmarkRunner
    sees identical shape from either flavor. Subclasses set
    `capabilities` (a class attribute) and implement `generate_batch`;
    `prepare_request` has an identity default so adapters that don't
    need it can skip it entirely.
    """

    # Subclass overrides with a populated InterfaceCapabilities. Reuses
    # the existing dataclass — no new capability type.
    capabilities: InterfaceCapabilities

    def __init__(self, model_name: str, config: dict[str, Any]):
        self.model_name = model_name
        self.config = config

    def prepare_request(self, sample: dict[str, Any], task: Any) -> dict[str, Any]:
        """Turn a task-level sample into a request.

        Default: identity (return the sample untouched). Override only
        if the adapter wants to peek, transform, or attach metadata
        before `generate_batch` runs. BenchmarkRunner calls this once
        per sample before batching.
        """
        return sample

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run inference on a batch.

        Returns one result dict per request: `{output, raw, error,
        error_type}`. Subclasses do lazy model loading on first call.
        Resource cleanup happens via Python GC when the Adapter object
        is dropped — no `unload()` method needed.

        Partial-batch failures: failed items have non-empty `error`
        and `error_type`; successful items have non-empty `output`.
        The existing BenchmarkRunner retry path consumes this shape
        unchanged.
        """
        raise NotImplementedError
```

That's the entire contract. `prepare_request` is there because
BenchmarkRunner uses it; the identity default means adapters never
have to write a no-op. Everything else is per-adapter implementation
detail.

#### Why these are the only required parts

| Capability | How we deliver it | What we don't add |
|---|---|---|
| Lazy init | `if self._model is None: …` inside `generate_batch` | No `load()` protocol method |
| Resource cleanup | Python GC drops `self._model` when the Adapter goes out of scope; `del adapter` at the call site forces it now | No `unload()` protocol method |
| Discoverability | `pkgutil.iter_modules(src.adapters.__path__)` lists files on disk | No central `REGISTRY` dict |
| Unified Task access | `BaseAdapter.generate_batch` and `BaseInterface.generate_batch` have the same shape — Task code is duck-typed against the shape | No wrapper adapters around existing interfaces |
| Request introspection | `prepare_request` defaults to identity; adapters that want to inspect/transform override it. Result `raw` field carries post-hoc evidence for everything else | No new protocol method — uses the same `prepare_request` BenchmarkRunner already calls |
| Retry | BenchmarkRunner retries `generate_batch` whole; adapters return partial-success batches | No new retry primitives |
| Capabilities declaration | Reuse `InterfaceCapabilities` as a class attribute on the Adapter | No new `AdapterCapabilities` dataclass |

Each row is one capability kept, with the part it doesn't require.

### Discovery and instantiation

`src/adapters/__init__.py` is twelve lines.

```python
import importlib
import pkgutil
from .base import BaseAdapter

def list_adapters() -> list[str]:
    return sorted(
        m.name for m in pkgutil.iter_modules(__path__)
        if not m.name.startswith("_") and m.name != "base"
    )

def get_adapter(name: str, model_name: str, config: dict) -> BaseAdapter:
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown adapter {name!r}. Available: {list_adapters()}"
        ) from exc
    return module.Adapter(model_name, config)
```

Convention: every adapter module exports a class literally named
`Adapter`. No `__all__`, no entry point, no registration call. The
filesystem is the registry. A typo'd adapter name in a YAML errors with
a list of valid adapter names — same UX as a registry, no second source
of truth.

### Model YAML shape

The YAML names the adapter as a string and carries adapter-specific
knobs in a block named after the adapter — same pattern as the
existing `provider: transformers_audio` + `transformers_audio: {…}`.

```yaml
# configs/models/voxtral-mini-4b.yaml
model:
  name: mistralai/Voxtral-Mini-4B-Realtime-2602
adapter: voxtral_chat
voxtral_chat:
  trust_remote_code: true
  torch_dtype: float16
  chat_prompt: "Transcribe this audio. Output only the transcription, no commentary."
tasks:
  - transcription
metadata:
  model_type: voxtral
  is_cloud: false
```

If `adapter:` is **absent**, `connection.py` falls through to the
existing `get_interface_for_provider` routing — every existing YAML
keeps working untouched.

### Connection routing

`src/engine/connection.py` grows one branch:

```python
# inside get_interface_for_provider, top of the function:
if "adapter" in model_config:
    adapter_name = model_config["adapter"]
    adapter_config = model_config.get(adapter_name, {})
    return get_adapter(adapter_name, model_name, adapter_config)
# … existing routing below
```

The function name lies slightly post-change (it returns either an
Interface or an Adapter), but they're duck-typed identical at the
callsite. Renaming `get_interface_for_provider` to something like
`get_inference_provider` is a future-spec concern. Not in scope here.

### v1 built-in adapters

Two adapters ship in this spec — the minimum to deliver the
multi-architecture ASR benchmark.

#### `voxtral_chat`

`src/adapters/voxtral_chat.py` — `Adapter` class loading
`VoxtralForConditionalGeneration` (or the canonical class per the model
card) via `AutoModelForCausalLM` with `trust_remote_code=True`, using
`AutoProcessor.apply_chat_template` to build the input message
sequence (audio + text prompt), decoding the generated text as the
transcription.

Config knobs accepted under the `voxtral_chat:` YAML block:
`trust_remote_code` (bool, default `true`), `torch_dtype` (str,
default `"float16"`), `device` (str, default `"auto"`), `chat_prompt`
(str, default `"Transcribe this audio."`), `max_new_tokens` (int,
default `256`).

#### `qwen3_asr_chat`

`src/adapters/qwen3_asr_chat.py` — `Adapter` class loading
Qwen3-ASR via `AutoModelForCausalLM` + `AutoProcessor` per the model
card's instructions, building the chat input with audio embedding + text
prompt, decoding.

Same config knob shape as `voxtral_chat` plus any Qwen-specific
processor options surfaced by the model card. Hold off on inventing
options until the smoke run shows they're needed.

### Capabilities declarations

Both adapters declare:

```python
capabilities = InterfaceCapabilities(
    supports_audio_input=True,
    supports_text_output=True,
    supports_streaming=False,
    supports_multimodal=False,   # audio only, not vision
    supports_schema=False,
    supports_logprobs=False,
)
```

The existing capability-compatibility check in
`task_capabilities_compatible` consumes this dataclass already; no
change needed there.

### Testing

| Test | What it covers |
|---|---|
| `tests/test_adapters_base.py` | `BaseAdapter` instantiation raises `NotImplementedError` on `generate_batch`; subclasses with `capabilities` set pass an instance check |
| `tests/test_adapters_registry.py` | `list_adapters()` returns the on-disk file list; `get_adapter("nope", …)` raises `ValueError` with the available list in the message; `get_adapter("voxtral_chat", …)` returns a `VoxtralChatAdapter` |
| `tests/test_adapters_voxtral_chat.py` | Construction with a mocked model+processor; one round-trip through `generate_batch` returning the standard result shape; an induced exception lands in the `error` field |
| `tests/test_adapters_qwen3_asr_chat.py` | Same shape as the Voxtral test |

Tests don't download weights. They mock `transformers.AutoModelForCausalLM` and `AutoProcessor` to return fakes that record calls and emit fixed outputs. Real-model smokes happen via `benchy eval` after the unit tests pass.

## Out of scope (explicit)

- Migrating existing `provider:`-based YAMLs to use `adapter:`. Future
  spec. Each conversion is mechanical (~3 line YAML edit per file).
- A `whisper_pipeline` wrapper adapter or `openai_chat` wrapper
  adapter. The existing provider path already handles these without
  edits; wrapping adds parts with no behavior change.
- A `canary_nemo` adapter. Blocked on a decision about whether
  `nemo-toolkit[asr]` lands as a benchy extras group. Separate spec.
- Renaming `get_interface_for_provider`. The function still works
  correctly under its current name even though it now returns either
  flavor.
- Async/sync uniformity. `BaseAdapter.generate_batch` is `async` to
  match the existing `BaseInterface.generate_batch` shape; that's the
  only reason. Adapters with synchronous implementations wrap the
  body in `asyncio.to_thread`.

## File map

| Path | Action | Lines |
|---|---|---|
| `src/adapters/__init__.py` | create | ~12 |
| `src/adapters/base.py` | create | ~25 |
| `src/adapters/voxtral_chat.py` | create | ~80 |
| `src/adapters/qwen3_asr_chat.py` | create | ~80 |
| `src/engine/connection.py` | modify | +6 |
| `configs/models/voxtral-mini-4b-transformers.yaml` | rewrite | ~12 |
| `configs/models/qwen3-asr-0.6b-transformers.yaml` | rewrite | ~12 |
| `tests/test_adapters_base.py` | create | ~25 |
| `tests/test_adapters_registry.py` | create | ~30 |
| `tests/test_adapters_voxtral_chat.py` | create | ~40 |
| `tests/test_adapters_qwen3_asr_chat.py` | create | ~40 |

Total: ~360 lines, 10 new files, 1 modified file.

The voxtral and qwen3-asr YAMLs lose the `-transformers` suffix
conceptually (they're no longer transformers-pipeline-routed) but the
filename keeps `-transformers` for git-history continuity. Renaming is
a future-spec cosmetic.

## Migration story (informational)

When a future spec migrates the rest of the Interface system, the
shape is:

1. Convert each existing model YAML to declare an `adapter:` field.
   `provider: openai` → `adapter: openai_chat`. Mechanical pass.
2. Once every YAML has an `adapter:`, delete the
   `get_interface_for_provider` fallback in `connection.py`.
3. The Interface concept stops being a top-level abstraction. It
   becomes "transport plumbing some adapters reuse" — same code,
   demoted status.

None of this is in this spec's scope. The current spec is the
*enabling* move; the future spec is the *consolidating* move. Each
ships independently.

## Risks

- **An adapter's model card is wrong and the v1 smoke fails.** Each
  adapter ships with its own smoke run in the implementation plan; a
  blocked adapter doesn't block the layer. Document and proceed.
- **`importlib`-based discovery hides typos until runtime.** Mitigated
  by the `Available: [...]` error message and `list_adapters()` as a
  cheap CLI probe.
- **Class-attribute capabilities can be mutated at runtime.** Same risk
  as today's `InterfaceCapabilities`-as-class-attr pattern; no
  regression.
- **`async def generate_batch` is heavier than synchronous for some
  adapters.** Wrap synchronous bodies in `asyncio.to_thread` — the
  retry/observability code expects the awaitable.

## Open implementation-time questions

- Whether to use `pkgutil.iter_modules` (stdlib) or `importlib.resources`
  (newer, more friendly). Default: `pkgutil.iter_modules`. They produce
  the same list here; the legacy one is one fewer import.
- Whether `voxtral_chat` and `qwen3_asr_chat` should share a small
  parent class for the chat-template + decode pattern. Default: no.
  Premature DRY; revisit if a third LLM-style ASR adapter shows up and
  the duplication is real.
- Whether the YAML block name should be the adapter name (matches
  current `provider:` block pattern) or a fixed key like
  `adapter_config:`. Default: adapter-name block (current
  recommendation). Reverse only if it confuses readers, which the
  existing provider blocks evidently don't.
