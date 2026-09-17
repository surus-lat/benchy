"""`hf:` — local inference via `transformers`, routed by architecture.

This is the "using the right framework to run different model
architectures" promise from the vision, made concrete: an HF repo id does
not by itself say which runtime understands it, so `hf:` inspects the
repo's config and picks one of four internal families:

    transformers_audio   any transformers ASR pipeline (Whisper, etc) — default
    voxtral_chat          Mistral Voxtral speech-seq2seq models
    qwen3_asr_chat        Qwen3-ASR (needs the `qwen-asr` PyPI package)
    canary_nemo           NVIDIA Canary / FastConformer (needs `nemo-toolkit`)

URL grammar: `hf:<repo_id>`, e.g. `hf:openai/whisper-large-v3-turbo`.
`family=<name>` selects a family explicitly, skipping detection — required
for `canary_nemo` in practice, since NeMo checkpoints have no standard HF
`model_type` `AutoConfig` can read (detection falls back to a `"canary"`
substring match in the repo id when the config lookup itself fails, but an
explicit `family=` is more honest). Every other family/model-loading
keyword (`device=`, `torch_dtype=`, `trust_remote_code=`, ...) is that
family's own; see `whisper.py`/`voxtral.py`/`qwen3_asr.py`/`canary.py`.

**Loading is lazy end-to-end.** `load_hf()` and this package never import
`torch`/`transformers`/`nemo`/`qwen_asr` at import time — each family only
imports its runtime inside its first `invoke()` call. That has one
consequence worth being explicit about, since it bends the module-wide
"invoke() never raises" policy: a family whose dependency is missing (or
whose weights fail to load) raises `LoadError` **from `invoke()`**, because
that is when loading actually happens here — this is a setup/config
failure, not a per-request provider failure, so it is treated like one
(raise), not swallowed into `Response(error=...)`. Once a family's model
is loaded, a failure *during* inference on a given request does become
`Response(error=...)`, same as every other scheme.
"""

from __future__ import annotations

import importlib
from typing import Any

from benchy.core import LoadError

_FAMILY_MODULES = {
    "transformers_audio": "benchy.system.hf.whisper",
    "voxtral_chat": "benchy.system.hf.voxtral",
    "qwen3_asr_chat": "benchy.system.hf.qwen3_asr",
    "canary_nemo": "benchy.system.hf.canary",
}


def known_families() -> list[str]:
    return sorted(_FAMILY_MODULES)


def _detect_family(repo_id: str, *, trust_remote_code: bool) -> str:
    """Inspect the repo's HF config and map it to a family name.

    Hits the network (or the local HF cache) via `AutoConfig.from_pretrained`
    — callers that need this to stay hermetic (tests) should pass
    `family=...` explicitly instead, or monkeypatch
    `transformers.AutoConfig.from_pretrained`.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        if "canary" in repo_id.lower():
            return "canary_nemo"
        raise LoadError(
            f"hf: could not read the HF config for {repo_id!r} to detect its architecture "
            f"({exc}); pass family=... explicitly (one of: {', '.join(known_families())})"
        ) from exc

    model_type = (getattr(config, "model_type", "") or "").lower()
    architectures = getattr(config, "architectures", None) or []
    normalized_archs = [a.lower().replace("_", "").replace("-", "") for a in architectures]

    if model_type == "voxtral" or any("voxtral" in a for a in normalized_archs):
        return "voxtral_chat"
    if model_type in {"qwen3_asr", "qwen3-asr"} or any("qwen3asr" in a for a in normalized_archs):
        return "qwen3_asr_chat"
    if "canary" in repo_id.lower():
        return "canary_nemo"
    return "transformers_audio"


def load_hf(rest: str, **opts: Any) -> Any:
    if not rest:
        raise LoadError("hf: URL must include a repo id, e.g. 'hf:openai/whisper-large-v3-turbo'")

    family = opts.pop("family", None)
    if family is None:
        trust_remote_code = bool(opts.get("trust_remote_code", False))
        family = _detect_family(rest, trust_remote_code=trust_remote_code)
    elif family not in _FAMILY_MODULES:
        raise LoadError(f"hf: unknown family {family!r} (known: {', '.join(known_families())})")

    module = importlib.import_module(_FAMILY_MODULES[family])
    return module.load(rest, **opts)
