"""Shared device/dtype resolution for the `hf:` families.

`torch` is imported lazily, inside these functions, never at module level.
"""

from __future__ import annotations


def resolve_device(requested: str) -> str:
    """Resolve `"auto"` to the best available backend; pass anything else through."""
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


_DTYPE_ALIASES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def resolve_dtype(name: str):
    """Resolve a dtype name (`float32`/`float16`/`half`/`bfloat16`/...) to a torch dtype."""
    import torch

    key = _DTYPE_ALIASES.get(name)
    if key is None:
        raise ValueError(f"unsupported torch_dtype: {name!r}")
    return getattr(torch, key)
