"""vLLM server configuration utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os


def _default_hf_cache() -> str:
    return (
        os.getenv("HF_HOME")
        or os.getenv("HF_CACHE")
        or os.getenv("HF_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface")
    )


def _default_vllm_venv_path() -> str:
    override = os.getenv("VLLM_VENV")
    if override:
        return override

    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return str(parent / ".venv")

    return str(Path.cwd() / ".venv")


@dataclass
class VLLMServerConfig:
    """Configuration for starting a vLLM server."""

    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    enforce_eager: bool = True
    # Only set when you explicitly want to cap multimodal inputs (or disable them).
    # If unset, vLLM defaults apply and no --limit-mm-per-prompt flag is passed.
    limit_mm_per_prompt: Optional[str] = None
    hf_cache: str = field(default_factory=_default_hf_cache)
    hf_token: str = ""
    startup_timeout: int = 900
    cuda_devices: Optional[str] = None
    kv_cache_memory: Optional[int] = None
    vllm_venv_path: str = field(default_factory=_default_vllm_venv_path)
    vllm_version: Optional[str] = None
    multimodal: bool = True
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    trust_remote_code: bool = True
    tokenizer_mode: Optional[str] = None
    config_format: Optional[str] = None
    load_format: Optional[str] = None
    tool_call_parser: Optional[str] = None
    enable_auto_tool_choice: bool = False
    kv_cache_dtype: Optional[str] = None
    kv_offloading_size: Optional[int] = None
    skip_mm_profiling: bool = False

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]],
        *,
        cuda_devices: Optional[str] = None,
    ) -> "VLLMServerConfig":
        values = dict(config or {})
        if cuda_devices is not None:
            values["cuda_devices"] = cuda_devices
        allowed = {field for field in cls.__dataclass_fields__}
        filtered = {key: value for key, value in values.items() if key in allowed}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
