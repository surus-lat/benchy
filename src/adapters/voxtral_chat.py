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
    capabilities = InterfaceCapabilities(supports_audio=True)

    def __init__(self, model_name: str, config: dict[str, Any]):
        super().__init__(model_name, config)
        self._model = None
        self._processor = None
        self._device = "cpu"

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
                if self._device != "cpu":
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
