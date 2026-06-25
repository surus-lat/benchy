"""In-process HuggingFace transformers ASR interface.

Mirrors :class:`OpenAIAudioInterface`'s contract but runs the model inside
the benchy process via :func:`transformers.pipeline` instead of hitting an
HTTP endpoint. No client, no auth, runs on CPU by default.

The interface speaks the same shape as :class:`OpenAIAudioInterface`
(``prepare_request`` emits ``{sample_id, audio_path, language}``;
``generate_batch`` returns ``{output, raw, error, error_type}`` dicts), so
``TranscriptionHandler``/FLEURS subtasks/WER metrics see no difference.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..engine.protocols import InterfaceCapabilities, parse_interface_capabilities

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """Resolve ``device: auto`` to the best available backend."""
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


def _resolve_dtype(name: str):
    """Resolve a dtype name (``float32``/``float16``/``bfloat16``) to a torch dtype."""
    import torch

    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported torch_dtype: {name!r}")
    return aliases[name]


class TransformersAudioInterface:
    """HuggingFace transformers ASR pipeline backend."""

    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        self.model_name = model_name
        self.provider_type = connection_info.get("provider_type", "transformers_audio")
        self.timeout = connection_info.get("timeout", 600)
        self.max_retries = connection_info.get("max_retries", 1)
        self.supports_language_kwarg = bool(
            connection_info.get("supports_language_kwarg", True)
        )

        max_concurrent = connection_info.get("max_concurrent")
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            max_concurrent = 1
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self._capabilities = parse_interface_capabilities(
            connection_info.get("capabilities"),
            default=InterfaceCapabilities(
                supports_multimodal=False,
                supports_logprobs=False,
                supports_schema=False,
                supports_files=False,
                supports_streaming=False,
                supports_audio=True,
                supports_batch=True,
            ),
        )

        device = _resolve_device(connection_info.get("device", "auto"))
        dtype_name = connection_info.get("torch_dtype", "float32")
        chunk_length_s = connection_info.get("chunk_length_s", 30)
        trust_remote_code = bool(connection_info.get("trust_remote_code", False))

        from transformers import pipeline

        torch_dtype = _resolve_dtype(dtype_name)

        logger.info(
            "Loading transformers ASR pipeline (model=%s, device=%s, dtype=%s, chunk_length_s=%s, trust_remote_code=%s)",
            model_name,
            device,
            dtype_name,
            chunk_length_s,
            trust_remote_code,
        )
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=chunk_length_s,
            trust_remote_code=trust_remote_code,
        )
        logger.info("Initialized TransformersAudioInterface for %s", model_name)

    def prepare_request(self, sample: Dict, task: Any) -> Dict:
        if "audio_path" not in sample:
            raise ValueError(
                f"Sample '{sample.get('id', '?')}' missing 'audio_path' — "
                "transcription tasks must populate this field"
            )
        return {
            "sample_id": sample["id"],
            "audio_path": sample["audio_path"],
            "language": sample.get("language") or getattr(task, "language", None),
        }

    async def _transcribe_single(
        self,
        audio_path: str,
        language: Optional[str],
        sample_id: str,
    ) -> Dict[str, Any]:
        def _run() -> str:
            kwargs: Dict[str, Any] = {}
            if language and self.supports_language_kwarg:
                kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}
            result = self._pipeline(audio_path, **kwargs)
            if isinstance(result, dict):
                return str(result.get("text", "")).strip()
            return str(result).strip()

        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_run), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Transcription timeout for sample %s", sample_id)
            return {
                "output": None,
                "raw": None,
                "error": f"transcription timed out after {self.timeout}s",
                "error_type": "TimeoutError",
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            logger.exception("Transcription failed for sample %s", sample_id)
            return {
                "output": None,
                "raw": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

        return {"output": text, "raw": text, "error": None, "error_type": None}

    async def _generate_with_limit(self, req: Dict) -> Dict:
        async with self._semaphore:
            return await self._transcribe_single(
                audio_path=req["audio_path"],
                language=req.get("language"),
                sample_id=req["sample_id"],
            )

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        tasks = [self._generate_with_limit(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[Dict] = []
        successful = 0
        errors = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Request %s failed: %s",
                    requests[i].get("sample_id", "?"),
                    result,
                )
                processed.append(
                    {
                        "output": None,
                        "raw": None,
                        "error": str(result),
                        "error_type": "connectivity_error",
                    }
                )
                errors += 1
            else:
                processed.append(result)
                if result.get("output") is not None:
                    successful += 1
                elif result.get("error"):
                    errors += 1

        logger.info(
            "Batch: %d/%d successful, %d errors", successful, len(requests), errors
        )
        return processed

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        # Pipeline is already loaded in __init__; if we got here, it works.
        return self._pipeline is not None

    @property
    def supports_multimodal(self) -> bool:
        return self._capabilities.supports_multimodal

    @property
    def supports_logprobs(self) -> bool:
        return self._capabilities.supports_logprobs

    @property
    def capabilities(self) -> InterfaceCapabilities:
        return self._capabilities

    async def close(self) -> None:
        self._pipeline = None
