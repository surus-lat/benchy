"""OpenAI Whisper interface for audio transcription.

Subclasses ``OpenAIInterface`` so we inherit the AsyncOpenAI client setup,
concurrency semaphore, and retry classifier. Only three things differ from
the chat path: capability defaults, request preparation, and the per-request
generation method (which hits ``/v1/audio/transcriptions`` instead of
``/v1/chat/completions``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..engine.protocols import InterfaceCapabilities, parse_interface_capabilities
from ..engine.retry import run_with_retries
from .openai_interface import OpenAIInterface

logger = logging.getLogger(__name__)


class OpenAIAudioInterface(OpenAIInterface):
    """Interface for OpenAI Whisper-style audio transcription endpoints."""

    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        super().__init__(connection_info, model_name)
        # Whisper exposes audio but none of the chat-style features. Replace the
        # capabilities the parent built (chat defaults) with audio-appropriate ones.
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
        logger.info(f"Initialized OpenAIAudioInterface for {model_name}")

    def prepare_request(self, sample: Dict, task) -> Dict:
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
    ) -> Dict:
        result: Dict[str, Any] = {
            "output": None,
            "raw": None,
            "error": None,
            "error_type": None,
        }

        async def attempt_fn(_attempt: int) -> Dict[str, Any]:
            with open(audio_path, "rb") as f:
                transcription = await self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=f,
                    language=language,
                    response_format="text",
                    timeout=self.timeout,
                )
            text = str(transcription).strip() if transcription else ""
            return {
                "output": text,
                "raw": text,
                "error": None,
                "error_type": None,
            }

        response, error, error_type = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=self._classify_api_error,
        )
        if response is not None:
            return response
        result["error"] = error
        result["error_type"] = error_type
        return result

    async def _generate_with_limit(self, req: Dict) -> Dict:
        async with self._semaphore:
            return await self._transcribe_single(
                audio_path=req["audio_path"],
                language=req.get("language"),
                sample_id=req["sample_id"],
            )
