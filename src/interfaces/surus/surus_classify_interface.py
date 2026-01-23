"""SURUS AI interface for /classify endpoint (classification tasks)."""

import json
import logging
from typing import Dict, Optional

import httpx

from ..http_interface import HTTPInterface
from ...engine.protocols import InterfaceCapabilities
from ...engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class SurusClassifyInterface(HTTPInterface):
    """Interface for SURUS AI /classify endpoint.

    The SURUS classify endpoint accepts:
      - text: string
      - labels: list[str]

    And returns an OpenAI-compatible response shape like:
      {
        "choices": [{
          "message": {"content": "{\"label\":\"positive\"}"}
        }],
        ...
      }

    Benchy `classify` tasks are multiple-choice. This interface uses:
      - `sample["text"]` as the input text to classify
      - `sample["choices"]` as the allowed label strings
    """

    capabilities = InterfaceCapabilities(
        supports_multimodal=False,
        supports_logprobs=False,
        supports_schema=False,
        supports_files=False,
        supports_streaming=False,
        request_modes=["raw_payload"],
    )

    def __init__(
        self, config: Dict, model_name: str, provider_type: str = "surus_classify"
    ):
        super().__init__(config, model_name, provider_type)
        logger.info(f"SURUS /classify endpoint: {self.endpoint}")

    def prepare_request(self, sample: Dict, task) -> Dict:
        text = str(sample.get("text", "") or "")

        labels = sample.get("choices") or []
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels) or not labels:
            raise ValueError("SurusClassifyInterface requires sample.choices as non-empty list[str].")

        return {
            "text": text,
            "labels": labels,
            "sample_id": sample["id"],
        }

    def build_test_request(self) -> Dict:
        return {
            "text": "Classify the sentiment.\n\nText: I love this!\n\nAnswer (label only):",
            "labels": ["positive", "negative", "neutral"],
            "sample_id": "connection_test",
        }

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "text": request["text"],
            "labels": request["labels"],
        }

        response = await client.post(
            self.endpoint,
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict) -> Dict:
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        try:
            if not isinstance(response, dict):
                raise TypeError(f"Response is not a JSON object: {type(response)}")

            predicted: Optional[str] = None

            # Preferred: OpenAI-compatible "choices" format.
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
                result["raw"] = content

                # content is usually a JSON string like {"label": "..."}
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        predicted = (
                            parsed.get("label")
                            or parsed.get("text")
                            or parsed.get("prediction")
                        )
                    elif isinstance(parsed, str):
                        predicted = parsed
                except json.JSONDecodeError:
                    predicted = str(content).strip()
            else:
                # Fallback: older/simple response formats.
                predicted = (
                    response.get("text")
                    or response.get("label")
                    or response.get("prediction")
                )
                result["raw"] = json.dumps(response, ensure_ascii=False)

            if not isinstance(predicted, str) or not predicted.strip():
                raise KeyError("Missing or invalid prediction in response")

            result["output"] = predicted
        except Exception as e:
            result["raw"] = json.dumps(response, ensure_ascii=False) if isinstance(response, dict) else str(response)
            result["error"] = f"Invalid response structure: {e}"
            result["error_type"] = "invalid_response"

        return result

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        logger.info(f"Testing SURUS /classify API at {self.endpoint}")

        test_request = self.build_test_request()

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await self._make_request_with_client(client, test_request)
                parsed = self._parse_response(response or {})
                if parsed.get("output") is not None:
                    return True
            raise ValueError("Empty/invalid response")

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=classify_http_exception,
        )

        if result:
            logger.info(f"Connected to SURUS /classify at {self.endpoint}")
            return True

        logger.error(f"Failed to connect to SURUS /classify after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False
