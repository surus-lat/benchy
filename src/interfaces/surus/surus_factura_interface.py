"""SURUS AI interface for /factura endpoint (image extraction without schema)."""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import httpx

from ..http_interface import HTTPInterface
from ...engine.protocols import InterfaceCapabilities
from ...engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class SurusFacturaInterface(HTTPInterface):
    """Interface for SURUS AI /factura endpoint for image extraction.

    This interface handles multimodal (image) inputs without sending a schema.
    The endpoint returns extracted data that is compared against the benchmark schema.
    """

    capabilities = InterfaceCapabilities(
        supports_multimodal=True,
        supports_schema=False,
        supports_files=True,
    )
    supports_multimodal = True

    def __init__(self, config: Dict, model_name: str, provider_type: str = "surus_factura"):
        """Initialize SURUS Factura interface."""
        super().__init__(config, model_name, provider_type)
        logger.info(f"SURUS /factura endpoint: {self.endpoint}")

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for SURUS Factura endpoint."""
        return {
            "image_path": sample["image_path"],
            "sample_id": sample["id"],
        }

    def _image_to_data_url(self, image_path: str) -> str:
        """Convert local image file to base64 data URL."""
        path = Path(image_path)

        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        with open(path, "rb") as f:
            image_data = f.read()

        b64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        """Make request to SURUS /factura endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        image_path = request["image_path"]
        if image_path.startswith(("http://", "https://", "data:")):
            image_url = image_path
        else:
            image_url = self._image_to_data_url(image_path)
            logger.debug(f"Converted local image to data URL ({len(image_url)} chars)")

        data = {
            "image_url": image_url,
        }

        response = await client.post(
            self.endpoint,
            headers=headers,
            json=data,
        )

        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict) -> Dict:
        """Parse SURUS Factura response to benchy format."""
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        try:
            if "data" in response:
                result["output"] = response["data"]
                result["raw"] = json.dumps(response["data"])
            elif "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                result["raw"] = content
                result["output"] = json.loads(content)
            else:
                raise KeyError("Neither 'data' nor 'choices' found in response")
        except KeyError as e:
            logger.warning(f"Unexpected response format: {e}")
            result["raw"] = json.dumps(response)
            result["error"] = f"Unexpected response format: {e}"
            result["error_type"] = "invalid_response"
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            result["error"] = f"JSON parse error: {e}"
            result["error_type"] = "invalid_response"

        return result

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to SURUS Factura endpoint."""
        logger.info(f"Testing SURUS Factura API at {self.endpoint}")

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                data = {
                    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                }

                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                )

                status_code = response.status_code
                error_text = response.text[:200] if hasattr(response, "text") else ""

                if 200 <= status_code < 300:
                    return True
                if status_code == 401:
                    raise httpx.HTTPStatusError(
                        f"HTTP 401 Unauthorized: {error_text}",
                        request=response.request,
                        response=response,
                    )
                if status_code == 403:
                    raise httpx.HTTPStatusError(
                        f"HTTP 403 Forbidden: {error_text}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                return False

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=lambda exc, attempt: classify_http_exception(exc, attempt),
        )

        if result:
            logger.info(f"Connected to SURUS Factura at {self.endpoint}")
            return True

        logger.error(f"Failed to connect to SURUS Factura after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False
