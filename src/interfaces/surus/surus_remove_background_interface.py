"""SURUS AI interface for /remove-background endpoint (image manipulation)."""

import base64
import json
import logging
from typing import Dict, Optional

import httpx

from ..http_interface import HTTPInterface
from ..common.image_preprocessing import encode_image_data_url
from ...engine.protocols import InterfaceCapabilities
from ...engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class SurusRemoveBackgroundInterface(HTTPInterface):
    """Interface for SURUS AI /remove-background endpoint.

    The SURUS remove-background endpoint accepts:
      - image_url: URL string (e.g., "https://example.com/image.jpg")
      - prompt: optional string for text-based segmentation
      - focus: optional "foreground" or "background" (for simple/deep processing)
      - process: optional "simple" or "deep" (requires focus parameter)

    And returns a response like:
      {
        "results": [{
          "mask": {"base64": "..."} or "image": {"base64": "..."}
        }],
        ...
      }

    This interface prioritizes image over mask if both are present.
    """

    capabilities = InterfaceCapabilities(
        supports_multimodal=True,
        supports_logprobs=False,
        supports_schema=False,
        supports_files=True,
        supports_streaming=False,
        request_modes=["raw_payload"],
    )

    def __init__(
        self, config: Dict, model_name: str, provider_type: str = "surus_remove_background"
    ):
        super().__init__(config, model_name, provider_type)
        logger.info(f"SURUS /remove-background endpoint: {self.endpoint}")

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for the remove-background endpoint.

        Args:
            sample: Sample dict with image_path and text (prompt)
            task: Task instance (provides get_prompt)

        Returns:
            Request dict for _make_request_with_client
        """
        image_path = sample.get("image_path")
        if not image_path:
            raise ValueError("SurusRemoveBackgroundInterface requires sample.image_path")

        # Get prompt from task
        system_prompt, user_prompt = task.get_prompt(sample)
        prompt = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        # Temporary: Use empty prompt.
        prompt = ""
        image_base64_uri = encode_image_data_url(
            image_path,
            max_edge=self.image_max_edge,
            logger=logger,
        )

        request = {
            "image_base64": image_base64_uri,
            "sample_id": sample["id"],
        }
        
        # Add prompt if provided (for text-based segmentation)
        # Otherwise use focus/process for automatic foreground extraction
        if prompt and prompt.strip():
            request["prompt"] = prompt
        else:
            # Default to foreground removal with simple processing
            request["focus"] = "background"
            request["process"] = "deep"
        
        return request

    def build_test_request(self) -> Dict:
        """Build a minimal request payload for connection tests."""
        # Create a minimal 1x1 PNG for testing
        minimal_png = base64.b64encode(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
            b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        ).decode("utf-8")
        
        return {
            "image_base64": f"data:image/png;base64,{minimal_png}",
            "focus": "foreground",
            "process": "simple",
            "sample_id": "connection_test",
        }

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        """Make request to SURUS /remove-background endpoint.

        Args:
            client: HTTP client
            request: Request dict with image_base64 and optional prompt/focus/process

        Returns:
            Response dictionary or None on error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build payload with image_base64 and optional parameters
        data = {
            "image_base64": request["image_base64"]
        }
        
        # Add optional parameters if present
        if "prompt" in request and request["prompt"]:
            data["prompt"] = request["prompt"]
        if "focus" in request:
            data["focus"] = request["focus"]
        if "process" in request:
            data["process"] = request["process"]

        response = await client.post(
            self.endpoint,
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict) -> Dict:
        """Parse SURUS response to benchy format.

        The SURUS remove-background endpoint returns:
        {
            "results": [{
                "mask": {"base64": "..."} or "image": {"base64": "..."}
            }],
            "detail": "error message" (if error),
            ...
        }

        Priority: Try mask first, then fallback to image.

        Args:
            response: SURUS API response

        Returns:
            Dict with 'output', 'raw', 'error', 'error_type'
        """
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        try:
            if not isinstance(response, dict):
                raise TypeError(f"Response is not a JSON object: {type(response)}")

            # Check for API error (detail field with empty results)
            if "detail" in response:
                detail = response.get("detail")
                results = response.get("results", [])
                if not results or not isinstance(results, list):
                    raise ValueError(f"API error: {detail}")

            # Navigate to results array
            results = response.get("results")
            if not isinstance(results, list) or not results:
                raise KeyError("Missing or empty 'results' array in response")

            first_result = results[0]
            if not isinstance(first_result, dict):
                raise TypeError("First result is not a dictionary")

            # Prefer image (full RGBA cutout) for storage/visualization
            # Fallback to mask (grayscale) if image not available
            base64_data = None
            source = None

            image_data = first_result.get("image")
            if isinstance(image_data, dict):
                base64_data = image_data.get("base64")
                if base64_data and isinstance(base64_data, str):
                    source = "image"

            # Fallback to mask if image not available
            if not base64_data:
                mask_data = first_result.get("mask")
                if isinstance(mask_data, dict):
                    base64_data = mask_data.get("base64")
                    if base64_data and isinstance(base64_data, str):
                        source = "mask"

            if not base64_data:
                raise KeyError("No valid 'image.base64' or 'mask.base64' found in response")

            # Strip any data URI prefix if present (normalize to raw base64)
            if base64_data.startswith("data:"):
                # Extract just the base64 part after the comma
                parts = base64_data.split(",", 1)
                if len(parts) == 2:
                    base64_data = parts[1]

            result["output"] = base64_data
            result["raw"] = json.dumps(
                {"source": source, "base64_length": len(base64_data)},
                ensure_ascii=False
            )

        except Exception as e:
            result["raw"] = json.dumps(response, ensure_ascii=False) if isinstance(response, dict) else str(response)
            result["error"] = f"Invalid response structure: {e}"
            result["error_type"] = "invalid_response"

        return result

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to SURUS /remove-background endpoint.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Testing SURUS /remove-background API at {self.endpoint}")

        test_request = self.build_test_request()

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await self._make_request_with_client(client, test_request)
                
                # Log raw response for debugging
                logger.debug(f"Connection test response: {response}")
                
                # For connection test, we just need to verify the API responds with valid structure
                # Empty results are acceptable for the minimal test image
                if isinstance(response, dict):
                    # Check if response has the expected structure (results array + model field)
                    if "results" in response and "model" in response:
                        logger.info(f"Test connection successful - API responded with valid structure")
                        return True
                    # If there's a detail field with an error message, that's a real error
                    if "detail" in response:
                        raise ValueError(f"API error: {response.get('detail')}")
                
                raise ValueError(f"Invalid response structure: {response}")

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=classify_http_exception,
        )

        if result:
            logger.info(f"Connected to SURUS /remove-background at {self.endpoint}")
            return True

        logger.error(f"Failed to connect to SURUS /remove-background after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False
