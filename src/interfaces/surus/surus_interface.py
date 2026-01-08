"""SURUS AI interface for /extract endpoint."""

import json
import logging
from typing import Dict, Optional

import httpx

from ..http_interface import HTTPInterface
from ...engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class SurusInterface(HTTPInterface):
    """Interface for SURUS AI /extract endpoint."""

    def __init__(self, config: Dict, model_name: str, provider_type: str = "surus"):
        """Initialize SURUS interface.

        Args:
            config: Configuration dictionary
            model_name: System identifier
            provider_type: Provider type (default: "surus")
        """
        super().__init__(config, model_name, provider_type)
        logger.info(f"SURUS /extract endpoint: {self.endpoint}")

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        """
        Send the provided text and optional schema to the SURUS /extract endpoint and return the parsed JSON response.
        
        Parameters:
            request (Dict): Dictionary containing:
                - "text" (str): The input text to extract from.
                - "schema" (Optional[Dict]): An optional JSON schema; if present it will be sanitized for OpenAI compatibility.
        
        Returns:
            dict: Parsed JSON response from the SURUS API.
        
        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.ConnectError: If a connection error occurs.
            httpx.HTTPStatusError: If the response has a non-2xx HTTP status.
        """
        text = request["text"]
        schema = request["schema"]
        
        # Sanitize schema for OpenAI compatibility (SURUS uses OpenAI-compatible format)
        if schema:
            from ...common.schema_sanitizer import sanitize_schema_for_openai_strict
            schema = sanitize_schema_for_openai_strict(schema)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "json_schema": schema
        }
        
        response = await client.post(
            self.endpoint,
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict) -> Dict:
        """Parse SURUS response to benchy format.

        SURUS returns OpenAI-compatible format:
        {
            "choices": [{
                "message": {
                    "content": "{"key": "value"}"  // JSON as string
                }
            }],
            ...
        }

        Args:
            response: SURUS API response

        Returns:
            Dict with 'output', 'raw', 'error', 'error_type'
        """
        result = {"output": None, "raw": None, "error": None, "error_type": None}
        
        try:
            content = response["choices"][0]["message"]["content"]
            result["raw"] = content
            
            # SURUS returns clean JSON (no markdown), so parse directly
            result["output"] = json.loads(content)
        except (KeyError, IndexError) as e:
            result["error"] = f"Invalid response structure: {e}"
            result["error_type"] = "invalid_response"
        except json.JSONDecodeError as e:
            result["error"] = f"Failed to parse JSON: {e}"
            result["error_type"] = "invalid_response"
        
        return result

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to SURUS /extract endpoint.
        
        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Testing SURUS /extract API at {self.endpoint}")

        test_request = self.build_test_request()

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await self._make_request_with_client(client, test_request)
                if response:
                    return True
            raise ValueError("Empty response")

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=classify_http_exception,
        )

        if result:
            logger.info(f"Connected to SURUS /extract at {self.endpoint}")
            return True

        logger.error(f"Failed to connect to SURUS /extract after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False