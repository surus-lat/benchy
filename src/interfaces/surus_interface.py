"""SURUS AI interface for /extract endpoint."""

import json
import logging
from typing import Dict, Optional

import httpx

from .http_interface import HTTPInterface

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
        text: str,
        schema: Dict
    ) -> Optional[Dict]:
        """Make request to SURUS /extract endpoint.

        Args:
            client: HTTP client
            text: Input text
            schema: JSON schema

        Returns:
            Response dictionary or None on error
        """
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
            Dict with 'output', 'raw', 'error'
        """
        result = {"output": None, "raw": None, "error": None}
        
        content = response["choices"][0]["message"]["content"]
        result["raw"] = content
        
        # SURUS returns clean JSON (no markdown), so parse directly
        result["output"] = json.loads(content)
        
        return result

