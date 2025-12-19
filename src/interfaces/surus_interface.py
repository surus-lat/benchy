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
            
        Raises:
            httpx.TimeoutException: On request timeout
            httpx.ConnectError: On connection failure
            httpx.HTTPStatusError: On HTTP error responses
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
        logger.info(f"🚀 Testing SURUS /extract API at {self.endpoint}")
        
        # Simple test request
        test_text = "Test connection."
        test_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            }
        }
        
        last_error = None
        for attempt in range(max_retries):
            logger.info(f"Connection test attempt {attempt + 1}/{max_retries}...")
            
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await self._make_request_with_client(client, test_text, test_schema)
                    if response:
                        logger.info(f"✓ Connected to SURUS /extract at {self.endpoint}")
                        return True
            except httpx.TimeoutException as e:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"  ⏱ Attempt {attempt + 1} timed out after {timeout}s")
            except httpx.ConnectError as e:
                last_error = f"Connection failed: {e}"
                logger.warning(f"  🔌 Attempt {attempt + 1} connection failed: {e}")
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_text = e.response.text[:200] if hasattr(e.response, 'text') else ""
                
                # Only accept 2xx as success
                # 401/403 indicate auth/quota issues - these should fail the connection test
                if 200 <= status_code < 300:
                    logger.info(f"✓ Connected to SURUS /extract at {self.endpoint}")
                    return True
                elif status_code == 401:
                    last_error = f"HTTP 401 Unauthorized: Invalid API key or authentication failed. {error_text}"
                    logger.error(f"  🔑 Attempt {attempt + 1} authentication failed (HTTP 401)")
                elif status_code == 403:
                    last_error = f"HTTP 403 Forbidden: API key valid but access denied (quota exhausted or insufficient permissions). {error_text}"
                    logger.error(f"  🚫 Attempt {attempt + 1} access denied (HTTP 403): {error_text}")
                else:
                    last_error = f"HTTP {status_code}: {error_text}"
                    logger.warning(f"  ❌ Attempt {attempt + 1} got HTTP {status_code}: {error_text}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"  ⚠ Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
        
        logger.error(f"✗ Failed to connect to SURUS /extract after {max_retries} attempts")
        if last_error:
            logger.error(f"  Last error: {last_error}")
        return False

