"""SURUS AI interface for /ocr endpoint (image extraction)."""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .http_interface import HTTPInterface

logger = logging.getLogger(__name__)


class SurusOCRInterface(HTTPInterface):
    """Interface for SURUS AI /ocr endpoint for image extraction.
    
    This interface handles multimodal (image) inputs, converting local
    image files to base64 data URLs for the API.
    """

    # Flag for compatibility check with multimodal tasks
    supports_multimodal = True

    def __init__(self, config: Dict, model_name: str, provider_type: str = "surus_ocr"):
        """Initialize SURUS OCR interface.

        Args:
            config: Configuration dictionary
            model_name: System identifier
            provider_type: Provider type (default: "surus_ocr")
        """
        super().__init__(config, model_name, provider_type)
        logger.info(f"SURUS /ocr endpoint: {self.endpoint}")

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for SURUS OCR endpoint.
        
        Handles image_path from image_extraction task samples.
        
        Args:
            sample: Raw sample with image_path, schema, expected, etc.
            task: Task instance (not used for HTTP, but kept for interface consistency)
            
        Returns:
            Dict formatted for this interface's generate_batch()
        """
        return {
            "image_path": sample["image_path"],
            "schema": sample["schema"],
            "sample_id": sample["id"],
        }

    def _image_to_data_url(self, image_path: str) -> str:
        """Convert local image file to base64 data URL.
        
        Args:
            image_path: Path to local image file
            
        Returns:
            Data URL string (data:image/jpeg;base64,...)
        """
        path = Path(image_path)
        
        # Determine MIME type
        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")
        
        # Read and encode
        with open(path, "rb") as f:
            image_data = f.read()
        
        b64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

    async def _generate_single(
        self,
        image_path: str,
        schema: Dict,
        sample_id: str,
    ) -> Dict:
        """Generate structured output for a single image sample.

        Args:
            image_path: Path to local image file
            schema: Target JSON schema
            sample_id: Sample identifier for logging

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error', 'error_type'
        """
        import asyncio
        
        result = {"output": None, "raw": None, "error": None, "error_type": None}
        last_error = None
        last_error_type = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await self._make_request_with_client(
                        client, image_path, schema
                    )
                    
                    if response:
                        result = self._parse_response(response)
                        if result["output"] is not None:
                            return result
                        # If we got a response but couldn't parse it, that's an invalid_response
                        last_error = result.get("error", "Empty response")
                        last_error_type = result.get("error_type", "invalid_response")
            except httpx.TimeoutException:
                last_error = f"Timeout after {self.timeout}s"
                last_error_type = "connectivity_error"
                logger.debug(f"[{sample_id}] Attempt {attempt + 1} timed out")
                # Exponential backoff for connectivity errors
                if attempt < self.max_retries - 1:
                    delay = min(2 ** attempt, 16)  # 1s, 2s, 4s, 8s, 16s max
                    logger.debug(f"[{sample_id}] Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
            except httpx.ConnectError as e:
                last_error = f"Connection failed: {e}"
                last_error_type = "connectivity_error"
                logger.debug(f"[{sample_id}] Attempt {attempt + 1} connection failed")
                # Exponential backoff for connectivity errors
                if attempt < self.max_retries - 1:
                    delay = min(2 ** attempt, 16)  # 1s, 2s, 4s, 8s, 16s max
                    logger.debug(f"[{sample_id}] Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                last_error = f"HTTP {status_code}"
                # 5xx errors are connectivity errors, others are invalid responses
                if status_code >= 500:
                    last_error_type = "connectivity_error"
                    logger.debug(f"[{sample_id}] Attempt {attempt + 1} got HTTP {status_code} (connectivity error)")
                    # Exponential backoff for connectivity errors
                    if attempt < self.max_retries - 1:
                        delay = min(2 ** attempt, 16)  # 1s, 2s, 4s, 8s, 16s max
                        logger.debug(f"[{sample_id}] Waiting {delay}s before retry...")
                        await asyncio.sleep(delay)
                else:
                    last_error_type = "invalid_response"
                    logger.debug(f"[{sample_id}] Attempt {attempt + 1} got HTTP {status_code} (invalid response)")
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                # Unknown exceptions are treated as connectivity errors (might be network-related)
                last_error_type = "connectivity_error"
                logger.debug(f"[{sample_id}] Attempt {attempt + 1} failed: {last_error}")
                # Exponential backoff for connectivity errors
                if attempt < self.max_retries - 1:
                    delay = min(2 ** attempt, 16)  # 1s, 2s, 4s, 8s, 16s max
                    logger.debug(f"[{sample_id}] Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
        
        result["error"] = last_error or "All retry attempts exhausted"
        result["error_type"] = last_error_type
        logger.warning(f"[{sample_id}] All {self.max_retries} attempts failed: {result['error']} (type: {result['error_type']})")
        return result

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        image_path: str,
        schema: Dict
    ) -> Optional[Dict]:
        """Make request to SURUS /ocr endpoint.

        Args:
            client: HTTP client
            image_path: Path to local image file (or URL)
            schema: JSON schema

        Returns:
            Response dictionary or None on error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if image_path is already a URL or needs conversion
        if image_path.startswith(("http://", "https://", "data:")):
            image_url = image_path
        else:
            # Convert local file to base64 data URL
            image_url = self._image_to_data_url(image_path)
            logger.debug(f"Converted local image to data URL ({len(image_url)} chars)")
        
        data = {
            "image_url": image_url,
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
        """Parse SURUS OCR response to benchy format.

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
        result = {"output": None, "raw": None, "error": None, "error_type": None}
        
        try:
            content = response["choices"][0]["message"]["content"]
            result["raw"] = content
            
            # SURUS returns clean JSON (no markdown), so parse directly
            result["output"] = json.loads(content)
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

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate structured outputs for a batch of image samples.

        Args:
            requests: List of request dicts with keys:
                - image_path: Path to image file
                - schema: Target JSON schema
                - sample_id: Sample identifier

        Returns:
            List of result dictionaries in same order as requests
        """
        import asyncio
        
        tasks = [
            self._generate_single(
                image_path=req["image_path"],
                schema=req["schema"],
                sample_id=req["sample_id"],
            )
            for req in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        successful = 0
        errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed: {result}")
                processed_results.append({
                    "output": None,
                    "raw": None,
                    "error": str(result),
                    "error_type": "connectivity_error",  # Exceptions are typically connectivity issues
                })
                errors += 1
            else:
                processed_results.append(result)
                if result.get("output") is not None:
                    successful += 1
                else:
                    errors += 1
        
        logger.info(f"📊 Batch: {successful}/{len(requests)} successful, {errors} errors")
        
        return processed_results

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to SURUS OCR endpoint.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"🚀 Testing SURUS OCR API at {self.endpoint}")
        
        # Minimal test - just check if the endpoint responds
        # We can't easily send a real test image, so we check headers only
        for attempt in range(max_retries):
            logger.info(f"Connection test attempt {attempt + 1}/{max_retries}...")
            
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    # Send a minimal request to verify auth and endpoint
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Minimal test with a tiny test schema
                    data = {
                        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        "json_schema": {
                            "type": "object",
                            "properties": {"test": {"type": "string"}}
                        }
                    }
                    
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        json=data
                    )
                    
                    # Accept any 2xx or 4xx (auth/validation) as "connected"
                    # 5xx would indicate server issues
                    if response.status_code < 500:
                        logger.info(f"✓ Connected to SURUS OCR at {self.endpoint}")
                        return True
                        
            except httpx.ConnectError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} error: {e}")
        
        logger.error(f"✗ Failed to connect to SURUS OCR after {max_retries} attempts")
        return False





