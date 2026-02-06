"""HTTP interface for task-optimized AI systems."""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any

import httpx

from ..engine.protocols import InterfaceCapabilities, parse_interface_capabilities
from ..engine.retry import RetryableError, classify_http_exception, run_with_retries
from .common.image_preprocessing import coerce_positive_int

logger = logging.getLogger(__name__)


class HTTPInterface:
    """Base interface for HTTP-based AI systems with custom endpoints."""

    capabilities = InterfaceCapabilities(supports_schema=True)

    def __init__(self, config: Dict, model_name: str, provider_type: str = "http"):
        """Initialize HTTP interface.

        Args:
            config: Configuration dictionary with model settings
            model_name: Name/identifier for this system (used for logging)
            provider_type: Type of provider
        """
        self.config = config.get(provider_type, {})
        self.model_name = model_name
        self.provider_type = provider_type
        self.capabilities = parse_interface_capabilities(
            self.config.get("capabilities"),
            default=self.capabilities,
        )
        
        # HTTP configuration
        self.endpoint = self.config["endpoint"]
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_invalid_response = self.config.get("retry_invalid_response", True)
        self.retry_on_4xx = self.config.get("retry_on_4xx", False)
        self.image_max_edge = coerce_positive_int(
            self.config.get("image_max_edge"),
            option_name="image_max_edge",
            logger=logger,
        )
        
        # Get API key
        api_key_env = self.config.get("api_key_env", f"{provider_type.upper()}_API_KEY")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Set {api_key_env} environment variable.\n"
                f"  - Add to .env file: {api_key_env}=your-key-here"
            )
        
        logger.info(f"Initialized {provider_type} HTTP interface for {model_name}")
        if isinstance(self.image_max_edge, int):
            logger.info(f"  Image scaling enabled: max edge={self.image_max_edge}px")

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for HTTP endpoint.
        
        HTTP interfaces use raw data, not prompts.
        Override this in subclasses for custom formatting.
        
        Args:
            sample: Raw sample with text, schema, expected, etc.
            task: Task instance (not used for HTTP, but kept for interface consistency)
            
        Returns:
            Dict formatted for this interface's generate_batch()
        """
        return {
            "text": sample["text"],
            "schema": sample["schema"],
            "sample_id": sample["id"],
        }

    def build_test_request(self) -> Dict[str, Any]:
        """Build a minimal request payload for connection tests."""
        return {
            "text": "Test connection.",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"}
                }
            },
            "sample_id": "connection_test",
        }

    async def _generate_single(
        self,
        request: Dict,
    ) -> Dict:
        """Generate structured output for a single sample.

        Args:
            request: Request dict from prepare_request()

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error', 'error_type'
        """
        sample_id = request.get("sample_id", "unknown")
        last_result: Optional[Dict[str, Any]] = None

        async def attempt_fn(_: int) -> Dict:
            nonlocal last_result
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await self._make_request_with_client(client, request)
                if not response:
                    raise RetryableError("Empty response", error_type="invalid_response", retry_after=0.0)
                result = self._parse_response(response)
                last_result = result
                if result.get("output") is None and self.retry_invalid_response:
                    raise RetryableError(
                        result.get("error", "Invalid response"),
                        error_type=result.get("error_type", "invalid_response"),
                        retry_after=0.0,
                    )
                return result

        result, error, error_type = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=lambda exc, attempt: classify_http_exception(
                exc,
                attempt,
                retry_on_4xx=self.retry_on_4xx,
            ),
        )

        if result is not None:
            return result

        fallback = last_result or {"output": None, "raw": None, "error": None, "error_type": None}
        fallback["output"] = None
        fallback["error"] = error
        fallback["error_type"] = error_type
        logger.warning(
            f"[{sample_id}] All {self.max_retries} attempts failed: {fallback['error']} (type: {fallback['error_type']})"
        )
        return fallback

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        """Make request with given client. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _make_request_with_client")

    def _parse_response(self, response: Dict) -> Dict:
        """Parse response to benchy format. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _parse_response")

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate structured outputs for a batch of samples.

        Args:
            requests: List of request dicts from prepare_request()

        Returns:
            List of result dictionaries in same order as requests
        """
        tasks = [
            self._generate_single(
                request=req,
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
        
        logger.info(f"Batch: {successful}/{len(requests)} successful, {errors} errors")
        
        return processed_results

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to HTTP endpoint.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Testing {self.provider_type} API at {self.endpoint}")
        
        test_request = self.build_test_request()

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await self._make_request_with_client(client, test_request)
                if response:
                    return True
            raise RetryableError("Empty response", error_type="invalid_response", retry_after=0.0)

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=lambda exc, attempt: classify_http_exception(
                exc,
                attempt,
                retry_on_4xx=self.retry_on_4xx,
            ),
        )

        if result:
            logger.info(f"Connected to {self.provider_type} at {self.endpoint}")
            return True

        logger.error(f"Failed to connect to {self.provider_type} after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False
