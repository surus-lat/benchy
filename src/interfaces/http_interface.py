"""HTTP interface for task-optimized AI systems."""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class HTTPInterface:
    """Base interface for HTTP-based AI systems with custom endpoints."""

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
        
        # HTTP configuration
        self.endpoint = self.config["endpoint"]
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Get API key
        api_key_env = self.config.get("api_key_env", f"{provider_type.upper()}_API_KEY")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Set {api_key_env} environment variable.\n"
                f"  - Add to .env file: {api_key_env}=your-key-here"
            )
        
        logger.info(f"Initialized {provider_type} HTTP interface for {model_name}")

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

    async def _make_request(self, text: str, schema: Dict) -> Dict:
        """Make HTTP request to endpoint.
        
        Override this in subclasses for provider-specific formatting.
        
        Args:
            text: Text to process
            schema: JSON schema for extraction
            
        Returns:
            Response dictionary
        """
        raise NotImplementedError("Subclasses must implement _make_request")

    async def _generate_single(
        self,
        text: str,
        schema: Dict,
        sample_id: str,
    ) -> Dict:
        """Generate structured output for a single sample.

        Args:
            text: Input text
            schema: Target JSON schema
            sample_id: Sample identifier for logging

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error'
        """
        result = {"output": None, "raw": None, "error": None}

        for attempt in range(self.max_retries):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await self._make_request_with_client(client, text, schema)
                
                if response:
                    result = self._parse_response(response)
                    if result["output"] is not None:
                        return result
        
        result["error"] = "All retry attempts exhausted"
        logger.warning(f"[{sample_id}] All {self.max_retries} attempts failed")
        return result

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        text: str,
        schema: Dict
    ) -> Optional[Dict]:
        """Make request with given client. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _make_request_with_client")

    def _parse_response(self, response: Dict) -> Dict:
        """Parse response to benchy format. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _parse_response")

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate structured outputs for a batch of samples.

        Args:
            requests: List of request dicts with keys:
                - text: Input text
                - schema: Target JSON schema
                - sample_id: Sample identifier

        Returns:
            List of result dictionaries in same order as requests
        """
        tasks = [
            self._generate_single(
                text=req["text"],
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
        """Test connection to HTTP endpoint.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"🚀 Testing {self.provider_type} API at {self.endpoint}")
        
        # Simple test request
        test_text = "Test connection."
        test_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            }
        }
        
        for attempt in range(max_retries):
            logger.info(f"Connection test attempt {attempt + 1}/{max_retries}...")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await self._make_request_with_client(client, test_text, test_schema)
                if response:
                    logger.info(f"✓ Connected to {self.provider_type} at {self.endpoint}")
                    return True
        
        logger.error(f"✗ Failed to connect to {self.provider_type} after {max_retries} attempts")
        return False

