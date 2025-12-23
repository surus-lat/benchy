"""Template interface - copy and customize for new AI system integrations.

To create a new interface:
1. Copy this file: cp src/interfaces/_template.py src/interfaces/my_interface.py
2. Rename the class and update implementation
3. Add to __init__.py exports
4. Optionally add to engine/connection.py for auto-selection
"""

import asyncio
import logging
import os
from typing import Dict, List, Any

from ..engine.protocols import BaseInterface, InterfaceCapabilities
from ..engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class TemplateInterface(BaseInterface):
    """Template interface for custom AI systems.
    
    Inherits from BaseInterface protocol for IDE autocompletion and type checking.
    
    Copy this and customize for your specific API.
    """
    
    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        """Initialize the interface.
        
        Args:
            connection_info: Standardized connection dict with:
                - base_url: API endpoint
                - api_key or api_key_env: Authentication
                - timeout: Request timeout
                - max_retries: Retry attempts
                - Other API-specific settings
            model_name: Name of model/system being evaluated
        """
        self.model_name = model_name
        self.base_url = connection_info["base_url"]
        self.timeout = connection_info.get("timeout", 120)
        self.max_retries = connection_info.get("max_retries", 3)
        
        # Get API key
        api_key = connection_info.get("api_key")
        if not api_key:
            api_key_env = connection_info.get("api_key_env", "API_KEY")
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found. Set {api_key_env} environment variable.")
        self.api_key = api_key
        
        # Initialize your client here
        # self.client = ...
        
        logger.info(f"Initialized TemplateInterface for {model_name}")
        logger.info(f"  Base URL: {self.base_url}")

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request from task sample.
        
        This method adapts task data to your API's format.
        
        For LLM-like interfaces (need prompts):
            system_prompt, user_prompt = task.get_prompt(sample)
            return {
                "messages": [...],
                "sample_id": sample["id"],
            }
        
        For HTTP interfaces (raw data):
            return {
                "text": sample["text"],
                "sample_id": sample["id"],
            }
        
        Args:
            sample: Sample dict from task.get_samples()
            task: Task instance (call task.get_prompt() if needed)
            
        Returns:
            Request dict for generate_batch()
        """
        # Example: LLM-style with prompts
        system_prompt, user_prompt = task.get_prompt(sample)
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "sample_id": sample["id"],
            # Add any other fields your API needs
        }

    async def _generate_single(self, request: Dict) -> Dict:
        """Generate output for a single request.
        
        Args:
            request: Request dict from prepare_request()
            
        Returns:
            Result dict with 'output', 'raw', 'error'
        """
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        async def attempt_fn(_: int) -> Dict:
            # Make your API call here
            # response = await self.client.generate(...)
            # raw_output = response.text
            # result["raw"] = raw_output
            # result["output"] = parse_output(raw_output)
            result["error"] = "Not implemented"
            return result

        response, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=classify_http_exception,
        )

        fallback = dict(result)
        fallback["error"] = error
        return response or fallback

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate outputs for a batch of requests.
        
        Args:
            requests: List of request dicts from prepare_request()
            
        Returns:
            List of result dicts with 'output', 'raw', 'error'
        """
        # Run requests concurrently
        tasks = [self._generate_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed = []
        successful = 0
        errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed: {result}")
                processed.append({
                    "output": None,
                    "raw": None,
                    "error": str(result),
                    "error_type": "connectivity_error",
                })
                errors += 1
            else:
                processed.append(result)
                if result.get("output") is not None:
                    successful += 1
                elif result.get("error"):
                    errors += 1
        
        logger.info(f"Batch: {successful}/{len(requests)} successful, {errors} errors")
        return processed

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to the API.
        
        Args:
            max_retries: Max connection attempts
            timeout: Timeout per attempt
            
        Returns:
            True if connection successful
        """
        logger.info(f"Testing connection to {self.base_url}")
        
        for attempt in range(max_retries):
            try:
                # Make a simple test request
                # response = await self.client.test()
                
                logger.info(f"Connected to API at {self.base_url}")
                return True
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
        
        logger.error(f"Failed to connect after {max_retries} attempts")
        return False

    @property
    def supports_multimodal(self) -> bool:
        """Whether this interface supports multimodal inputs."""
        return False  # Override if your API supports images/audio

    @property
    def capabilities(self) -> InterfaceCapabilities:
        """Structured capability flags for compatibility checks."""
        return InterfaceCapabilities(
            supports_multimodal=self.supports_multimodal,
            supports_logprobs=False,
            supports_schema=False,
            supports_files=False,
        )
