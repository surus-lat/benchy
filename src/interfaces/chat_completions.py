"""Chat completions interface for OpenAI-compatible APIs.

This interface works with any OpenAI-compatible endpoint:
- OpenAI API
- vLLM server (local or remote)
- Anthropic via compatibility layer
- Other OpenAI-compatible services

The interface is provider-agnostic - it just needs a base_url and api_key.
Supports multimodal (vision) requests when samples include image_path.
"""

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ChatCompletionsInterface:
    """Interface for OpenAI-compatible chat completions APIs.
    
    This interface adapts task data for chat-based LLM APIs.
    It calls task.get_prompt() to build messages.
    """
    
    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        """Initialize the chat completions interface.
        
        Args:
            connection_info: Connection configuration with:
                - base_url: API endpoint URL (e.g., "https://api.openai.com/v1")
                - api_key: API key string, or None to read from env
                - api_key_env: Env var name for API key (default: OPENAI_API_KEY)
                - timeout: Request timeout in seconds (default: 120)
                - max_retries: Max retry attempts (default: 3)
                - temperature: Generation temperature (default: 0.0)
                - max_tokens: Max tokens to generate (default: 2048)
                - max_tokens_param_name: Parameter name to use ("max_tokens" or "max_completion_tokens", default: "max_tokens")
                - max_concurrent: Max concurrent requests (default: 2 for cloud)
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.base_url = connection_info["base_url"]
        self.timeout = connection_info.get("timeout", 120)
        self.max_retries = connection_info.get("max_retries", 3)
        self.temperature = connection_info.get("temperature", 0.0)
        self.max_tokens = connection_info.get("max_tokens", 2048)
        self.max_tokens_param_name = connection_info.get("max_tokens_param_name", "max_tokens")
        
        # Get API key
        api_key = connection_info.get("api_key")
        if not api_key:
            api_key_env = connection_info.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env)
            if not api_key:
                # For local vLLM servers, use a dummy key
                api_key = "EMPTY"
        
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        
        # Optional: structured outputs for vLLM (v0.12.0+ API)
        self.use_structured_outputs = connection_info.get("use_structured_outputs", False)
        
        # Rate limiting for cloud APIs (avoid 429 errors)
        # Default to 2 concurrent for cloud APIs, higher for local vLLM
        is_cloud = "openai.com" in self.base_url or "anthropic.com" in self.base_url
        default_concurrent = 2 if is_cloud else 20
        max_concurrent = connection_info.get("max_concurrent", default_concurrent)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._is_cloud = is_cloud
        
        logger.info(f"Initialized ChatCompletionsInterface for {model_name}")
        logger.info(f"  Base URL: {self.base_url}")
        if is_cloud:
            logger.info(f"  Rate limit: max {max_concurrent} concurrent requests")
    
    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request by getting prompts from task.
        
        Args:
            sample: Sample dictionary with id, text, expected, etc.
                    May include image_path for multimodal requests.
            task: Task instance with get_prompt() method
            
        Returns:
            Request dict with system_prompt, user_prompt, schema, sample_id,
            and optionally image_path for multimodal requests.
        """
        system_prompt, user_prompt = task.get_prompt(sample)
        request = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": sample.get("schema"),  # May be None for non-schema tasks
            "sample_id": sample["id"],
        }
        
        # Include image_path for multimodal requests
        if "image_path" in sample:
            request["image_path"] = sample["image_path"]
        
        return request
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Get media type from image file extension.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Media type string (e.g., "image/jpeg")
        """
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(ext, "image/jpeg")

    async def _generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        sample_id: str,
        image_path: Optional[str] = None,
    ) -> Dict:
        """Generate output for a single request.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Optional JSON schema for guided generation
            sample_id: Sample ID for logging
            image_path: Optional path to image for multimodal requests
            
        Returns:
            Dict with 'output', 'raw', 'error'
        """
        result = {"output": None, "raw": None, "error": None}
        
        # Build user content once (multimodal or text-only)
        if image_path:
            base64_image = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}}
            ]
        else:
            user_content = user_prompt
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            self.max_tokens_param_name: self.max_tokens,
            "timeout": self.timeout,
        }
        
        if schema and self.use_structured_outputs:
            params["extra_body"] = {"structured_outputs": {"json": schema}}
        elif schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema}
            }
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(**params)
                raw_output = response.choices[0].message.content
                result["raw"] = raw_output
                
                cleaned = self._extract_json(raw_output)
                try:
                    result["output"] = json.loads(cleaned)
                except json.JSONDecodeError:
                    result["output"] = raw_output
                
                return result
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
                
                if is_rate_limit and self._is_cloud:
                    # Exponential backoff for rate limits: 5s, 15s, 45s
                    wait_time = 5 * (3 ** attempt)
                    logger.warning(f"[{sample_id}] Rate limited, waiting {wait_time}s before retry {attempt + 2}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"[{sample_id}] Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    result["error"] = error_str
        
        return result
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()
        
        # Try markdown code blocks
        patterns = [
            r'```json\s*\n?(.*?)\n?```',
            r'```\s*\n?(.*?)\n?```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Try to find JSON object
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        
        if first_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace + 1]
        
        return text
    
    async def _generate_with_limit(self, req: Dict) -> Dict:
        """Generate with rate limiting via semaphore."""
        async with self._semaphore:
            return await self._generate_single(
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                schema=req.get("schema"),
                sample_id=req["sample_id"],
                image_path=req.get("image_path"),
            )

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate outputs for a batch of requests.
        
        Args:
            requests: List of request dicts from prepare_request()
            
        Returns:
            List of result dicts with 'output', 'raw', 'error'
        """
        tasks = [self._generate_with_limit(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed = []
        successful = 0
        errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed: {result}")
                processed.append({"output": None, "raw": None, "error": str(result)})
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
                # Try to list models (works for OpenAI, may fail for other providers)
                try:
                    models = await asyncio.wait_for(
                        self.client.models.list(),
                        timeout=timeout
                    )
                    logger.info(f"Connected to API at {self.base_url}")
                    
                    # Check if our model is available
                    model_ids = [m.id for m in models.data]
                    if self.model_name in model_ids:
                        logger.info(f"Model '{self.model_name}' is available")
                    else:
                        logger.warning(f"Model '{self.model_name}' not in list, but may still work")
                    
                    return True
                except (AttributeError, TypeError) as e:
                    # Some providers (Together AI, etc.) return different formats
                    # Fall back to a simple completion test
                    logger.info(f"Models list not compatible, testing with simple completion...")
                    
                    test_params = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "Hi"}],
                        self.max_tokens_param_name: 5,
                    }
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(**test_params),
                        timeout=timeout
                    )
                    
                    if response.choices and response.choices[0].message:
                        logger.info(f"Connected to API at {self.base_url}")
                        logger.info(f"Model '{self.model_name}' responded successfully")
                        return True
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
        
        logger.error(f"Failed to connect after {max_retries} attempts")
        return False
    
    @property
    def supports_multimodal(self) -> bool:
        """Whether this interface supports multimodal inputs."""
        return True  # Supports OpenAI vision API

