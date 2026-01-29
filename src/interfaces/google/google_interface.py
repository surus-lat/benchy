"""Generic Google Gemini interface for text and multimodal tasks."""

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, List

from google import genai
from PIL import Image

from ...engine.protocols import InterfaceCapabilities, parse_interface_capabilities
from ...engine.retry import RetryableError, classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class GoogleInterface:
    """Generic interface for Google Gemini models.
    
    Supports:
    - Text generation (like chat models)
    - Multimodal tasks (text + images)
    - Image artifact generation (e.g., background removal, image editing)
    - Structured outputs (JSON)
    
    Compatible with models like:
    - gemini-2.0-flash
    - gemini-2.5-flash
    - gemini-2.5-flash-image (for image manipulation)
    - gemini-pro
    - gemini-pro-vision
    """
    
    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        """Initialize Google Gemini interface.
        
        Args:
            connection_info: Configuration dictionary
            model_name: Model identifier (e.g., "gemini-2.5-flash")
        """
        self.model_name = model_name
        self.provider_type = connection_info.get("provider_type", "google")
        
        # HTTP/API configuration
        self.timeout = connection_info.get("timeout", 60)
        self.max_retries = connection_info.get("max_retries", 3)
        self.temperature = connection_info.get("temperature", 0.0)
        self.max_tokens = connection_info.get("max_tokens", 2048)
        self.retry_invalid_response = connection_info.get("retry_invalid_response", True)
        
        # Capabilities
        default_capabilities = InterfaceCapabilities(
            supports_multimodal=True,
            supports_logprobs=False,
            supports_schema=True,  # Gemini supports JSON mode
            supports_files=True,
            supports_streaming=False,
            supports_batch=True,
            request_modes=["chat"],  # Primary mode
        )
        self._capabilities = parse_interface_capabilities(
            connection_info.get("capabilities"),
            default=default_capabilities,
        )
        
        # Concurrency control
        is_cloud = True  # Google API is always cloud-based
        default_concurrent = connection_info.get("max_concurrent") or 3
        self._semaphore = asyncio.Semaphore(default_concurrent)
        
        # Get API key
        api_key = self._get_api_key(connection_info, "GOOGLE_API_KEY")
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"Initialized Google Gemini interface for {model_name}")
        logger.info(f"  Rate limit: max {default_concurrent} concurrent requests")
    
    def _get_api_key(self, connection_info: Dict[str, Any], default_env: str) -> str:
        """Get API key from config or environment."""
        api_key = connection_info.get("api_key")
        if not api_key:
            api_key_env = connection_info.get("api_key_env", default_env)
            api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {default_env} environment variable.\n"
                f"  - Add to .env file: {default_env}=your-key-here"
            )
        return api_key
    
    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request from task sample.
        
        Args:
            sample: Sample dict with prompts and optional image_path
            task: Task instance (provides get_prompt and answer_type)
        
        Returns:
            Request dict for generate_batch
        """
        answer_type = getattr(task, "answer_type", None)
        expects_image_artifact = answer_type == "image_artifact"
        
        # Get prompts from task
        system_prompt, user_prompt = task.get_prompt(sample)
        
        # Get schema if task expects structured output
        schema = sample.get("schema") if self._capabilities.supports_schema else None
        
        request = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
            "sample_id": sample["id"],
            "expects_image_artifact": expects_image_artifact,
        }
        
        # Add image if present (multimodal)
        if "image_path" in sample:
            if not self._capabilities.supports_multimodal:
                raise ValueError("Interface does not support multimodal inputs")
            request["image_path"] = sample["image_path"]
        
        return request
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text (handle markdown code blocks)."""
        text = text.strip()
        
        # Try to find JSON in markdown code blocks
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'```json(.*?)```',
            r'```(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                json_candidate = matches[0].strip()
                json_candidate = re.sub(r'^```.*?\n?', '', json_candidate)
                json_candidate = re.sub(r'\n?```$', '', json_candidate)
                return json_candidate.strip()
        
        # Try to find JSON braces
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            json_candidate = text[first_brace:last_brace + 1]
            if json_candidate.count("{") == json_candidate.count("}"):
                return json_candidate
        
        return text
    
    async def _generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        sample_id: str,
        image_path: Optional[str] = None,
        expects_image_artifact: bool = False,
    ) -> Dict:
        """Generate output for a single request.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            schema: Optional JSON schema for structured output
            sample_id: Sample identifier
            image_path: Optional path to input image (multimodal)
            expects_image_artifact: Whether task expects image output
        
        Returns:
            Result dict with output, raw, error, error_type
        """
        result = {"output": None, "raw": None, "error": None, "error_type": None}
        
        async def attempt_fn(_: int) -> Dict:
            nonlocal result
            result = await self._call_gemini_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                image_path=image_path,
                expects_image_artifact=expects_image_artifact,
            )
            
            # Retry if no output and retry_invalid_response is enabled
            if result.get("output") is None and self.retry_invalid_response:
                raise RetryableError(
                    result.get("error", "Invalid response"),
                    error_type=result.get("error_type", "invalid_response"),
                    retry_after=0.0,
                )
            return result
        
        response, error, error_type = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=classify_http_exception,
        )
        
        if response is not None:
            return response
        
        # Return error result
        result["output"] = None
        result["error"] = error
        result["error_type"] = error_type
        logger.warning(
            f"[{sample_id}] All {self.max_retries} attempts failed: {error} (type: {error_type})"
        )
        return result
    
    async def _call_gemini_api(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        image_path: Optional[str],
        expects_image_artifact: bool,
    ) -> Dict:
        """Call Google Gemini API.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            schema: Optional JSON schema
            image_path: Optional image path
            expects_image_artifact: Whether expecting image output
        
        Returns:
            Result dict with output, raw, error, error_type
        """
        result = {"output": None, "raw": None, "error": None, "error_type": None}
        
        try:
            # Prepare the prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = user_prompt
            
            # Add schema hint if structured output requested
            if schema and not expects_image_artifact:
                schema_str = json.dumps(schema, indent=2)
                full_prompt += f"\n\nPlease respond with valid JSON matching this schema:\n{schema_str}"
            
            # Prepare contents for API call
            contents = [full_prompt]
            
            # Add image if provided (multimodal)
            if image_path:
                image = Image.open(image_path)
                contents.append(image)
            
            # Prepare generation config
            config_params = {}
            if self.temperature is not None:
                config_params["temperature"] = self.temperature
            if self.max_tokens:
                config_params["max_output_tokens"] = self.max_tokens
            
            # Add response MIME type hint for JSON
            if schema and not expects_image_artifact:
                config_params["response_mime_type"] = "application/json"
            
            # Call Gemini API (synchronous call, but we're in async context)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**config_params) if config_params else None,
                )
            )
            
            # Process response based on expected output type
            if expects_image_artifact:
                # Extract image from response
                output_saved = False
                for part in response.parts:
                    if part.inline_data is not None:
                        # Get the image from the response
                        output_image = part.as_image()
                        
                        # Save to temp file to get base64
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        
                        try:
                            output_image.save(tmp_path)
                            with open(tmp_path, 'rb') as f:
                                image_bytes = f.read()
                            
                            # Encode as base64
                            base64_data = base64.b64encode(image_bytes).decode('utf-8')
                            result["output"] = base64_data
                            result["raw"] = json.dumps({
                                "model": self.model_name,
                                "base64_length": len(base64_data),
                                "format": "png"
                            }, ensure_ascii=False)
                            output_saved = True
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        break
                
                if not output_saved:
                    # Check for text response (might be an error)
                    text_parts = [part.text for part in response.parts if part.text is not None]
                    if text_parts:
                        error_msg = " ".join(text_parts)
                        result["error"] = f"No image generated. Response: {error_msg}"
                    else:
                        result["error"] = "No image was generated in the response."
                    result["error_type"] = "invalid_response"
                    result["raw"] = json.dumps({"model": self.model_name, "text_parts": text_parts}, ensure_ascii=False)
            else:
                # Extract text response
                text_parts = []
                for part in response.parts:
                    if part.text is not None:
                        text_parts.append(part.text)
                
                if not text_parts:
                    result["error"] = "No text content in response"
                    result["error_type"] = "invalid_response"
                    return result
                
                raw_text = "".join(text_parts)
                result["raw"] = raw_text
                
                # Parse JSON if schema was provided
                if schema:
                    try:
                        # Try direct JSON parse first (Gemini with response_mime_type)
                        result["output"] = json.loads(raw_text)
                    except json.JSONDecodeError:
                        # Fallback: extract JSON from markdown/text
                        cleaned = self._extract_json(raw_text)
                        try:
                            result["output"] = json.loads(cleaned)
                        except json.JSONDecodeError as e:
                            result["error"] = f"JSON parse error: {e}"
                            result["error_type"] = "invalid_response"
                else:
                    # Plain text output
                    result["output"] = raw_text.strip()
        
        except Exception as e:
            result["error"] = f"Google API error: {str(e)}"
            result["error_type"] = "api_error"
            result["raw"] = str(e)
            logger.error(f"Error calling Google Gemini API: {e}")
        
        return result
    
    async def _generate_with_limit(self, req: Dict) -> Dict:
        """Generate with concurrency limiting."""
        async with self._semaphore:
            return await self._generate_single(
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                schema=req.get("schema"),
                sample_id=req["sample_id"],
                image_path=req.get("image_path"),
                expects_image_artifact=req.get("expects_image_artifact", False),
            )
    
    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate outputs for a batch of samples.
        
        Args:
            requests: List of request dicts from prepare_request()
        
        Returns:
            List of result dictionaries in same order as requests
        """
        tasks = [self._generate_with_limit(req) for req in requests]
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
                    "error_type": "connectivity_error",
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
        """Test connection to Google Gemini API.
        
        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Testing Google Gemini API with model {self.model_name}")
        
        async def attempt_fn(_: int) -> bool:
            result = await self._call_gemini_api(
                system_prompt="",
                user_prompt="Hi, this is a connection test.",
                schema=None,
                image_path=None,
                expects_image_artifact=False,
            )
            if result.get("output") is not None:
                logger.info("Test connection successful - API is reachable")
                return True
            raise RetryableError(
                result.get("error", "Connection test failed"),
                error_type="connectivity_error",
                retry_after=0.0
            )
        
        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=classify_http_exception,
        )
        
        if result:
            logger.info(f"Connected to Google Gemini API with model {self.model_name}")
            return True
        
        logger.error(f"Failed to connect to Google Gemini after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False
    
    @property
    def supports_multimodal(self) -> bool:
        return self._capabilities.supports_multimodal
    
    @property
    def supports_logprobs(self) -> bool:
        return False  # Gemini doesn't support logprobs
    
    @property
    def capabilities(self) -> InterfaceCapabilities:
        return self._capabilities
