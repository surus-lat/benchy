"""Google Gemini interface for image manipulation (remove background)."""

import base64
import json
import logging
import os
import tempfile
from typing import Dict, Optional, Any, List
import asyncio

from google import genai
from PIL import Image

from ..http_interface import HTTPInterface
from ...engine.protocols import InterfaceCapabilities
from ...engine.retry import classify_http_exception, run_with_retries, RetryableError
from ..common.image_preprocessing import coerce_positive_int, load_pil_image

logger = logging.getLogger(__name__)


class GoogleRemoveBackgroundInterface(HTTPInterface):
    """Interface for Google Gemini image manipulation models.
    
    This interface supports models like gemini-2.5-flash-image that can
    perform image editing tasks such as background removal.
    
    The interface expects samples with:
      - image_path: Path to input image
      - text: Optional prompt for the task
    
    And returns base64-encoded PNG images with transparent backgrounds.
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
        self, config: Dict, model_name: str, provider_type: str = "google"
    ):
        """Initialize Google interface.
        
        Args:
            config: Configuration dictionary
            model_name: Model identifier (e.g., "gemini-2.5-flash-image")
            provider_type: Provider type identifier
        """
        # Initialize parent (HTTPInterface) but we'll override most of its behavior
        # We still need it for the config parsing and API key handling
        self.config = config.get(provider_type, {})
        self.model_name = model_name
        self.provider_type = provider_type
        self.capabilities = InterfaceCapabilities(
            supports_multimodal=True,
            supports_logprobs=False,
            supports_schema=False,
            supports_files=True,
            supports_streaming=False,
            request_modes=["raw_payload"],
        )
        
        # Get API key
        api_key_env = self.config.get("api_key_env", "GOOGLE_API_KEY")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Set {api_key_env} environment variable.\n"
                f"  - Add to .env file: {api_key_env}=your-key-here"
            )
        
        # HTTP configuration (for compatibility)
        self.endpoint = self.config.get("endpoint", "https://generativelanguage.googleapis.com/v1")
        self.timeout = self.config.get("timeout", 60)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_invalid_response = self.config.get("retry_invalid_response", True)
        self.retry_on_4xx = self.config.get("retry_on_4xx", False)
        self.image_max_edge = coerce_positive_int(
            self.config.get("image_max_edge"),
            option_name="image_max_edge",
            logger=logger,
        )
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"Initialized Google Gemini interface for {model_name}")
        if isinstance(self.image_max_edge, int):
            logger.info(f"  Image scaling enabled: max edge={self.image_max_edge}px")
    
    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for Google Gemini.
        
        Args:
            sample: Sample dict with image_path and text (prompt)
            task: Task instance (provides get_prompt)
        
        Returns:
            Request dict with image and prompt
        """
        image_path = sample.get("image_path")
        if not image_path:
            raise ValueError("GoogleRemoveBackgroundInterface requires sample.image_path")
        
        # Get prompt from task
        system_prompt, user_prompt = task.get_prompt(sample)
        prompt = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        
        # Use a default prompt if none provided
        if not prompt or not prompt.strip():
            prompt = "Remove the background from this image and return only the subject with a transparent background."
        
        return {
            "image_path": image_path,
            "prompt": prompt,
            "sample_id": sample["id"],
        }
    
    def build_test_request(self) -> Dict[str, Any]:
        """Build a minimal request payload for connection tests."""
        # For testing, we need a minimal image
        # Create a simple 1x1 image in memory
        return {
            "image_path": None,  # Will use a synthetic image
            "prompt": "Test connection",
            "sample_id": "connection_test",
        }
    
    async def _generate_single(self, request: Dict) -> Dict:
        """Generate output for a single sample using Google Gemini.
        
        Args:
            request: Request dict from prepare_request()
        
        Returns:
            Dictionary with 'output' (base64 PNG), 'raw', 'error', 'error_type'
        """
        sample_id = request.get("sample_id", "unknown")
        last_result: Optional[Dict[str, Any]] = None
        
        async def attempt_fn(_: int) -> Dict:
            nonlocal last_result
            result = await self._call_gemini_api(request)
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
            classify_error=classify_http_exception,
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
    
    async def _call_gemini_api(self, request: Dict) -> Dict:
        """Call Google Gemini API to process the image.
        
        Args:
            request: Request dict with image_path and prompt
        
        Returns:
            Result dict with output (base64 PNG), raw, error, error_type
        """
        result = {"output": None, "raw": None, "error": None, "error_type": None}
        
        try:
            # Load the image
            image_path = request.get("image_path")
            input_image = None
            if image_path:
                input_image = load_pil_image(
                    image_path,
                    max_edge=self.image_max_edge,
                    logger=logger,
                )
            else:
                # Create a minimal test image
                input_image = Image.new('RGB', (1, 1), color='white')
            
            prompt = request.get("prompt", "Remove the background from this image.")
            
            # Call Gemini API (synchronous call, but we're in async context)
            # We'll run it in an executor to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, input_image],
                )
            )
            
            # Process the response and extract the image
            output_saved = False
            for part in response.parts:
                if part.inline_data is not None:
                    # Get the image from the response
                    output_image = part.as_image()
                    
                    # The Google GenAI Image.save() method only accepts a file path
                    # So we need to save to a temp file first, then read it back
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    try:
                        # Save to temp file
                        output_image.save(tmp_path)
                        
                        # Read back as bytes
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
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    
                    break
            
            if not output_saved:
                # Check if there's text in the response (might be an error message)
                text_parts = [part.text for part in response.parts if part.text is not None]
                if text_parts:
                    error_msg = " ".join(text_parts)
                    result["error"] = f"No image generated. Response: {error_msg}"
                else:
                    result["error"] = "No image was generated in the response."
                result["error_type"] = "invalid_response"
                result["raw"] = json.dumps({
                    "model": self.model_name,
                    "text_parts": text_parts
                }, ensure_ascii=False)
        
        except Exception as e:
            result["error"] = f"Google API error: {str(e)}"
            result["error_type"] = "api_error"
            result["raw"] = str(e)
            logger.error(f"Error calling Google Gemini API: {e}")
        finally:
            if "input_image" in locals() and input_image is not None:
                try:
                    input_image.close()
                except Exception:
                    pass
        
        return result
    
    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate outputs for a batch of samples.
        
        Args:
            requests: List of request dicts from prepare_request()
        
        Returns:
            List of result dictionaries in same order as requests
        """
        tasks = [self._generate_single(request=req) for req in requests]
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
        
        test_request = self.build_test_request()
        
        async def attempt_fn(_: int) -> bool:
            result = await self._call_gemini_api(test_request)
            if result.get("output") is not None or result.get("error_type") == "invalid_response":
                # Connection is working even if the minimal test doesn't produce an image
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
    
    async def _make_request_with_client(self, client, request: Dict) -> Optional[Dict]:
        """Not used - we override generate_batch directly."""
        raise NotImplementedError("GoogleRemoveBackgroundInterface uses direct API calls")
    
    def _parse_response(self, response: Dict) -> Dict:
        """Not used - we override generate_batch directly."""
        raise NotImplementedError("GoogleRemoveBackgroundInterface uses direct API calls")
