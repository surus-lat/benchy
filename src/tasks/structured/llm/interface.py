"""Interface for communicating with vLLM server."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class VLLMInterface:
    """Client for vLLM server using OpenAI-compatible API with async support."""

    def __init__(self, config: Dict, model_name: str):
        """Initialize the vLLM interface.

        Args:
            config: Configuration dictionary with model settings
            model_name: Name/identifier of the model to use
        """
        self.config = config["model"]
        self.model_name = model_name
        self.batch_size = config.get("performance", {}).get("batch_size", 20)
        
        # Use async client for better performance
        self.client = AsyncOpenAI(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
        )

    async def _generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        sample_id: str,
    ) -> Dict:
        """Generate structured output for a single sample (async).

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Target JSON schema (should already be sanitized)
            sample_id: Sample identifier for logging

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error'
        """
        result = {
            "output": None,
            "raw": None,
            "error": None,
        }

        for attempt in range(self.config["max_retries"]):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={
                        "guided_json": schema,
                    },
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    timeout=self.config["timeout"],
                )

                raw_output = response.choices[0].message.content
                result["raw"] = raw_output

                # Parse JSON output
                try:
                    result["output"] = json.loads(raw_output)
                except json.JSONDecodeError as e:
                    logger.debug(f"[{sample_id}] Failed to parse JSON output: {e}")
                    result["error"] = f"JSON parse error: {e}"

                return result

            except Exception as e:
                logger.debug(f"[{sample_id}] Attempt {attempt + 1} failed: {e}")
                if attempt == self.config["max_retries"] - 1:
                    logger.warning(f"[{sample_id}] All {self.config['max_retries']} attempts failed: {e}")
                    result["error"] = str(e)
                    return result
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return result

    async def generate_batch(
        self,
        requests: List[Dict],
    ) -> List[Dict]:
        """Generate structured outputs for a batch of samples (async).

        Args:
            requests: List of request dicts with keys:
                - system_prompt: System message
                - user_prompt: User message
                - schema: Target JSON schema (should already be sanitized)
                - sample_id: Sample identifier

        Returns:
            List of result dictionaries in same order as requests
        """
        tasks = [
            self._generate_single(
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                schema=req["schema"],
                sample_id=req["sample_id"],
            )
            for req in requests
        ]
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed with exception: {result}")
                processed_results.append({
                    "output": None,
                    "raw": None,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        
        return processed_results

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to vLLM server with retries.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Testing connection to vLLM server (attempt {attempt + 1}/{max_retries})...")
                
                # Set a reasonable timeout for the connection test
                models = await asyncio.wait_for(
                    self.client.models.list(),
                    timeout=timeout
                )
                
                logger.info(f"✓ Successfully connected to vLLM server at {self.config['base_url']}")
                logger.info(f"✓ Available models: {[m.id for m in models.data]}")
                
                # Verify our model is available
                available_model_ids = [m.id for m in models.data]
                if self.model_name not in available_model_ids:
                    logger.warning(f"⚠️  Model '{self.model_name}' not found in available models")
                    logger.warning(f"Available models: {available_model_ids}")
                    logger.warning("Continuing anyway - model name might still work...")
                else:
                    logger.info(f"✓ Model '{self.model_name}' is available")
                
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"Connection attempt {attempt + 1} timed out after {timeout}s")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        
        logger.error(f"✗ Failed to connect to vLLM server after {max_retries} attempts")
        logger.error(f"Please check that:")
        logger.error(f"  1. vLLM server is running at {self.config['base_url']}")
        logger.error(f"  2. The server is accessible from this machine")
        logger.error(f"  3. The API key (if required) is correct")
        return False

