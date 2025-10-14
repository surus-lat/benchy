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
        
        # Track if this is a known problematic model for cleaner logging
        known_chat_template_issues = [
            "ByteDance-Seed/Seed-X-Instruct-7B",
            "ByteDance-Seed/Seed-X-PPO-7B"
        ]
        self.is_problematic_model = any(model in self.model_name for model in known_chat_template_issues)

    def _extract_json_from_output(self, raw_output: str) -> str:
        """Extract JSON from model output, handling various formats.
        
        Args:
            raw_output: Raw text output from the model
            
        Returns:
            Cleaned JSON string ready for parsing
        """
        import re
        
        # Remove leading/trailing whitespace
        text = raw_output.strip()
        
        # Try to find JSON within markdown code blocks
        # Look for ```json ... ``` or ``` ... ``` patterns
        json_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',      # ``` ... ```
            r'```json(.*?)```',         # ```json...``` (no newlines)
            r'```(.*?)```',             # ```...``` (no newlines)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Take the first match and clean it
                json_candidate = matches[0].strip()
                # Remove any remaining markdown artifacts
                json_candidate = re.sub(r'^```.*?\n?', '', json_candidate)
                json_candidate = re.sub(r'\n?```$', '', json_candidate)
                return json_candidate.strip()
        
        # If no code blocks found, look for JSON-like content
        # Find the first { and last } that might contain JSON
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = text[first_brace:last_brace + 1]
            # Basic validation - check if it looks like JSON
            if json_candidate.count('{') == json_candidate.count('}'):
                return json_candidate
        
        # If all else fails, return the original text
        return text

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

        # Use the flag set in __init__
        use_completions_fallback = self.is_problematic_model
        
        for attempt in range(self.config["max_retries"]):
            try:
                # For models with known chat template issues, try completions API first
                if use_completions_fallback and attempt == 0:
                    logger.info(f"[{sample_id}] Using completions API fallback for model with known chat template issues")
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                    
                    response = await self.client.completions.create(
                        model=self.model_name,
                        prompt=combined_prompt,
                        temperature=self.config["temperature"],
                        max_tokens=self.config["max_tokens"],
                        timeout=self.config["timeout"],
                    )

                    raw_output = response.choices[0].text
                    result["raw"] = raw_output

                    # Parse JSON output with filtering
                    try:
                        # Clean the output by removing markdown code blocks and extracting JSON
                        cleaned_output = self._extract_json_from_output(raw_output)
                        result["output"] = json.loads(cleaned_output)
                        logger.info(f"[{sample_id}] ✓ Completions API successful - JSON parsed")
                    except json.JSONDecodeError as e:
                        logger.warning(f"[{sample_id}] ⚠️ Completions API successful but JSON parse failed: {e}")
                        logger.info(f"[{sample_id}] Raw output (first 200 chars): {raw_output[:200]}...")
                        result["error"] = f"JSON parse error: {e}"

                    return result
                else:
                    # Try chat completions (preferred method with guided JSON)
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

                    # Parse JSON output with filtering
                    try:
                        # Clean the output by removing markdown code blocks and extracting JSON
                        cleaned_output = self._extract_json_from_output(raw_output)
                        result["output"] = json.loads(cleaned_output)
                    except json.JSONDecodeError as e:
                        logger.debug(f"[{sample_id}] Failed to parse JSON output: {e}")
                        result["error"] = f"JSON parse error: {e}"

                    return result

            except Exception as e:
                error_str = str(e)
                logger.debug(f"[{sample_id}] Attempt {attempt + 1} failed: {e}")
                
                # Check if this is a chat template error or HTTP 400 error - try completions API as fallback
                is_chat_template_error = (
                    "chat template" in error_str.lower() and "not allowed" in error_str.lower()
                ) or (
                    "400" in error_str and "bad request" in error_str.lower()
                ) or (
                    hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400
                )
                
                if is_chat_template_error:
                    logger.info(f"[{sample_id}] Chat template error detected, trying completions API fallback...")
                    try:
                        # Fallback to completions API (no chat template required)
                        # Combine system and user prompts into a single prompt
                        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                        
                        response = await self.client.completions.create(
                            model=self.model_name,
                            prompt=combined_prompt,
                            temperature=self.config["temperature"],
                            max_tokens=self.config["max_tokens"],
                            timeout=self.config["timeout"],
                        )

                        raw_output = response.choices[0].text
                        result["raw"] = raw_output

                        # Parse JSON output with filtering
                        try:
                            # Clean the output by removing markdown code blocks and extracting JSON
                            cleaned_output = self._extract_json_from_output(raw_output)
                            result["output"] = json.loads(cleaned_output)
                            logger.info(f"[{sample_id}] ✓ Fallback to completions API successful - JSON parsed")
                        except json.JSONDecodeError as e:
                            logger.warning(f"[{sample_id}] ⚠️ Fallback successful but JSON parse failed: {e}")
                            logger.info(f"[{sample_id}] Raw output (first 200 chars): {raw_output[:200]}...")
                            result["error"] = f"JSON parse error (fallback): {e}"

                        return result
                        
                    except Exception as fallback_e:
                        logger.warning(f"[{sample_id}] Fallback to completions API also failed: {fallback_e}")
                        # Continue to normal retry logic below
                
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
        
        # Handle any exceptions that occurred and provide summary
        processed_results = []
        successful = 0
        json_parse_errors = 0
        other_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed with exception: {result}")
                processed_results.append({
                    "output": None,
                    "raw": None,
                    "error": str(result),
                })
                other_errors += 1
            else:
                processed_results.append(result)
                if result.get("output") is not None:
                    successful += 1
                elif result.get("error") and "JSON parse error" in result.get("error", ""):
                    json_parse_errors += 1
                else:
                    other_errors += 1
        
        # Log batch summary
        total = len(requests)
        logger.info(f"📊 Batch completed: {successful}/{total} successful, {json_parse_errors} JSON parse errors, {other_errors} other errors")
        
        return processed_results

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to vLLM server with retries.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        # Log model-specific information
        if self.is_problematic_model:
            logger.info(f"🔧 Using completions API fallback for model: {self.model_name}")
        else:
            logger.info(f"🚀 Using chat completions API for model: {self.model_name}")
            
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

