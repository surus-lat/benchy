"""Interface for communicating with vLLM server and cloud providers."""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class VLLMInterface:
    """Client for vLLM server and cloud providers using OpenAI-compatible API with async support."""

    def __init__(self, config: Dict, model_name: str, provider_type: str = "vllm"):
        """Initialize the interface.

        Args:
            config: Configuration dictionary with model settings
            model_name: Name/identifier of the model to use
            provider_type: Type of provider ('vllm', 'openai', or 'anthropic')
        """
        self.config = config["model"]
        self.model_name = model_name
        self.batch_size = config.get("performance", {}).get("batch_size", 20)
        self.provider_type = provider_type
        
        # Initialize appropriate client based on provider type
        if provider_type == "anthropic":
            # Import Anthropic client dynamically
            try:
                from anthropic import AsyncAnthropic
                # Get API key from config or environment (handle "EMPTY" as None)
                config_key = self.config.get("api_key")
                if config_key == "EMPTY" or not config_key:
                    config_key = None
                api_key = config_key or os.getenv(self.config.get("api_key_env", "ANTHROPIC_API_KEY"))
                if not api_key:
                    raise ValueError(
                        "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.\n"
                        "  - Add to .env file: ANTHROPIC_API_KEY=sk-ant-your-key\n"
                        "  - Or export: export ANTHROPIC_API_KEY='sk-ant-your-key'\n"
                        "  - Get your key at: https://console.anthropic.com/settings/keys"
                    )
                if api_key == "sk-ant-your-anthropic-api-key-here" or len(api_key) < 20:
                    raise ValueError(
                        "Anthropic API key appears to be invalid (placeholder or too short).\n"
                        "  - Replace the placeholder in your .env file with your actual key\n"
                        "  - Get your key at: https://console.anthropic.com/settings/keys\n"
                        "  - Key should start with: sk-ant-api..."
                    )
                self.anthropic_client = AsyncAnthropic(api_key=api_key)
                self.client = None
                logger.info("Initialized Anthropic client")
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        else:
            # OpenAI-compatible client (for vLLM and OpenAI)
            api_key = self.config.get("api_key", "EMPTY")
            if provider_type == "openai":
                # Get API key from environment for OpenAI (handle "EMPTY" and None)
                config_key = self.config.get("api_key")
                if config_key == "EMPTY" or not config_key:
                    config_key = None
                api_key = config_key or os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment variable.\n"
                        "  - Add to .env file: OPENAI_API_KEY=sk-proj-your-key\n"
                        "  - Or export: export OPENAI_API_KEY='sk-proj-your-key'\n"
                        "  - Get your key at: https://platform.openai.com/api-keys"
                    )
                if api_key == "sk-your-openai-api-key-here" or len(api_key) < 20:
                    raise ValueError(
                        "OpenAI API key appears to be invalid (placeholder or too short).\n"
                        "  - Replace the placeholder in your .env file with your actual key\n"
                        "  - Get your key at: https://platform.openai.com/api-keys\n"
                        "  - Key should start with: sk-proj-... or sk-..."
                    )
            
            self.client = AsyncOpenAI(
                base_url=self.config["base_url"],
                api_key=api_key,
            )
            self.anthropic_client = None
            logger.info(f"Initialized OpenAI-compatible client for {provider_type}")
        
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

        # Handle Anthropic provider separately
        if self.provider_type == "anthropic":
            return await self._generate_single_anthropic(system_prompt, user_prompt, schema, sample_id)

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
                    # Try chat completions (with guided JSON for vLLM only)
                    chat_params = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.config["temperature"],
                        "timeout": self.config["timeout"],
                    }
                    
                    # Handle max_tokens vs max_completion_tokens (GPT-5+ uses new parameter)
                    # Try new parameter first, fall back to old if rejected
                    if self.provider_type == "openai" and any(model in self.model_name.lower() for model in ["gpt-5", "o1", "o3", "o4"]):
                        # Newer models use max_completion_tokens
                        chat_params["max_completion_tokens"] = self.config["max_tokens"]
                    else:
                        # Older models and vLLM use max_tokens
                        chat_params["max_tokens"] = self.config["max_tokens"]
                    
                    # Only add guided_json for vLLM (not OpenAI)
                    if self.provider_type == "vllm":
                        chat_params["extra_body"] = {"guided_json": schema}
                    
                    response = await self.client.chat.completions.create(**chat_params)

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
                
                # Handle max_tokens parameter error for GPT-5+ models
                if "max_tokens" in error_str.lower() and "max_completion_tokens" in error_str.lower():
                    logger.info(f"[{sample_id}] Detected max_tokens/max_completion_tokens parameter mismatch, retrying with correct parameter...")
                    # This will be retried with the other parameter format
                    if attempt < self.config["max_retries"] - 1:
                        continue
                
                # Check if this is a chat template error - try completions API as fallback (vLLM only)
                # Don't try completions fallback for OpenAI models (they're chat-only)
                is_chat_template_error = (
                    "chat template" in error_str.lower() and "not allowed" in error_str.lower()
                ) or (
                    "400" in error_str and "bad request" in error_str.lower() and self.provider_type == "vllm"
                )
                
                if is_chat_template_error and self.provider_type == "vllm":
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
    
    async def _generate_single_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        sample_id: str,
    ) -> Dict:
        """Generate structured output for a single sample using Anthropic API.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Target JSON schema (for prompt guidance, not enforced)
            sample_id: Sample identifier for logging

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error'
        """
        result = {
            "output": None,
            "raw": None,
            "error": None,
        }
        
        # Add JSON schema to the user prompt since Anthropic doesn't support guided_json
        schema_str = json.dumps(schema, indent=2)
        enhanced_user_prompt = f"{user_prompt}\n\nPlease respond with valid JSON matching this schema:\n{schema_str}"
        
        for attempt in range(self.config["max_retries"]):
            try:
                response = await self.anthropic_client.messages.create(
                    model=self.model_name,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": enhanced_user_prompt}
                    ]
                )
                
                # Extract text from response
                raw_output = response.content[0].text
                result["raw"] = raw_output
                
                # Parse JSON output
                try:
                    cleaned_output = self._extract_json_from_output(raw_output)
                    result["output"] = json.loads(cleaned_output)
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
        """Test connection to server with retries.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        # Handle Anthropic provider
        if self.provider_type == "anthropic":
            logger.info(f"🚀 Testing connection to Anthropic API for model: {self.model_name}")
            for attempt in range(max_retries):
                try:
                    # Test with a simple message
                    response = await asyncio.wait_for(
                        self.anthropic_client.messages.create(
                            model=self.model_name,
                            max_tokens=10,
                            messages=[{"role": "user", "content": "Hi"}]
                        ),
                        timeout=timeout
                    )
                    logger.info(f"✓ Successfully connected to Anthropic API")
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
            
            logger.error(f"✗ Failed to connect to Anthropic API after {max_retries} attempts")
            return False
        
        # Handle OpenAI-compatible providers (vLLM, OpenAI)
        # Log model-specific information
        if self.is_problematic_model:
            logger.info(f"🔧 Using completions API fallback for model: {self.model_name}")
        else:
            logger.info(f"🚀 Using chat completions API for model: {self.model_name}")
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Testing connection to {self.provider_type} server (attempt {attempt + 1}/{max_retries})...")
                
                # Set a reasonable timeout for the connection test
                models = await asyncio.wait_for(
                    self.client.models.list(),
                    timeout=timeout
                )
                
                logger.info(f"✓ Successfully connected to {self.provider_type} server at {self.config['base_url']}")
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
        
        logger.error(f"✗ Failed to connect to {self.provider_type} server after {max_retries} attempts")
        logger.error(f"Please check that:")
        logger.error(f"  1. {self.provider_type} server is running at {self.config['base_url']}")
        logger.error(f"  2. The server is accessible from this machine")
        logger.error(f"  3. The API key (if required) is correct")
        return False

