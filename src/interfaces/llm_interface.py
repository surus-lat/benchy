"""Interface for LLM providers (vLLM, OpenAI, Anthropic)."""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMInterface:
    """Async client for LLM providers using OpenAI-compatible API."""

    def __init__(self, config: Dict, model_name: str, provider_type: str = "vllm"):
        """Initialize the LLM interface.

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
            from anthropic import AsyncAnthropic
            api_key = self._get_api_key("ANTHROPIC_API_KEY")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)
            self.client = None
            logger.info("Initialized Anthropic client")
        else:
            # OpenAI-compatible client (for vLLM and OpenAI)
            api_key = self._get_api_key("OPENAI_API_KEY") if provider_type == "openai" else "EMPTY"
            self.client = AsyncOpenAI(
                base_url=self.config["base_url"],
                api_key=api_key,
            )
            self.anthropic_client = None
            logger.info(f"Initialized OpenAI-compatible client for {provider_type}")
        
        # Track problematic models for fallback logic
        known_chat_template_issues = [
            "ByteDance-Seed/Seed-X-Instruct-7B",
            "ByteDance-Seed/Seed-X-PPO-7B"
        ]
        self.is_problematic_model = any(model in self.model_name for model in known_chat_template_issues)
        
        # Get API endpoint preference (default: "auto")
        self.api_endpoint = self.config.get("api_endpoint", "auto")
        
        # Rate limiting for cloud providers (avoid 429 errors)
        # vLLM can handle high concurrency, cloud APIs cannot
        if provider_type in ["openai", "anthropic"]:
            max_concurrent = self.config.get("max_concurrent", 3)
            self._semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"Rate limiting enabled: max {max_concurrent} concurrent requests")
        else:
            self._semaphore = None

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for LLM provider.
        
        LLM interfaces need system and user prompts from the task.
        
        Args:
            sample: Raw sample with text, schema, expected, etc.
            task: Task instance that provides get_prompt() method
            
        Returns:
            Dict formatted for this interface's generate_batch()
        """
        system_prompt, user_prompt = task.get_prompt(sample)
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": sample.get("schema"),  # May be None for non-structured tasks
            "sample_id": sample["id"],
        }
    
    def _get_api_key(self, env_var: str) -> str:
        """Get API key from config or environment.
        
        Args:
            env_var: Environment variable name for the API key
            
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not found or invalid
        """
        config_key = self.config.get("api_key")
        if config_key and config_key != "EMPTY":
            api_key = config_key
        else:
            api_key_env_var = self.config.get("api_key_env", env_var)
            api_key = os.getenv(api_key_env_var)
        
        if not api_key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable.\n"
                f"  - Add to .env file: {env_var}=your-key-here\n"
                f"  - Or export: export {env_var}='your-key-here'"
            )
        
        if len(api_key) < 20:
            raise ValueError(f"API key appears invalid (too short): {api_key[:10]}...")
        
        return api_key

    def _extract_json_from_output(self, raw_output: str) -> str:
        """Extract JSON from model output, handling markdown code blocks.
        
        Args:
            raw_output: Raw text output from the model
            
        Returns:
            Cleaned JSON string ready for parsing
        """
        text = raw_output.strip()
        
        # Try to find JSON within markdown code blocks
        json_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'```json(.*?)```',
            r'```(.*?)```',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                json_candidate = matches[0].strip()
                json_candidate = re.sub(r'^```.*?\n?', '', json_candidate)
                json_candidate = re.sub(r'\n?```$', '', json_candidate)
                return json_candidate.strip()
        
        # Look for JSON-like content (first { to last })
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = text[first_brace:last_brace + 1]
            if json_candidate.count('{') == json_candidate.count('}'):
                return json_candidate
        
        return text

    async def _generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        sample_id: str,
    ) -> Dict:
        """Generate structured output for a single sample.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Target JSON schema
            sample_id: Sample identifier for logging

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error'
        """
        result = {"output": None, "raw": None, "error": None}

        # Handle Anthropic provider separately
        if self.provider_type == "anthropic":
            return await self._generate_single_anthropic(system_prompt, user_prompt, schema, sample_id)

        # Determine which API to use
        use_completions_api = (
            self.api_endpoint == "completions" or 
            (self.api_endpoint == "auto" and self.is_problematic_model)
        )
        force_chat_only = self.api_endpoint == "chat"
        
        for attempt in range(self.config["max_retries"]):
            # Try completions API if configured or for problematic models
            if use_completions_api and attempt == 0:
                if self.api_endpoint == "completions":
                    logger.debug(f"[{sample_id}] Using completions API (configured)")
                else:
                    logger.info(f"[{sample_id}] Using completions API for model with chat template issues")
                
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

                # For translation tasks (no schema), output is just the text
                if schema:
                    cleaned_output = self._extract_json_from_output(raw_output)
                    try:
                        result["output"] = json.loads(cleaned_output)
                        logger.debug(f"[{sample_id}] ✓ Completions API successful")
                    except json.JSONDecodeError as e:
                        logger.warning(f"[{sample_id}] ⚠️ JSON parse failed: {e}")
                        result["error"] = f"JSON parse error: {e}"
                else:
                    # No schema - output is plain text (e.g., translation)
                    result["output"] = raw_output.strip()
                    logger.debug(f"[{sample_id}] ✓ Completions API successful (text output)")

                return result
            
            # Try chat completions API
            chat_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "timeout": self.config["timeout"],
            }
            
            # Handle temperature (some models don't support it)
            if not (self.provider_type == "openai" and any(m in self.model_name.lower() for m in ["gpt-5", "o1", "o3", "o4"])):
                chat_params["temperature"] = self.config["temperature"]
            
            # Handle max_tokens vs max_completion_tokens
            if self.provider_type == "openai" and any(m in self.model_name.lower() for m in ["gpt-5", "o1", "o3", "o4"]):
                chat_params["max_completion_tokens"] = self.config["max_tokens"]
            else:
                chat_params["max_tokens"] = self.config["max_tokens"]
            
            # Only add structured_outputs for vLLM (v0.12.0+ API) if schema is provided
            if self.provider_type == "vllm" and schema:
                chat_params["extra_body"] = {"structured_outputs": {"json": schema}}
            
            response = await self.client.chat.completions.create(**chat_params)

            raw_output = response.choices[0].message.content
            result["raw"] = raw_output

            cleaned_output = self._extract_json_from_output(raw_output)
            try:
                result["output"] = json.loads(cleaned_output)
            except json.JSONDecodeError as e:
                logger.debug(f"[{sample_id}] Failed to parse JSON: {e}")
                result["error"] = f"JSON parse error: {e}"

            return result

        # All retries failed
        logger.warning(f"[{sample_id}] All {self.config['max_retries']} attempts failed")
        result["error"] = "All retry attempts exhausted"
        return result
    
    async def _generate_single_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        sample_id: str,
    ) -> Dict:
        """Generate structured output using Anthropic API.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Target JSON schema (for prompt guidance)
            sample_id: Sample identifier for logging

        Returns:
            Dictionary with 'output' (parsed JSON), 'raw' (string), 'error'
        """
        result = {"output": None, "raw": None, "error": None}
        
        # Add JSON schema to prompt if provided (Anthropic doesn't support structured_outputs)
        if schema:
            schema_str = json.dumps(schema, indent=2)
            enhanced_user_prompt = f"{user_prompt}\n\nPlease respond with valid JSON matching this schema:\n{schema_str}"
        else:
            enhanced_user_prompt = user_prompt
        
        for attempt in range(self.config["max_retries"]):
            response = await self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                system=system_prompt,
                messages=[{"role": "user", "content": enhanced_user_prompt}]
            )
            
            raw_output = response.content[0].text
            result["raw"] = raw_output
            
            # For translation tasks (no schema), output is just the text
            if schema:
                cleaned_output = self._extract_json_from_output(raw_output)
                try:
                    result["output"] = json.loads(cleaned_output)
                except json.JSONDecodeError as e:
                    logger.debug(f"[{sample_id}] Failed to parse JSON: {e}")
                    result["error"] = f"JSON parse error: {e}"
            else:
                # No schema - output is plain text (e.g., translation)
                result["output"] = raw_output.strip()
            
            return result
        
        logger.warning(f"[{sample_id}] All {self.config['max_retries']} attempts failed")
        result["error"] = "All retry attempts exhausted"
        return result

    async def _generate_with_limit(self, req: Dict) -> Dict:
        """Generate with optional rate limiting."""
        if self._semaphore:
            async with self._semaphore:
                return await self._generate_single(
                    system_prompt=req["system_prompt"],
                    user_prompt=req["user_prompt"],
                    schema=req["schema"],
                    sample_id=req["sample_id"],
                )
        else:
            return await self._generate_single(
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                schema=req["schema"],
                sample_id=req["sample_id"],
            )

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate structured outputs for a batch of samples.

        Args:
            requests: List of request dicts with keys:
                - system_prompt: System message
                - user_prompt: User message
                - schema: Target JSON schema
                - sample_id: Sample identifier

        Returns:
            List of result dictionaries in same order as requests
        """
        tasks = [self._generate_with_limit(req) for req in requests]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        successful = 0
        json_parse_errors = 0
        other_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed: {result}")
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
        
        total = len(requests)
        logger.info(f"📊 Batch: {successful}/{total} successful, {json_parse_errors} JSON errors, {other_errors} other errors")
        
        return processed_results

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to LLM provider.

        Args:
            max_retries: Maximum number of connection attempts
            timeout: Timeout per attempt in seconds

        Returns:
            True if connection successful, False otherwise
        """
        # Handle Anthropic provider
        if self.provider_type == "anthropic":
            logger.info(f"🚀 Testing Anthropic API for model: {self.model_name}")
            for attempt in range(max_retries):
                response = await asyncio.wait_for(
                    self.anthropic_client.messages.create(
                        model=self.model_name,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Hi"}]
                    ),
                    timeout=timeout
                )
                logger.info(f"✓ Connected to Anthropic API, model '{self.model_name}' available")
                return True
            
            logger.error(f"✗ Failed to connect to Anthropic API after {max_retries} attempts")
            return False
        
        # Handle OpenAI-compatible providers (vLLM, OpenAI)
        if self.is_problematic_model:
            logger.info(f"🔧 Using completions API for model: {self.model_name}")
        else:
            logger.info(f"🚀 Using chat completions API for model: {self.model_name}")
            
        for attempt in range(max_retries):
            logger.info(f"Testing connection to {self.provider_type} (attempt {attempt + 1}/{max_retries})...")
            
            models = await asyncio.wait_for(
                self.client.models.list(),
                timeout=timeout
            )
            
            logger.info(f"✓ Connected to {self.provider_type} at {self.config['base_url']}")
            logger.info(f"✓ Available models: {[m.id for m in models.data]}")
            
            available_model_ids = [m.id for m in models.data]
            if self.model_name not in available_model_ids:
                logger.warning(f"⚠️ Model '{self.model_name}' not in available models")
                logger.warning("Continuing anyway - model name might still work...")
            else:
                logger.info(f"✓ Model '{self.model_name}' is available")
            
            return True
        
        logger.error(f"✗ Failed to connect to {self.provider_type} after {max_retries} attempts")
        return False

