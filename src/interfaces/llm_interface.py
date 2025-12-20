"""Interface for LLM providers (vLLM, OpenAI, Anthropic)."""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Suppress harmless httpx cleanup errors when event loop is closed
# These occur when httpx tries to clean up connections in background tasks
# after the event loop has been closed - they're non-fatal
def _suppress_event_loop_closed_error(loop, context):
    """Suppress 'Event loop is closed' errors from httpx cleanup tasks."""
    exception = context.get('exception')
    if isinstance(exception, RuntimeError) and 'Event loop is closed' in str(exception):
        # This is a harmless cleanup error, suppress it
        return
    # For other exceptions, use default handler
    if hasattr(loop, 'default_exception_handler'):
        loop.default_exception_handler(context)
    else:
        # Fallback: just log to stderr
        import sys
        print(f"Exception in async task: {context}", file=sys.stderr)

# Set up the exception handler to suppress these errors
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is already running, we'll set it up when the interface is created
        pass
    else:
        loop.set_exception_handler(_suppress_event_loop_closed_error)
except RuntimeError:
    # No event loop exists yet, will be set up later
    pass


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
            
            # Set up exception handler to suppress harmless httpx cleanup errors
            try:
                loop = asyncio.get_running_loop()
                loop.set_exception_handler(_suppress_event_loop_closed_error)
            except RuntimeError:
                # No running loop, will be set up when loop starts
                pass
        
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
        # Check if this is a multiple choice task that should use logprobs
        answer_type = getattr(task, "answer_type", None)
        use_logprobs = answer_type == "multiple_choice"
        requires_logprobs = getattr(task, "requires_logprobs", use_logprobs)
        use_logprobs = use_logprobs and requires_logprobs and sample.get("choices") is not None
        
        if use_logprobs and hasattr(task, "get_prompt_for_logprobs"):
            system_prompt, user_prompt = task.get_prompt_for_logprobs(sample)
        else:
            system_prompt, user_prompt = task.get_prompt(sample)
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": sample.get("schema"),  # May be None for non-structured tasks
            "sample_id": sample["id"],
            "use_logprobs": use_logprobs,
            "choices": sample.get("choices") if use_logprobs else None,
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
        use_logprobs: bool = False,
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Generate structured output for a single sample.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Target JSON schema
            sample_id: Sample identifier for logging
            use_logprobs: If True, use logprobs for multiple choice (choices must be provided)
            choices: List of choice strings for logprobs evaluation

        Returns:
            Dictionary with 'output' (parsed JSON or choice index), 'raw' (string), 'error'
        """
        result = {"output": None, "raw": None, "error": None}

        # Handle multiple choice with logprobs
        if use_logprobs and choices:
            return await self._generate_with_logprobs(system_prompt, user_prompt, choices, sample_id)

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
    
    async def _generate_with_logprobs(
        self,
        system_prompt: str,
        user_prompt: str,
        choices: List[str],
        sample_id: str,
    ) -> Dict:
        """Generate prediction using log probabilities for multiple choice.
        
        For each choice, we get the logprob of that choice text appearing
        after the prompt. We use the completion API to get logprobs directly.
        
        Args:
            system_prompt: System message
            user_prompt: User message (should end with "Respuesta:" or similar)
            choices: List of choice strings to evaluate
            sample_id: Sample identifier for logging
            
        Returns:
            Dictionary with 'output' (choice index), 'raw' (logprobs info), 'error'
        """
        result = {"output": None, "raw": None, "error": None}
        
        # Handle Anthropic - they don't support logprobs, fall back to text generation
        if self.provider_type == "anthropic":
            logger.warning(f"[{sample_id}] Anthropic doesn't support logprobs, falling back to text generation")
            return await self._generate_single_anthropic(system_prompt, user_prompt, None, sample_id)
        
        try:
            choice_logprobs = []
            
            # Combine system and user prompts for completion API
            combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            
            # For each choice, get logprob by using completion API
            for idx, choice in enumerate(choices):
                # Create prompt ending with the choice
                # We want to see the logprob of the choice text
                choice_prompt = f"{combined_prompt} {choice}"
                
                # Use completions API with logprobs (more reliable than chat for logprobs)
                try:
                    response = await self.client.completions.create(
                        model=self.model_name,
                        prompt=choice_prompt,
                        logprobs=5,  # Get top 5 logprobs
                        max_tokens=1,  # We only need to see the logprob of what comes after
                        temperature=0.0,  # Deterministic for logprobs
                        timeout=self.config["timeout"],
                    )
                    
                    # Get logprob from the response
                    if response.choices[0].logprobs:
                        # The logprobs are for tokens in the response
                        # We want the logprob of the choice text itself
                        # For now, use the average logprob of the first few tokens
                        token_logprobs = response.choices[0].logprobs.token_logprobs or []
                        if token_logprobs:
                            # Average logprob (sum would be better but this is simpler)
                            avg_logprob = sum(token_logprobs) / len(token_logprobs)
                            choice_logprobs.append((idx, avg_logprob, choice))
                        else:
                            # Fallback: try to get from top_logprobs
                            top_logprobs = response.choices[0].logprobs.top_logprobs
                            if top_logprobs and len(top_logprobs) > 0:
                                # Get the highest logprob from the first token
                                first_token_probs = top_logprobs[0]
                                if first_token_probs:
                                    max_logprob = max(tp.logprob for tp in first_token_probs)
                                    choice_logprobs.append((idx, max_logprob, choice))
                                else:
                                    choice_logprobs.append((idx, 0.0, choice))
                            else:
                                logger.warning(f"[{sample_id}] No logprobs for choice {idx}, using 0.0")
                                choice_logprobs.append((idx, 0.0, choice))
                    else:
                        logger.warning(f"[{sample_id}] No logprobs in response for choice {idx}, using 0.0")
                        choice_logprobs.append((idx, 0.0, choice))
                        
                except Exception as e:
                    logger.warning(f"[{sample_id}] Error getting logprobs for choice {idx}: {e}")
                    choice_logprobs.append((idx, float('-inf'), choice))
            
            # Select choice with highest logprob
            if choice_logprobs:
                best_choice = max(choice_logprobs, key=lambda x: x[1])
                result["output"] = best_choice[0]  # Return the index
                result["raw"] = f"Logprobs: {[(c[0], f'{c[1]:.4f}') for c in choice_logprobs]}, Selected: {best_choice[0]}"
                logger.debug(f"[{sample_id}] Selected choice {best_choice[0]} ({choices[best_choice[0]]}) with logprob {best_choice[1]:.4f}")
            else:
                result["error"] = "No logprobs obtained for any choice"
                
        except Exception as e:
            logger.error(f"[{sample_id}] Error in logprobs generation: {e}")
            result["error"] = f"Logprobs error: {e}"
        
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
                    use_logprobs=req.get("use_logprobs", False),
                    choices=req.get("choices"),
                )
        else:
            return await self._generate_single(
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                schema=req["schema"],
                sample_id=req["sample_id"],
                use_logprobs=req.get("use_logprobs", False),
                choices=req.get("choices"),
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
