"""OpenAI-compatible interface (vLLM, OpenAI, Anthropic, Together)."""

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from openai import AsyncOpenAI

from ..engine.protocols import InterfaceCapabilities, parse_interface_capabilities
from ..engine.retry import (
    RetryDecision,
    classify_http_exception,
    run_with_retries,
)

logger = logging.getLogger(__name__)


class OpenAIInterface:
    """Interface for OpenAI-compatible chat/completions APIs."""

    def __init__(self, connection_info: Dict[str, Any], model_name: str):
        """Initialize the OpenAI-compatible interface."""
        self.model_name = model_name
        self.base_url = connection_info["base_url"]
        self.provider_type = connection_info.get("provider_type", "openai")

        self.timeout = connection_info.get("timeout", 120)
        self.max_retries = connection_info.get("max_retries", 3)
        self.temperature = connection_info.get("temperature", 0.0)
        self.max_tokens = connection_info.get("max_tokens", 2048)
        self.max_tokens_param_name = connection_info.get("max_tokens_param_name") or "max_tokens"
        self.api_endpoint = connection_info.get("api_endpoint") or "auto"

        self.use_structured_outputs = connection_info.get("use_structured_outputs", False)
        base_capabilities = InterfaceCapabilities(
            supports_multimodal=True,
            supports_logprobs=False,
            supports_schema=True,
            supports_files=True,
            supports_batch=True,
        )
        self._capabilities = parse_interface_capabilities(
            connection_info.get("capabilities"),
            default=base_capabilities,
        )
        if "supports_logprobs" in connection_info:
            self._capabilities = parse_interface_capabilities(
                {"supports_logprobs": connection_info.get("supports_logprobs")},
                default=self._capabilities,
            )
        self._supports_logprobs = self._capabilities.supports_logprobs
        self.logprobs_top_k = connection_info.get("logprobs_top_k") or 20

        problematic_models = connection_info.get("problematic_models") or [
            "ByteDance-Seed/Seed-X-Instruct-7B",
            "ByteDance-Seed/Seed-X-PPO-7B",
        ]
        self.is_problematic_model = any(name in self.model_name for name in problematic_models)

        is_cloud = any(host in self.base_url for host in ["openai.com", "anthropic.com", "together.xyz"])
        default_concurrent = 2 if is_cloud else 20
        max_concurrent = connection_info.get("max_concurrent")
        if max_concurrent is None:
            max_concurrent = default_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        if self.provider_type == "anthropic":
            from anthropic import AsyncAnthropic

            api_key = self._get_api_key(connection_info, "ANTHROPIC_API_KEY", allow_empty=False)
            self.anthropic_client = AsyncAnthropic(api_key=api_key)
            self.client = None
            logger.info("Initialized Anthropic client")
        else:
            api_key = self._get_api_key(connection_info, "OPENAI_API_KEY", allow_empty=True)
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)
            self.anthropic_client = None
            logger.info(f"Initialized OpenAI-compatible client for {self.provider_type}")

        logger.info(f"Initialized OpenAIInterface for {model_name}")
        logger.info(f"  Base URL: {self.base_url}")
        if is_cloud:
            logger.info(f"  Rate limit: max {max_concurrent} concurrent requests")

    def _get_api_key(self, connection_info: Dict[str, Any], env_var: str, *, allow_empty: bool) -> str:
        api_key = connection_info.get("api_key")
        if not api_key:
            api_key_env = connection_info.get("api_key_env", env_var)
            api_key = os.getenv(api_key_env)
        if not api_key:
            if allow_empty:
                return "EMPTY"
            raise ValueError(
                f"API key not found. Set {env_var} environment variable or provide api_key."
            )
        return api_key

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request by getting prompts from task."""
        answer_type = getattr(task, "answer_type", None)
        use_logprobs = answer_type == "multiple_choice"
        requires_logprobs = getattr(task, "requires_logprobs", use_logprobs)
        prefers_logprobs = getattr(task, "prefers_logprobs", False)
        # Capability: only enable logprobs if required or preferred and supported.
        if use_logprobs:
            use_logprobs = requires_logprobs or (prefers_logprobs and self.supports_logprobs)

        # Task can override prompt formatting for logprobs-based scoring.
        if use_logprobs and hasattr(task, "get_prompt_for_logprobs"):
            system_prompt, user_prompt = task.get_prompt_for_logprobs(sample)
        else:
            system_prompt, user_prompt = task.get_prompt(sample)

        allow_logprobs_fallback = use_logprobs and not requires_logprobs
        if use_logprobs and not sample.get("choices"):
            if requires_logprobs:
                raise ValueError("Multiple-choice sample missing choices for logprobs scoring")
            use_logprobs = False
            allow_logprobs_fallback = False

        # Capability: drop schema payloads if the provider doesn't support them.
        schema = sample.get("schema") if self._capabilities.supports_schema else None
        request = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
            "sample_id": sample["id"],
            "use_logprobs": use_logprobs,
            "choices": sample.get("choices") if use_logprobs else None,
            "choice_labels": sample.get("choice_labels") if use_logprobs else None,
            "allow_logprobs_fallback": allow_logprobs_fallback,
        }

        # Capability: fail fast if multimodal inputs aren't supported.
        if "image_path" in sample:
            if not self._capabilities.supports_multimodal:
                raise ValueError("Interface does not support multimodal inputs")
            request["image_path"] = sample["image_path"]

        return request

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_image_media_type(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(ext, "image/jpeg")

    def _extract_json(self, text: str) -> str:
        text = text.strip()

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

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            json_candidate = text[first_brace:last_brace + 1]
            if json_candidate.count("{") == json_candidate.count("}"):
                return json_candidate

        return text

    def _classify_api_error(self, exc: Exception, attempt: int) -> RetryDecision:
        if isinstance(exc, ValueError):
            return RetryDecision(False, 0.0, "invalid_response")
        return classify_http_exception(exc, attempt)

    def _max_tokens_key(self) -> str:
        if self.provider_type == "openai":
            for marker in ["gpt-5", "o1", "o3", "o4"]:
                if marker in self.model_name.lower() and self.max_tokens_param_name == "max_tokens":
                    return "max_completion_tokens"
        return self.max_tokens_param_name

    async def _generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        sample_id: str,
        image_path: Optional[str] = None,
        use_logprobs: bool = False,
        choices: Optional[List[str]] = None,
        choice_labels: Optional[List[str]] = None,
        allow_logprobs_fallback: bool = False,
    ) -> Dict:
        """Generate output for a single request."""
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        if use_logprobs:
            logprobs_result = await self._generate_with_logprobs(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                choices=choices or [],
                choice_labels=choice_labels,
                sample_id=sample_id,
            )
            # If logprobs fail (e.g., no candidate tokens), fall back to completions.
            if allow_logprobs_fallback and logprobs_result.get("error"):
                return await self._generate_single(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    sample_id=sample_id,
                    image_path=image_path,
                    use_logprobs=False,
                    choices=None,
                    choice_labels=None,
                    allow_logprobs_fallback=False,
                )
            return logprobs_result

        if self.provider_type == "anthropic":
            return await self._generate_single_anthropic(system_prompt, user_prompt, schema, sample_id)

        use_completions_api = (
            self.api_endpoint == "completions"
            or (self.api_endpoint == "auto" and self.is_problematic_model)
        )
        if image_path:
            use_completions_api = False

        async def attempt_fn(_: int) -> Dict:
            if use_completions_api:
                combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                response = await self.client.completions.create(
                    model=self.model_name,
                    prompt=combined_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                raw_output = response.choices[0].text
                result["raw"] = raw_output
                if raw_output is None:
                    result["error"] = "Empty response content"
                    result["error_type"] = "invalid_response"
                    return result
                if schema:
                    cleaned = self._extract_json(raw_output)
                    try:
                        result["output"] = json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        result["error"] = f"JSON parse error: {e}"
                        result["error_type"] = "invalid_response"
                else:
                    result["output"] = raw_output.strip()
                return result

            if image_path:
                base64_image = self._encode_image(image_path)
                media_type = self._get_image_media_type(image_path)
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}},
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
                "timeout": self.timeout,
            }

            if not (self.provider_type == "openai" and any(m in self.model_name.lower() for m in ["gpt-5", "o1", "o3", "o4"])):
                params["temperature"] = self.temperature

            params[self._max_tokens_key()] = self.max_tokens

            if schema and self.use_structured_outputs:
                params["extra_body"] = {"structured_outputs": {"json": schema}}
            elif schema:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "extraction", "strict": True, "schema": schema},
                }

            response = await self.client.chat.completions.create(**params)
            raw_output = response.choices[0].message.content
            result["raw"] = raw_output
            if raw_output is None:
                result["error"] = "Empty response content"
                result["error_type"] = "invalid_response"
                return result

            if schema:
                cleaned = self._extract_json(raw_output)
                try:
                    result["output"] = json.loads(cleaned)
                except json.JSONDecodeError as e:
                    result["error"] = f"JSON parse error: {e}"
                    result["error_type"] = "invalid_response"
            else:
                result["output"] = raw_output.strip()

            return result

        response, error, error_type = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=self._classify_api_error,
        )

        if response is not None:
            return response

        result["error"] = error
        result["error_type"] = error_type
        return result

    async def _generate_with_logprobs(
        self,
        system_prompt: str,
        user_prompt: str,
        choices: List[str],
        choice_labels: Optional[List[str]],
        sample_id: str,
    ) -> Dict:
        """Generate prediction using log probabilities for multiple choice."""
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        # Capability: enforce logprobs support before issuing the request.
        if not self.supports_logprobs:
            result["error"] = "Logprobs not supported by interface"
            result["error_type"] = "invalid_response"
            return result
        if not choices:
            result["error"] = "No choices provided for logprobs scoring"
            result["error_type"] = "invalid_response"
            return result

        combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        if not combined_prompt.endswith((" ", "\n")):
            combined_prompt = combined_prompt + " "

        async def attempt_fn(_: int) -> Dict:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=combined_prompt,
                temperature=0.0,
                max_tokens=1,
                logprobs=self.logprobs_top_k,
                timeout=self.timeout,
            )

            logprobs = getattr(response.choices[0], "logprobs", None)
            if not logprobs or not getattr(logprobs, "top_logprobs", None):
                raise ValueError("No logprobs returned from provider")

            top_logprobs = logprobs.top_logprobs
            first_token = top_logprobs[0] if isinstance(top_logprobs, list) and top_logprobs else top_logprobs
            token_logprobs = self._coerce_top_logprobs(first_token)

            labels = self._resolve_choice_labels(choices, choice_labels)
            label_scores = {}
            for label in labels:
                variants = self._letter_variants(label)
                best = max((token_logprobs.get(v, float("-inf")) for v in variants), default=float("-inf"))
                label_scores[label] = best

            best_label = max(label_scores.items(), key=lambda item: item[1])
            if best_label[1] == float("-inf"):
                raise ValueError("No candidate labels found in logprobs")

            selected_idx = labels.index(best_label[0])
            return {
                "output": selected_idx,
                "raw": f"logprobs={label_scores} selected={best_label[0]}",
                "error": None,
                "error_type": None,
            }

        response, error, error_type = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=self._classify_api_error,
        )

        if response is not None:
            return response

        result["error"] = error
        result["error_type"] = error_type
        return result

    def _choice_labels(self, count: int) -> List[str]:
        return [chr(ord("A") + i) for i in range(count)]

    def _resolve_choice_labels(
        self,
        choices: List[str],
        choice_labels: Optional[List[str]],
    ) -> List[str]:
        # Allow numeric labels (0/1/2...) or custom labels for logprobs scoring.
        if choice_labels and len(choice_labels) == len(choices):
            return list(choice_labels)
        return self._choice_labels(len(choices))

    def _letter_variants(self, letter: str) -> List[str]:
        base = {letter, letter.lower()}
        variants = set(base)
        prefixes = [" ", "\n", "\n\n"]
        for prefix in prefixes:
            for token in base:
                variants.add(f"{prefix}{token}")
        for suffix in [")", ".", ":"]:
            for token in base:
                variants.add(f"{token}{suffix}")
                for prefix in prefixes:
                    variants.add(f"{prefix}{token}{suffix}")
        return list(variants)

    def _coerce_top_logprobs(self, top_logprobs: Any) -> Dict[str, float]:
        if isinstance(top_logprobs, dict):
            return top_logprobs
        if isinstance(top_logprobs, list):
            mapping: Dict[str, float] = {}
            for entry in top_logprobs:
                if isinstance(entry, dict):
                    if "token" in entry and "logprob" in entry:
                        mapping[entry["token"]] = entry["logprob"]
                    else:
                        for token, logprob in entry.items():
                            if isinstance(logprob, (int, float)):
                                mapping[token] = float(logprob)
                else:
                    token = getattr(entry, "token", None)
                    logprob = getattr(entry, "logprob", None)
                    if token is not None and logprob is not None:
                        mapping[token] = float(logprob)
            return mapping
        return {}

    async def close(self) -> None:
        """Close any underlying async clients."""
        await self._close_client(self.client, "openai")
        await self._close_client(self.anthropic_client, "anthropic")

    async def _close_client(self, client: Any, label: str) -> None:
        if client is None:
            return
        close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close_fn is None:
            return
        try:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug(f"Failed to close {label} client: {exc}")

    async def _generate_single_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict],
        sample_id: str,
    ) -> Dict:
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        if schema:
            schema_str = json.dumps(schema, indent=2)
            enhanced_user_prompt = (
                f"{user_prompt}\n\nPlease respond with valid JSON matching this schema:\n{schema_str}"
            )
        else:
            enhanced_user_prompt = user_prompt

        async def attempt_fn(_: int) -> Dict:
            response = await self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": enhanced_user_prompt}],
            )

            raw_output = response.content[0].text
            result["raw"] = raw_output
            if raw_output is None:
                result["error"] = "Empty response content"
                result["error_type"] = "invalid_response"
                return result

            if schema:
                cleaned = self._extract_json(raw_output)
                try:
                    result["output"] = json.loads(cleaned)
                except json.JSONDecodeError as e:
                    result["error"] = f"JSON parse error: {e}"
                    result["error_type"] = "invalid_response"
            else:
                result["output"] = raw_output.strip()

            return result

        response, error, error_type = await run_with_retries(
            attempt_fn,
            max_retries=self.max_retries,
            classify_error=self._classify_api_error,
        )

        if response is not None:
            return response

        result["error"] = error
        result["error_type"] = error_type
        return result

    async def _generate_with_limit(self, req: Dict) -> Dict:
        async with self._semaphore:
            return await self._generate_single(
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                schema=req.get("schema"),
                sample_id=req["sample_id"],
                image_path=req.get("image_path"),
                use_logprobs=req.get("use_logprobs", False),
                choices=req.get("choices"),
                choice_labels=req.get("choice_labels"),
                allow_logprobs_fallback=req.get("allow_logprobs_fallback", False),
            )

    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        tasks = [self._generate_with_limit(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        successful = 0
        errors = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i]['sample_id']} failed: {result}")
                processed.append({"output": None, "raw": None, "error": str(result), "error_type": "connectivity_error"})
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
        if self.provider_type == "anthropic":
            logger.info(f"Testing Anthropic API for model: {self.model_name}")

            async def attempt_fn(_: int) -> bool:
                await asyncio.wait_for(
                    self.anthropic_client.messages.create(
                        model=self.model_name,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Hi"}],
                    ),
                    timeout=timeout,
                )
                return True

            result, error, _ = await run_with_retries(
                attempt_fn,
                max_retries=max_retries,
                classify_error=self._classify_api_error,
            )
            if result:
                logger.info(f"Connected to Anthropic API, model '{self.model_name}' available")
                return True
            logger.error(f"Failed to connect to Anthropic API after {max_retries} attempts")
            if error:
                logger.error(f"  Last error: {error}")
            return False

        logger.info(f"Testing connection to {self.base_url}")

        async def attempt_fn(_: int) -> bool:
            try:
                models = await asyncio.wait_for(self.client.models.list(), timeout=timeout)
                model_ids = [m.id for m in models.data]
                if self.model_name in model_ids:
                    logger.info(f"Model '{self.model_name}' is available")
                else:
                    logger.warning(f"Model '{self.model_name}' not in list, but may still work")
                return True
            except (AttributeError, TypeError):
                test_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    self._max_tokens_key(): 5,
                }
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**test_params),
                    timeout=timeout,
                )
                if response.choices and response.choices[0].message:
                    return True
                raise ValueError("Invalid response from provider")

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=self._classify_api_error,
        )

        if result:
            logger.info(f"Connected to API at {self.base_url}")
            return True

        logger.error(f"Failed to connect after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False

    @property
    def supports_multimodal(self) -> bool:
        return self._capabilities.supports_multimodal

    @property
    def supports_logprobs(self) -> bool:
        return self._supports_logprobs

    @property
    def capabilities(self) -> InterfaceCapabilities:
        return self._capabilities
