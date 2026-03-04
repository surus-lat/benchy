"""SURUS AI interface for /factura endpoint (invoice extraction)."""

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from ..http_interface import HTTPInterface
from ..common.image_preprocessing import encode_image_data_url
from ...engine.protocols import InterfaceCapabilities
from ...engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)


class SurusFacturaInterface(HTTPInterface):
    """Interface for SURUS AI /factura endpoint for image extraction.

    This interface handles multimodal (image) inputs.
    The endpoint returns extracted data that is compared against the benchmark schema.
    """

    capabilities = InterfaceCapabilities(
        supports_multimodal=True,
        supports_schema=True,
        supports_files=True,
    )
    supports_multimodal = True

    def __init__(self, config: Dict, model_name: str, provider_type: str = "surus_factura"):
        """Initialize SURUS Factura interface."""
        super().__init__(config, model_name, provider_type)
        # Production docs use the `image` key; some legacy deployments accept `image_url`.
        self.request_image_field = self.config.get("request_image_field", "image")
        # Some factura deployments are strict about image decoding; normalize to JPEG by default.
        self.force_jpeg_payload = bool(self.config.get("force_jpeg_payload", True))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 90))
        logger.info(f"SURUS /factura endpoint: {self.endpoint}")

    def prepare_request(self, sample: Dict, task) -> Dict:
        """Prepare request for SURUS Factura endpoint."""
        return {
            "image_path": sample["image_path"],
            "sample_id": sample["id"],
        }

    @staticmethod
    def _flatten_error_tokens(payload: Any) -> List[str]:
        """Recursively flatten JSON error payload into comparable text tokens."""
        tokens: List[str] = []

        if isinstance(payload, dict):
            for key, value in payload.items():
                tokens.append(str(key).lower())
                tokens.extend(SurusFacturaInterface._flatten_error_tokens(value))
            return tokens

        if isinstance(payload, list):
            for item in payload:
                tokens.extend(SurusFacturaInterface._flatten_error_tokens(item))
            return tokens

        if payload is not None:
            tokens.append(str(payload).lower())

        return tokens

    def _should_retry_with_image_url(self, response: httpx.Response) -> bool:
        """Retry with `image_url` only when 4xx body indicates field-name mismatch."""
        if response.status_code not in (400, 422):
            return False
        if self.request_image_field == "image_url":
            return False

        tokens: List[str] = []
        try:
            tokens.extend(self._flatten_error_tokens(response.json()))
        except Exception:
            pass

        response_text = (response.text or "").strip().lower()
        if response_text:
            tokens.append(response_text)

        haystack = " | ".join(tokens)
        if not haystack:
            return False

        requested_field = str(self.request_image_field).lower()
        references_requested = requested_field in haystack
        references_image_url = "image_url" in haystack
        has_field_label = any(
            marker in haystack for marker in ("field", "parameter", "property")
        )
        if not has_field_label:
            return False

        # Retry only for field-name mismatch semantics, not for generic payload validation.
        explicit_wrong_field = any(
            marker in haystack
            for marker in (
                "unknown field",
                "unexpected field",
                "extra field",
                "unrecognized field",
                "unknown parameter",
                "unexpected parameter",
                "not allowed",
            )
        )
        explicit_image_url_required = references_image_url and any(
            marker in haystack
            for marker in (
                "required",
                "missing",
                "field required",
                "required field",
                "missing field",
                "must include",
            )
        )

        if references_requested and explicit_wrong_field:
            return True
        if explicit_image_url_required:
            return True
        if references_requested and references_image_url and any(
            marker in haystack
            for marker in ("use image_url", "expected image_url", "instead")
        ):
            return True
        return False

    def _image_to_data_url(self, image_path: str) -> str:
        """Convert local image file to base64 data URL."""
        if self.force_jpeg_payload:
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    processed = image.copy()

                    # Optional resize consistency with other interfaces.
                    if self.image_max_edge:
                        width, height = processed.size
                        longest_edge = max(width, height)
                        if longest_edge > self.image_max_edge:
                            scale = self.image_max_edge / float(longest_edge)
                            new_size = (
                                max(1, int(round(width * scale))),
                                max(1, int(round(height * scale))),
                            )
                            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                            processed = processed.resize(new_size, resample=resample)

                    if processed.mode not in {"RGB", "L"}:
                        processed = processed.convert("RGB")
                    elif processed.mode == "L":
                        processed = processed.convert("RGB")

                    output = io.BytesIO()
                    processed.save(
                        output,
                        format="JPEG",
                        quality=max(1, min(100, self.jpeg_quality)),
                        optimize=True,
                    )
                payload = base64.b64encode(output.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{payload}"
            except Exception as exc:
                logger.warning(
                    "Failed to normalize image to JPEG for SURUS Factura (%s). Falling back to raw encoding.",
                    exc,
                )

        return encode_image_data_url(
            image_path,
            max_edge=self.image_max_edge,
            logger=logger,
        )

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        """Make request to SURUS /factura endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        image_path = request["image_path"]
        if image_path.startswith(("http://", "https://", "data:")):
            image_url = image_path
        else:
            image_url = self._image_to_data_url(image_path)
            logger.debug(f"Converted local image to data URL ({len(image_url)} chars)")

        data = {
            self.request_image_field: image_url,
        }

        active_field = self.request_image_field
        response = await client.post(
            self.endpoint,
            headers=headers,
            json=data,
        )

        # Backward compatibility for older deployments that expect `image_url`.
        if self._should_retry_with_image_url(response):
            active_field = "image_url"
            fallback_data = {
                "image_url": image_url,
            }
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=fallback_data,
            )

        if response.status_code >= 400:
            error_text = (response.text or "").strip()
            if len(error_text) > 1000:
                error_text = error_text[:1000] + "..."
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} for {self.endpoint} "
                f"(payload field: {active_field}): {error_text}",
                request=response.request,
                response=response,
            )

        return response.json()

    def _parse_response(self, response: Any) -> Dict:
        """Parse SURUS Factura response to benchy format."""
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        try:
            if not isinstance(response, dict):
                raise KeyError("Unexpected response type: expected object")

            if "data" in response:
                result["output"] = response["data"]
                result["raw"] = json.dumps(response["data"])
            elif "choices" in response and isinstance(response["choices"], list) and response["choices"]:
                first_choice = response["choices"][0]
                message = first_choice.get("message") if isinstance(first_choice, dict) else None
                content = message.get("content") if isinstance(message, dict) else None

                if isinstance(content, dict):
                    result["output"] = content
                    result["raw"] = json.dumps(content)
                elif isinstance(content, str):
                    result["raw"] = content
                    try:
                        result["output"] = json.loads(content)
                    except json.JSONDecodeError as exc:
                        raise KeyError("choices[0].message.content is not valid JSON") from exc
                else:
                    raise KeyError("Unexpected choices content type")
            else:
                # Fallback for endpoints that return the extracted object directly.
                result["output"] = response
                result["raw"] = json.dumps(response)
        except KeyError as e:
            logger.warning(f"Unexpected response format: {e}")
            result["raw"] = json.dumps(response) if isinstance(response, (dict, list)) else str(response)
            result["error"] = f"Unexpected response format: {e}"
            result["error_type"] = "invalid_response"
        except (TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Response parse error: {e}")
            result["raw"] = json.dumps(response) if isinstance(response, (dict, list)) else str(response)
            result["error"] = f"Response parse error: {e}"
            result["error_type"] = "invalid_response"

        return result

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to SURUS Factura endpoint."""
        logger.info(f"Testing SURUS Factura API at {self.endpoint}")

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                # Minimal valid JPEG data URL payload (1x1 pixel). Some deployments
                # may still reject probe payload shape/content with 400/422 while being reachable.
                probe_image = (
                    "data:image/jpeg;base64,"
                    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBUQEBAVFRUVFRUVFRUVFRUVFRUWFhUVFRUYHSgg"
                    "GBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGxAQGi0lHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
                    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAA"
                    "AAADBAACBQYBB//EADYQAAICAQMCBAQEBQQDAAAAAAABAhEDEiExBBNBUSJhcQYygZGhsfAUI0JSYpLh8RVT"
                    "coKS/8QAGQEBAQEBAQEAAAAAAAAAAAAAAAECAwQF/8QAHhEBAQEBAQADAQEAAAAAAAAAAAECEQMhEjFBURP/"
                    "2gAMAwEAAhEDEQA/APv4ooooAKKKKACiiigAooooA//2Q=="
                )

                candidate_fields = [self.request_image_field]
                if self.request_image_field != "image_url":
                    candidate_fields.append("image_url")

                last_response: Optional[httpx.Response] = None
                for field in candidate_fields:
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        json={field: probe_image},
                    )
                    last_response = response
                    status_code = response.status_code

                    if 200 <= status_code < 300:
                        return True
                    if status_code in (400, 422):
                        logger.info(
                            "SURUS Factura endpoint reachable; probe payload rejected with HTTP %s (%s field)",
                            status_code,
                            field,
                        )
                        return True
                    if status_code == 401:
                        error_text = response.text[:200] if hasattr(response, "text") else ""
                        raise httpx.HTTPStatusError(
                            f"HTTP 401 Unauthorized: {error_text}",
                            request=response.request,
                            response=response,
                        )
                    if status_code == 403:
                        error_text = response.text[:200] if hasattr(response, "text") else ""
                        raise httpx.HTTPStatusError(
                            f"HTTP 403 Forbidden: {error_text}",
                            request=response.request,
                            response=response,
                        )

                if last_response is not None:
                    last_response.raise_for_status()
                return False

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=lambda exc, attempt: classify_http_exception(exc, attempt),
        )

        if result:
            logger.info(f"Connected to SURUS Factura at {self.endpoint}")
            return True

        logger.error(f"Failed to connect to SURUS Factura after {max_retries} attempts")
        if error:
            logger.error(f"  Last error: {error}")
        return False
