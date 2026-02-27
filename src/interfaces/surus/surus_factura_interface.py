"""SURUS AI interface for /factura endpoint (invoice extraction)."""

import base64
import io
import json
import logging
from typing import Dict, Optional

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
        if (
            response.status_code in (400, 422)
            and self.request_image_field != "image_url"
        ):
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

    def _parse_response(self, response: Dict) -> Dict:
        """Parse SURUS Factura response to benchy format."""
        result = {"output": None, "raw": None, "error": None, "error_type": None}

        try:
            if "data" in response:
                result["output"] = response["data"]
                result["raw"] = json.dumps(response["data"])
            elif "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                result["raw"] = content
                result["output"] = json.loads(content)
            elif isinstance(response, dict):
                # Fallback for endpoints that return the extracted object directly.
                result["output"] = response
                result["raw"] = json.dumps(response)
            else:
                raise KeyError("Unexpected response type")
        except KeyError as e:
            logger.warning(f"Unexpected response format: {e}")
            result["raw"] = json.dumps(response)
            result["error"] = f"Unexpected response format: {e}"
            result["error_type"] = "invalid_response"
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            result["error"] = f"JSON parse error: {e}"
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
