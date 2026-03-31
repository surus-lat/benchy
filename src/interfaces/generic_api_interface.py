"""Generic API interface for benchmarking arbitrary HTTP endpoints."""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from .http_interface import HTTPInterface
from .common.image_preprocessing import encode_image_data_url
from ..engine.protocols import InterfaceCapabilities
from ..engine.retry import classify_http_exception, run_with_retries

logger = logging.getLogger(__name__)

# Pattern for {{field}} or {{field|filter}} placeholders in body templates.
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)(?:\|(\w+))?\}\}")


def _extract_by_path(obj: Any, path: str) -> Any:
    """Extract a value from a nested object using dot-notation path.

    Supports dict keys and integer list indices.
    E.g. "data", "choices.0.message.content".
    """
    if not path:
        return obj
    for part in path.split("."):
        if obj is None:
            return None
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def _image_to_data_url(
    image_path: str,
    *,
    max_edge: Optional[int] = None,
    force_jpeg: bool = True,
    jpeg_quality: int = 90,
) -> str:
    """Convert a local image file to a base64 data URL."""
    if image_path.startswith(("http://", "https://", "data:")):
        return image_path

    if force_jpeg:
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                processed = image.copy()
                if max_edge:
                    w, h = processed.size
                    longest = max(w, h)
                    if longest > max_edge:
                        scale = max_edge / float(longest)
                        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                        processed = processed.resize(new_size, resample=resample)
                if processed.mode not in {"RGB", "L"}:
                    processed = processed.convert("RGB")
                elif processed.mode == "L":
                    processed = processed.convert("RGB")
                output = io.BytesIO()
                processed.save(output, format="JPEG", quality=max(1, min(100, jpeg_quality)), optimize=True)
            payload = base64.b64encode(output.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{payload}"
        except Exception as exc:
            logger.warning("Failed to normalize image to JPEG (%s). Falling back to raw encoding.", exc)

    return encode_image_data_url(image_path, max_edge=max_edge, logger=logger)


def render_body_template(template: str, sample: Dict[str, Any], *, image_max_edge: Optional[int] = None) -> Any:
    """Render a JSON body template by substituting ``{{field}}`` placeholders.

    Supported filters:
        ``{{field}}``              – plain string substitution
        ``{{field|base64_image}}`` – read image file, encode as base64 data URL
        ``{{field|json}}``         – embed value as raw JSON (for dicts/lists)
        ``{{field|str}}``          – force to string

    When a placeholder (with or without filter) is the *entire* JSON value
    (i.e. ``"{{field|json}}"``), the resolved Python object replaces the
    JSON string so that dicts/lists/numbers are preserved in the final
    payload.
    """
    parsed = json.loads(template)
    return _resolve_node(parsed, sample, image_max_edge=image_max_edge)


def _resolve_node(node: Any, sample: Dict[str, Any], *, image_max_edge: Optional[int] = None) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_node(v, sample, image_max_edge=image_max_edge) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_node(item, sample, image_max_edge=image_max_edge) for item in node]
    if isinstance(node, str):
        return _resolve_string(node, sample, image_max_edge=image_max_edge)
    return node


def _apply_filter(value: Any, filt: Optional[str], *, image_max_edge: Optional[int] = None) -> Any:
    if filt is None or filt == "str":
        return str(value) if value is not None else ""
    if filt == "base64_image":
        return _image_to_data_url(str(value), max_edge=image_max_edge)
    if filt == "json":
        return value  # keep native type
    logger.warning("Unknown template filter '%s', treating as plain string", filt)
    return str(value) if value is not None else ""


def _resolve_string(text: str, sample: Dict[str, Any], *, image_max_edge: Optional[int] = None) -> Any:
    """Resolve placeholders in a string value.

    If the *entire* string is a single placeholder, return the resolved
    value directly (preserving dicts/lists/numbers). Otherwise perform
    string interpolation so multiple placeholders can coexist in one
    string value.
    """
    # Fast path: entire value is one placeholder → return native type.
    m = _PLACEHOLDER_RE.fullmatch(text)
    if m:
        field, filt = m.group(1), m.group(2)
        raw = sample.get(field)
        return _apply_filter(raw, filt, image_max_edge=image_max_edge)

    # Slow path: inline substitution — all results become strings.
    def _replacer(match: re.Match) -> str:
        field, filt = match.group(1), match.group(2)
        raw = sample.get(field)
        resolved = _apply_filter(raw, filt, image_max_edge=image_max_edge)
        if isinstance(resolved, (dict, list)):
            return json.dumps(resolved)
        return str(resolved)

    return _PLACEHOLDER_RE.sub(_replacer, text)


class GenericAPIInterface(HTTPInterface):
    """Fully CLI-configurable interface for benchmarking arbitrary HTTP APIs.

    All behaviour is driven by configuration — no subclassing required.
    """

    capabilities = InterfaceCapabilities(
        supports_multimodal=True,
        supports_schema=True,
        supports_files=True,
    )

    def __init__(self, config: Dict, model_name: str, provider_type: str = "api"):
        super().__init__(config, model_name, provider_type)
        self.http_method = (self.config.get("http_method") or "POST").upper()
        self.body_template = self.config.get("body_template")
        self.response_path = self.config.get("response_path")
        self.extra_headers = self.config.get("headers") or {}
        self.force_jpeg = bool(self.config.get("force_jpeg_payload", True))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 90))

        if not self.body_template:
            raise ValueError(
                "GenericAPIInterface requires a body_template (--api-body-template). "
                "Example: '{\"image\": \"{{image_path|base64_image}}\"}'"
            )

        # Pre-parse to validate JSON.
        try:
            json.loads(self.body_template)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--api-body-template is not valid JSON: {exc}") from exc

        # Discover which sample fields the template references.
        self._template_fields = [m.group(1) for m in _PLACEHOLDER_RE.finditer(self.body_template)]
        self._template_has_images = any(
            m.group(2) == "base64_image" for m in _PLACEHOLDER_RE.finditer(self.body_template)
        )
        logger.info(
            "GenericAPI endpoint: %s (%s) — template fields: %s, response_path: %s",
            self.endpoint,
            self.http_method,
            self._template_fields,
            self.response_path or "(root)",
        )

    def prepare_request(self, sample: Dict, task) -> Dict:
        return {"sample": sample, "sample_id": sample.get("id", "unknown")}

    async def _make_request_with_client(
        self,
        client: httpx.AsyncClient,
        request: Dict,
    ) -> Optional[Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        sample = request.get("sample", request)
        body = render_body_template(
            self.body_template,
            sample,
            image_max_edge=self.image_max_edge,
        )

        response = await client.request(
            self.http_method,
            self.endpoint,
            headers=headers,
            json=body,
        )

        if response.status_code >= 400:
            error_text = (response.text or "").strip()
            if len(error_text) > 1000:
                error_text = error_text[:1000] + "..."
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} for {self.endpoint}: {error_text}",
                request=response.request,
                response=response,
            )

        return response.json()

    def _parse_response(self, response: Any) -> Dict:
        result: Dict[str, Any] = {"output": None, "raw": None, "error": None, "error_type": None}

        try:
            extracted = _extract_by_path(response, self.response_path) if self.response_path else response

            if isinstance(extracted, str):
                result["raw"] = extracted
                try:
                    result["output"] = json.loads(extracted)
                except json.JSONDecodeError:
                    result["output"] = extracted
            elif isinstance(extracted, dict):
                result["output"] = extracted
                result["raw"] = json.dumps(extracted)
            elif isinstance(extracted, list):
                result["output"] = extracted
                result["raw"] = json.dumps(extracted)
            else:
                result["output"] = extracted
                result["raw"] = str(extracted) if extracted is not None else None
        except Exception as exc:
            logger.warning("Response parse error: %s", exc)
            result["raw"] = json.dumps(response) if isinstance(response, (dict, list)) else str(response)
            result["error"] = f"Response parse error: {exc}"
            result["error_type"] = "invalid_response"

        return result

    def build_test_request(self) -> Dict[str, Any]:
        """Build a minimal probe request for connection testing."""
        # Build a sample with placeholder values for each template field.
        fake_sample: Dict[str, Any] = {"id": "connection_test"}
        for field_match in _PLACEHOLDER_RE.finditer(self.body_template):
            field, filt = field_match.group(1), field_match.group(2)
            if filt == "base64_image":
                fake_sample[field] = (
                    "data:image/jpeg;base64,"
                    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBUQEBAVFRUVFRUVFRUVFRUVFRUWFhUVFRUYHSgg"
                    "GBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGxAQGi0lHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
                    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAA"
                    "AAADBAACBQYBB//EADYQAAICAQMCBAQEBQQDAAAAAAABAhEDEiExBARBUSJhcQUygZGhsfAUI0JSYpLh8RVT"
                    "coKS/8QAGQEBAQEBAQEAAAAAAAAAAAAAAAECAwQF/8QAHhEBAQEBAQADAQEAAAAAAAAAAAECEQMhEjFBURP/"
                    "2gAMAwEAAhEDEQA/APv4ooooAKKKKACiiigAooooA//2Q=="
                )
            elif filt == "json":
                fake_sample[field] = {"type": "object", "properties": {"status": {"type": "string"}}}
            else:
                fake_sample[field] = "test"
        return {"sample": fake_sample, "sample_id": "connection_test"}

    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        logger.info("Testing GenericAPI at %s", self.endpoint)

        async def attempt_fn(_: int) -> bool:
            async with httpx.AsyncClient(timeout=timeout) as client:
                test_req = self.build_test_request()
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self.extra_headers,
                }
                sample = test_req.get("sample", test_req)
                body = render_body_template(
                    self.body_template,
                    sample,
                    image_max_edge=self.image_max_edge,
                )
                response = await client.request(
                    self.http_method,
                    self.endpoint,
                    headers=headers,
                    json=body,
                )
                status = response.status_code
                if 200 <= status < 300:
                    return True
                if status in (400, 422):
                    logger.info("GenericAPI endpoint reachable; probe rejected with HTTP %s", status)
                    return True
                if status in (401, 403):
                    error_text = response.text[:200] if hasattr(response, "text") else ""
                    raise httpx.HTTPStatusError(
                        f"HTTP {status}: {error_text}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return False

        result, error, _ = await run_with_retries(
            attempt_fn,
            max_retries=max_retries,
            classify_error=lambda exc, attempt: classify_http_exception(exc, attempt),
        )

        if result:
            logger.info("Connected to GenericAPI at %s", self.endpoint)
            return True

        logger.error("Failed to connect to GenericAPI after %s attempts", max_retries)
        if error:
            logger.error("  Last error: %s", error)
        return False
