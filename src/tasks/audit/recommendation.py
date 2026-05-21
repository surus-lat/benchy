"""Algorithm 2 — Recommendation Synthesis handler.

Synthesizes per-policy evaluation results into a final audit recommendation.
One LLM call with business rules, approval overrides, and decision guidance.

Data format (JSONL):
    {"id": "0", "text": "<serialized {transfer_data, policy_results} JSON>",
     "expected": {"recommendation": "reject", "downgrade_to": null}}

Reference: audit-ai-algorithms.md § Algorithm 2.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common import StructuredHandler, load_jsonl_dataset

logger = logging.getLogger(__name__)

_RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendation": {
            "type": "string",
            "enum": ["accept", "reject", "downgrade", "ask_for_justification"],
        },
        "downgrade_to": {
            "type": ["string", "null"],
            "enum": [
                "UTIM",
                "Ambulancia con Medico",
                "Ambulancia sin Medico",
                "Remise",
                None,
            ],
        },
    },
    "required": ["recommendation"],
}

_RECOMMENDATION_SYSTEM_PROMPT = (
    "You are an audit decision synthesizer. Given per-policy evaluation results "
    "and the original transfer data, produce a final audit recommendation.\n\n"
    "BUSINESS CONTEXT:\n"
    "- Transfers over 30 km are paid individually (commercial incentive to approve).\n"
    "- Transfers 30 km or less are covered by monthly UGL quotas (reject when a policy "
    "justifies it, or when it improves margins).\n"
    "- Complexity hierarchy: UTIM > Ambulancia con Medico > Ambulancia sin Medico > Remise.\n"
    "- Downgrade constraint: only ONE step down is permitted.\n\n"
    "POLICY RESULT SEMANTICS:\n"
    "- compliant: the policy's rejection/downgrade condition was NOT met.\n"
    "- non_compliant: the policy's rejection/downgrade condition WAS met.\n"
    "- insufficient_data: required fields were missing — could not evaluate.\n"
    "- does_not_apply: treated as compliant (policy does not apply to this transfer).\n\n"
    "APPROVAL OVERRIDES (override rejections, except for mental health):\n"
    "1. distance_km > 30 → recommend accept (also verify complexity reduction).\n"
    "2. UGL = Salta → recommend accept (commercial agreement always in force).\n"
    "3. UGL = Rio Negro AND complejidad = Ambulancia con Medico → recommend accept.\n"
    "4. UGL = Neuquen AND complejidad = Ambulancia con Medico → recommend accept.\n"
    "5. UGL = Tucuman AND motivo = estudios complementarios → lean toward accept.\n"
    "6. origen = domicilio AND motivo = internacion → recommend accept "
    "(exception: if salud_mental fired, apply clinical judgment).\n\n"
    "DECISION GUIDANCE:\n"
    "- Any non_compliant → lean toward reject or downgrade.\n"
    "- Approval override applies → override wins (unless mental health rules involved).\n"
    "- does_not_apply = compliant for recommendation purposes.\n"
    "- complejidad_apropiada differs from requested → consider downgrade.\n"
    "- Downgrade: confirm exactly one step down; set downgrade_to explicitly.\n"
    "- Multiple insufficient_data → lean toward ask_for_justification.\n"
    "- Unsure between reject and ask_for_justification → prefer ask_for_justification.\n\n"
    "Return a JSON object with: recommendation, reasoning, confidence, downgrade_to."
)


class Recommendation(StructuredHandler):
    """Recommendation synthesis task — Algorithm 2 of the audit pipeline."""

    name = "recommendation"
    display_name = "Recommendation Synthesis"
    description = "Synthesize policy evaluation results into a final audit recommendation"

    answer_type = "structured"
    requires_schema = True
    default_data_file = "recommendation.jsonl"

    system_prompt = ""

    _schema = _RECOMMENDATION_SCHEMA

    metrics_config: Optional[Dict[str, Any]] = {
        "extraction_quality_score": {
            "enabled": True,
            "weights": {
                "schema_validity": 0.0,
                "field_f1_partial": 1.0,
                "inverted_hallucination": 0.0,
            },
        },
        "partial_matching": {
            "string": {
                "exact_threshold": 1.0,
            },
            "number": {
                "relative_tolerance": 0.0,
                "absolute_tolerance": 0.0,
            },
        },
        "normalization": {
            "case_sensitive": True,
            "normalize_whitespace": True,
            "unicode_normalize": False,
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not hasattr(self, "_global_schema") or self._global_schema is None:
            self._global_schema = self._schema

    def load_dataset(self) -> List[Dict[str, Any]]:
        file_path = self.data_dir / self.default_data_file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Audit recommendation dataset not found: {file_path}\n"
                "Create .data/audit/train.jsonl with samples shaped as:\n"
                '  {"id": "0", "text": "<{transfer_data, policy_results} JSON>", '
                '"expected": {"recommendation": "reject", "downgrade_to": null}}'
            )

        raw = load_jsonl_dataset(file_path)
        processed: List[Dict[str, Any]] = []
        for idx, sample in enumerate(raw):
            result = self.preprocess_sample(sample, idx)
            if result is not None:
                processed.append(result)
            else:
                logger.warning("Skipping sample %d: missing required fields", idx)
        return processed

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        text = sample.get("text", "")
        return _RECOMMENDATION_SYSTEM_PROMPT, json.dumps(text, ensure_ascii=False)

    def _keep_scored_fields(self, prediction: Any) -> Any:
        if isinstance(prediction, dict):
            return {
                k: v for k, v in prediction.items()
                if k in ("recommendation", "downgrade_to")
            }
        return prediction

    def calculate_metrics(
        self, prediction: Any, expected: Any, sample: Dict[str, Any],
        error: Optional[str] = None, error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        prediction = self._keep_scored_fields(prediction)
        return super().calculate_metrics(prediction, expected, sample, error, error_type)