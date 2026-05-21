"""Algorithm 1 — Policy Evaluation handler.

Evaluates a medical transfer record against 7 audit policies in a single
batched LLM call. Returns per-policy compliance status.

Data format (JSONL):
    {"id": "0", "text": "<serialized audit_input JSON>",
     "expected": {"salud_mental": "compliant", "geriatrico": "compliant", ...}}

Reference: audit-ai-algorithms.md § Algorithm 1.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common import StructuredHandler, load_jsonl_dataset

logger = logging.getLogger(__name__)

_STATUS_ENUM = ["compliant", "non_compliant", "insufficient_data", "does_not_apply"]

_POLICY_IDS = [
    "salud_mental",
    "geriatrico",
    "dialisis",
    "rehabilitacion_cronica",
    "ugl_mendoza",
    "ugl_cordoba_downgrade",
    "ugl_tucuman_quimio",
]

_POLICIES_SCHEMA = {
    "type": "object",
    "properties": {pid: {"type": "string", "enum": _STATUS_ENUM} for pid in _POLICY_IDS},
    "required": _POLICY_IDS,
    "additionalProperties": False,
}

_POLICY_EVAL_SYSTEM_PROMPT = (
    "You are an audit evaluator. For each policy listed below, "
    "evaluate the provided transfer data and return a JSON object "
    "with the exact structure specified.\n\n"
    "--- POLICY 1: Salud Mental (salud_mental) ---\n"
    "Reject transfers related to mental health conditions. "
    "A transfer is related to mental health when: "
    "the destination is a mental health facility (centro de dia, "
    "clinica psiquiatrica, hospital de salud mental), "
    "the medical practice mentions psychology/psychiatry consultation, "
    "or the diagnosis explicitly indicates a mental health condition.\n\n"
    "--- POLICY 2: Geriatrico (geriatrico) ---\n"
    "Reject transfers between a private residence (domicilio particular) "
    "and a geriatric facility (geriátrico), in either direction.\n\n"
    "--- POLICY 3: Dialisis (dialisis) ---\n"
    "Reject transfers where the medical practice is dialysis "
    "(diálisis or hemodiálisis).\n\n"
    "--- POLICY 4: Rehabilitacion Cronica (rehabilitacion_cronica) ---\n"
    "Reject rehabilitation transfers when the same-diagnosis history "
    "spans more than 90 calendar days. Check historico_traslados for the "
    "patient's same-diagnosis date range.\n\n"
    "--- POLICY 5: UGL Mendoza — Amb sin Medico (ugl_mendoza) ---\n"
    "Reject when UGL is Mendoza and complexity (complejidad) "
    "is ambulancia sin medico.\n\n"
    "--- POLICY 6: UGL Cordoba — Downgrade a Remise (ugl_cordoba_downgrade) ---\n"
    "Downgrade to Remise when UGL is Cordoba, the practice is rehabilitation, "
    "and complexity is ambulancia sin medico.\n\n"
    "--- POLICY 7: UGL Tucuman — Quimio Fin de Semana (ugl_tucuman_quimio) ---\n"
    "Reject when UGL is Tucuman, the practice is chemotherapy (quimioterapia), "
    "and the transfer is scheduled on Saturday or Sunday.\n\n"
    "Return a JSON object with keys matching each policy ID. "
    "Each value must have: status (compliant|non_compliant|insufficient_data|does_not_apply). "
    "The transfer data to evaluate is provided in the user message."
)


class PolicyEval(StructuredHandler):
    """Policy evaluation task — Algorithm 1 of the audit pipeline."""

    name = "policy_eval"
    display_name = "Policy Evaluation"
    description = "Evaluate transfer data against 7 medical audit policies in a single batched LLM call"

    answer_type = "structured"
    requires_schema = True
    default_data_file = "policy_eval.jsonl"

    system_prompt = ""

    _schema = _POLICIES_SCHEMA

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
                f"Audit policy eval dataset not found: {file_path}\n"
                "Create .data/audit/train.jsonl with samples shaped as:\n"
                '  {"id": "0", "text": "<audit_input JSON>", '
                '"expected": {"salud_mental": "compliant", ...}}'
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
        return _POLICY_EVAL_SYSTEM_PROMPT, json.dumps(text, ensure_ascii=False)

    def _unwrap_policy_statuses(self, prediction: Any) -> Any:
        if not isinstance(prediction, dict):
            return prediction
        unwrapped: Dict[str, Any] = {}
        for key, value in prediction.items():
            if isinstance(value, dict) and "status" in value:
                unwrapped[key] = value["status"]
            else:
                unwrapped[key] = value
        return unwrapped

    def calculate_metrics(
        self, prediction: Any, expected: Any, sample: Dict[str, Any],
        error: Optional[str] = None, error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        prediction = self._unwrap_policy_statuses(prediction)
        return super().calculate_metrics(prediction, expected, sample, error, error_type)