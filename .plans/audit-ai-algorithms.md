# Audit AI Algorithms — Input and Output Schemas

## Overview

The audit AI workflow consists of two sequential LLM-based algorithms:

1. **Policy Evaluation** — evaluates each transfer against 7 medical audit policies in a single batched LLM call
2. **Recommendation Synthesis** — a second LLM call that synthesizes the policy results into a final audit decision

Both algorithms share the same canonical input shape: the **audit input document** (defined by `backend/schemas/audit_input.json`). Algorithm 2 additionally receives the output of Algorithm 1.

---

## Algorithm 1 — Policy Evaluation

**Source:** `auditor/src/auditor/core.py` — `Auditor.audit()` method (line 228)

### Purpose

Evaluate a medical transfer request against 7 predefined audit policies. Each policy checks one specific rejection or downgrade condition. All 7 policies are batched into a **single LLM API call** (they share the `auditoria_medica` group).

### Input Schema

#### Audit Input Document (`input_data: dict`)

The full transfer record, assembled from the extraction pipeline output. Defined by `backend/schemas/audit_input.json` (JSON Schema draft 2020-12).

**Required fields (must be present as keys, values can be `null`):**

| Field | Type | Description |
|-------|------|-------------|
| `complejidad` | `string \| null` | Raw complexity from IHSA. Values: `"UTIM"`, `"amb con medico"`, `"amb sin medico"`, `"remise"` |
| `tipo_vehiculo` | `string \| null` | Normalized vehicle type. Values: `"TRASLADO SIN MEDICO"`, `"AMBULANCIA COMUN CON MEDICO"`, `"REMISE"`, `"UTIM"` |

**Optional fields (key can be absent; value can be `null`):**

| Field | Type | Description |
|-------|------|-------------|
| `nombre_beneficiario` | `string \| null` | Beneficiary full name (from `apellido_y_nombre`) |
| `numero_beneficiario` | `string \| null` | Beneficiary ID number (from `n_de_beneficio`) |
| `edad` | `integer \| null` | Beneficiary age in years |
| `practica_a_realizar` | `string \| null` | Raw text describing the medical procedure (IHSA `CodMotivo`) |
| `practica_a_realizar_category` | `string \| null` | Standardized procedure category. Enum: `"Estudio"`, `"Interconsultas"`, `"Rehabilitación"`, `"Derivación centro mayor compl"`, `"Alta médica"`, `"Internación"`, `"Traslado Urgente sin OP"`, `"Radioterapia"`, `"Quimioterapia"`, `"Pre-post trasplantados"`, `"Falta de cama"`, `"Salud mental"` |
| `trayecto` | `string \| null` | Journey type. Enum: `"IDA"`, `"IDA Y VUELTA"` |
| `modalidad_traslado_paciente` | `string \| null` | Patient seating modality. Enum: `"sentado"`, `"silla_de_ruedas"`, `"camilla"` |
| `autovalido` | `boolean \| null` | Whether patient can travel independently (deferred — always `null`) |
| `diagnostico` | `string \| null` | Clinical diagnosis from solicitud form |
| `observaciones` | `string \| null` | Operator free-text notes from IHSA |
| `justificacion_traslado` | `string \| null` | Doctor's written justification for the transfer |
| `nombre_doctor` | `string \| null` | Full name of requesting physician |
| `matricula_doctor` | `string \| null` | Physician license/registration number |
| `ugl` | `string \| null` | UGL (Unidad de Gestión Local) name |
| `cliente` | `string \| null` | Client name from IHSA (`NomCliente`) |
| `distance_km` | `number \| null` | Round-trip distance in km (from geocoding) |
| `cronograma` | `array \| null` | Scheduled transfer dates/times. Each entry: `{ fecha: "DD/MM/YYYY", hora: "HH:MM" }` |
| `cant_de_traslados` | `integer \| null` | Number of transfers requested (min: 1) |
| `complejidad_apropiada` | `string \| null` | Medically appropriate complexity from diagnostic lookup. Enum: `"UTIM"`, `"amb con medico"`, `"amb sin medico"`, `"remise"` |
| `origen` | `object \| null` | Transfer origin: `{ domicilio: string, localidad: string, provincia: string, lat?: number, lon?: number }` |
| `destino` | `object \| null` | Transfer destination (same shape as `origen`) |
| `historico_traslados` | `object \| null` | Pre-computed historical context (see subsection below) |
| `_pun` | `integer \| null` | Internal IHSA identifier (metadata, ignored by policies) |
| `_case_id` | `string \| null` | Case/job identifier (metadata, ignored) |
| `_num_solicitud` | `string \| null` | IHSA solicitud number (metadata, ignored) |

#### Historical Context (`historico_traslados`)

Pre-computed and injected into the prompt. The LLM reads pre-aggregated facts — it **never** performs counting, filtering, or date math itself. `null` when the enrichment pipeline is not yet integrated.

```json
{
  "historico_traslados": {
    "entradas": [
      {
        "fecha": "2025-01-15",
        "diagnostico": "Enfermedad de Parkinson"
      }
    ],
    "paciente": {
      "total_traslados": 12,
      "mismo_diagnostico_traslados": 8,
      "mismo_diagnostico_primer_fecha": "2024-10-01",
      "mismo_diagnostico_ultima_fecha": "2025-04-30",
      "mismo_diagnostico_dias": 211
    },
    "ugl": {
      "solicitados_30d": 340,
      "rechazados_30d": 52,
      "amb_sin_medico_30d": 180
    }
  }
}
```

- `entradas`: raw transfer entries (last 120 days, max 50). Exposed so the LLM can semantically override exact-text-match aggregates
- `paciente`: patient-level aggregates based on exact text match of `diagnostico`
- `ugl`: UGL-level aggregates (last 30 days), consumed by future quota-aware policies

#### Prompt Construction

For each policy, the template in `auditor/policies.json` renders with two placeholders:
- `{policy_description}` → the policy's `description` field
- `{input_data}` → the full audit input document serialized as indented JSON

All rendered policy prompts are concatenated into a **single system message**:

```
You are an audit evaluator. For each policy listed below,
evaluate the provided data and return a JSON object with
the exact structure specified.

[Policy: Salud Mental (salud_mental)]
Policy: Rechaza traslados relacionados con salud mental...
{evaluation rules specific to this policy}
Transfer data to evaluate:
{full input_data JSON}

---

[Policy: Geriátrico (geriatrico)]
Policy: Rechaza traslados desde domicilio particular a geriatrico...
{evaluation rules specific to this policy}
Transfer data to evaluate:
{full input_data JSON}

---
... (all 7 policies)
```

The **user message** is the `input_data` JSON string.

#### LLM Configuration

| Parameter | Value |
|-----------|-------|
| `temperature` | `0.1` |
| `max_tokens` | `4096` |
| `logprobs` | `true` (enables confidence computation) |
| `response_format` | OpenAI structured output (`json_schema`, `strict: true`) |

### Output Schema

#### Direct LLM Response

Structured JSON enforced by OpenAI `response_format`:

```json
{
  "policies": {
    "<policy_id>": {
      "status": "compliant" | "non_compliant" | "insufficient_data" | "does_not_apply",
      "reasoning": "<string explanation>",
      "confidence": <float, 0.0 to 1.0>
    }
  }
}
```

The `policies` object **must** contain all 7 policy IDs as keys:
- `salud_mental`
- `geriatrico`
- `dialisis`
- `rehabilitacion_cronica`
- `ugl_mendoza`
- `ugl_cordoba_downgrade`
- `ugl_tucuman_quimio`

#### Status Values — Semantics

| Value | Meaning |
|-------|---------|
| `compliant` | This policy's rejection/downgrade condition was **NOT** met — no action needed from this policy |
| `non_compliant` | This policy's rejection/downgrade condition **WAS** met — consider reject or downgrade |
| `insufficient_data` | Required fields were missing — policy could not be evaluated |
| `does_not_apply` | This policy does not apply to this transfer (e.g., wrong UGL) — treated as `compliant` |

#### Post-Processed Result (`PolicyResult` dataclass)

After the LLM responds, the raw output is transformed into `PolicyResult` instances (`auditor/src/auditor/models.py:42-50`):

| Field | Type | Source |
|-------|------|--------|
| `status` | `PolicyStatus` | From LLM response |
| `reasoning` | `str` | From LLM response |
| `confidence` | `float` (0.0–1.0) | **Recomputed from token logprobs** (see below) |
| `evaluation_time_ms` | `float` | Wall-clock time of the grouped API call |
| `trace_id` | `str` | UUID generated per audit run |

#### Confidence Computation

The LLM's self-assessed `confidence` field is **discarded**. Instead, confidence is derived from token logprobs (`core.py:498-574`):

1. The raw response text is scanned for each policy ID marker (`"salud_mental":`)
2. For each policy, the character range containing its inner `{...}` object is located
3. All logprob tokens whose position falls within that range are collected
4. Per-policy confidence = `exp(avg_logprob)` across those tokens
5. Defaults to `0.5` if no logprobs are available or no tokens match a given policy

#### Aggregate Return Value

```python
(results: dict[str, PolicyResult], metrics: AuditMetrics)
```

`AuditMetrics` (`models.py:91-105`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `total_tokens` | `int` | Total tokens consumed (prompt + completion) |
| `latency_ms` | `float` | Total latency of the grouped API call |
| `model_used` | `str` | Model name (configured via env) |
| `calls_made` | `int` | Number of API calls made (1 for this group) |
| `per_call` | `list[dict]` | Per-call breakdown: group, tokens, latency, model |
| `trace` | `TraceInfo` | Trace metadata (trace_id, input_hash, policy_count, started_at) |
| `trail` | `list[AuditEvent]` | Structured event log for the audit trail |
| `api_calls` | `list[ApiCallRecord]` | Per-attempt API call records (including retries) |
| `per_group_time_ms` | `dict[str, float]` | Latency per group |

---

## Algorithm 2 — Recommendation Synthesis

**Source:** `backend/src/services/auditor_service.py` — `_run_recommendation_sync()` (line 328)

### Purpose

Synthesize the per-policy evaluation results into a single, actionable audit recommendation. This second LLM call applies business rules, approval overrides, and decision guidance that spans across policies.

### Input Schema

#### System Prompt

A fixed, detailed system prompt (`_RECOMMENDATION_SYSTEM_PROMPT`, line 167) setting the LLM's role and rules. It contains:

**Business Context:**
- Transfers >30km are paid individually (commercial incentive to approve)
- Transfers <=30km are covered by monthly UGL quotas (reject when policy justifies, or when it improves margins)
- Complexity hierarchy: `UTIM > Ambulancia con Médico > Ambulancia sin Médico > Remise`
- Downgrade constraint: only **one step down** permitted

**Policy Result Semantics:**
- `compliant`: rejection/downgrade condition was NOT met
- `non_compliant`: rejection/downgrade condition WAS met
- `insufficient_data`: required fields missing
- `does_not_apply`: treated as `compliant`

**Approval Overrides** (override rejections, except for mental health):
| Condition | Action |
|-----------|--------|
| `distance_km > 30` | Recommend `accept` (also verify complexity reduction) |
| UGL = Salta | Recommend `accept` (commercial agreement always in force) |
| UGL = Rio Negro AND complejidad = Ambulancia con Medico | Recommend `accept` |
| UGL = Neuquen AND complejidad = Ambulancia con Medico | Recommend `accept` |
| UGL = Tucuman AND motivo = estudios complementarios | Lean toward `accept` |
| origen = domicilio AND motivo = internacion | Recommend `accept` (exception: if `salud_mental` fired, apply clinical judgment) |

**Decision Guidance:**
- Any `non_compliant` → lean toward `reject` or `downgrade`
- Approval override applies → override wins (unless mental health rules involved)
- `does_not_apply` = `compliant` for recommendation purposes
- `complejidad_apropiada` differs from requested → consider `downgrade`
- Downgrade: confirm exactly one step down; set `downgrade_to` explicitly
- Multiple `insufficient_data` → lean toward `ask_for_justification`
- Unsure between `reject` and `ask_for_justification` → prefer `ask_for_justification`

#### User Message

Sent as a JSON object combining the transfer data and serialized policy results:

```json
{
  "transfer_data": {
    "complejidad": "amb sin medico",
    "tipo_vehiculo": "TRASLADO SIN MEDICO",
    "ugl": "Mendoza",
    "practica_a_realizar": "Internación hospitalaria",
    "practica_a_realizar_category": "Internación",
    "diagnostico": "Neumonía adquirida en la comunidad",
    "distance_km": 12.5,
    "origen": {
      "domicilio": "Av. San Martín 1234",
      "localidad": "Mendoza",
      "provincia": "Mendoza"
    },
    "destino": {
      "domicilio": "Hospital Central",
      "localidad": "Mendoza",
      "provincia": "Mendoza"
    },
    "cronograma": [
      { "fecha": "02/05/2025", "hora": "10:00" }
    ],
    "complejidad_apropiada": null,
    "historico_traslados": null
  },
  "policy_results": {
    "salud_mental": {
      "status": "compliant",
      "reasoning": "No se detectó patología de salud mental. El diagnóstico es Neumonía, el destino es Hospital Central (no centro de salud mental, no centro de día, no consulta psicológica/psiquiátrica).",
      "confidence": 0.99
    },
    "geriatrico": {
      "status": "compliant",
      "reasoning": "Origen clasificado como domicilio particular, pero destino es Hospital Central (centro médico, no geriátrico). No aplica rechazo geriátrico.",
      "confidence": 0.97
    },
    "dialisis": {
      "status": "does_not_apply",
      "reasoning": "La práctica es Internación hospitalaria, no diálisis.",
      "confidence": 0.99
    },
    "rehabilitacion_cronica": {
      "status": "does_not_apply",
      "reasoning": "La práctica es Internación, no rehabilitación.",
      "confidence": 0.99
    },
    "ugl_mendoza": {
      "status": "non_compliant",
      "reasoning": "RECHAZAR: amb sin medico en UGL Mendoza — politica comercial. UGL = Mendoza, complejidad = ambulancia sin medico.",
      "confidence": 0.98
    },
    "ugl_cordoba_downgrade": {
      "status": "does_not_apply",
      "reasoning": "No aplica: UGL es Mendoza, no Córdoba.",
      "confidence": 0.99
    },
    "ugl_tucuman_quimio": {
      "status": "does_not_apply",
      "reasoning": "No aplica: UGL es Mendoza, no Tucumán.",
      "confidence": 0.99
    }
  }
}
```

The `policy_results` object contains exactly one entry per policy. Each entry has:
- `status` — one of `compliant`, `non_compliant`, `insufficient_data`, `does_not_apply`
- `reasoning` — the explanation from Algorithm 1
- `confidence` — the logprob-derived confidence from Algorithm 1

#### LLM Configuration

| Parameter | Value |
|-----------|-------|
| `temperature` | `0.1` |
| `max_tokens` | `1200` |
| `logprobs` | `false` (not used in this step) |
| `response_format` | OpenAI structured output (`json_schema`, `strict: false`) |

### Output Schema

#### Direct LLM Response

Enforced by `_RECOMMENDATION_SCHEMA` (`auditor_service.py:223-246`):

```json
{
  "recommendation": "accept" | "reject" | "downgrade" | "ask_for_justification",
  "reasoning": "<concise explanation referencing relevant policies and overrides>",
  "confidence": <float, 0.0 to 1.0>,
  "downgrade_to": "UTIM" | "Ambulancia con Médico" | "Ambulancia sin Médico" | "Remise" | null
}
```

#### Field Specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `recommendation` | `string` | Yes | The audit decision. One of: `"accept"`, `"reject"`, `"downgrade"`, `"ask_for_justification"` |
| `reasoning` | `string` | Yes | Concise explanation linking policy results, overrides, and the final decision |
| `confidence` | `float` (0–1) | Yes | LLM self-assessed certainty in the recommendation |
| `downgrade_to` | `string \| null` | No | Target complexity level. Only set when `recommendation = "downgrade"`. One of: `"UTIM"`, `"Ambulancia con Médico"`, `"Ambulancia sin Médico"`, `"Remise"`. `null` otherwise |

#### Recommendation Values — Semantics

| Value | Meaning |
|-------|---------|
| `accept` | Approve the transfer as-is (at the requested complexity level) |
| `reject` | Deny the transfer entirely |
| `downgrade` | Approve at a reduced complexity level (exactly one step down). `downgrade_to` must be set |
| `ask_for_justification` | Request additional clinical information before deciding |

#### Downgrade Normalization

The `_normalize_downgrade_to` function (`auditor_service.py:525-528`) handles common LLM casing/spelling variants:

| LLM might return | Normalized to |
|------------------|---------------|
| `"utim"` | `"UTIM"` |
| `"ambulancia con medico"`, `"amb con medico"`, `"ambulancia con médico"` | `"Ambulancia con Médico"` |
| `"ambulancia sin medico"`, `"amb sin medico"`, `"ambulancia sin médico"` | `"Ambulancia sin Médico"` |
| `"remise"`, `"remis"` | `"Remise"` |
| `""`, `"null"` | `null` |

#### Safety/Fallback

If Algorithm 2's LLM call fails (network error, empty response, non-JSON), the orchestrator (`auditor_service.py:462-471`) falls back to:

```json
{
  "recommendation": "ask_for_justification",
  "reasoning": "Recommendation AI unavailable: <error message>",
  "confidence": 0.0,
  "downgrade_to": null
}
```

---

## The 7 Policies — Reference

Each policy is defined in `auditor/policies.json`. All are type `"llm"`, group `"auditoria_medica"`, `"required": false`.

| ID | Name | Trigger Condition | Action |
|----|------|-------------------|--------|
| `salud_mental` | Salud Mental | Mental health internacion, centro de dia, or psicologo/psiquiatra consult | Reject |
| `geriatrico` | Geriátrico | Transfer between domicilio particular and geriatrico (either direction) | Reject |
| `dialisis` | Dialisis | Transfer reason is dialysis (diálisis/hemodiálisis) | Reject |
| `rehabilitacion_cronica` | Rehabilitacion Cronica (>90 dias) | Same-diagnosis rehab transfers spanning >90 calendar days | Reject |
| `ugl_mendoza` | UGL Mendoza — Amb sin Medico | UGL = Mendoza + complejidad = ambulancia sin medico | Reject |
| `ugl_cordoba_downgrade` | UGL Cordoba — Downgrade a Remise | UGL = Cordoba + rehab + complejidad = ambulancia sin medico | Downgrade to Remise |
| `ugl_tucuman_quimio` | UGL Tucuman — Quimio Fin de Semana | UGL = Tucuman + quimioterapia + scheduled Saturday or Sunday | Reject |

---

## Complete Data Flow

```
┌─────────────────────────────┐
│  Extraction Pipeline Output │
│  (output_data dict)         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  audit_input_builder.py     │
│  build_audit_input()        │
│  → audit_input dict         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  ALGORITHM 1 — Policy Evaluation                           │
│  Auditor.audit(input_data)                                 │
│                                                             │
│  Input:  audit_input dict (30+ fields)                     │
│  LLM:    1 batched call, 7 policies, group "auditoria_medica" │
│  Output: {<policy_id>: PolicyResult, ...} × 7               │
│          + AuditMetrics                                     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  ALGORITHM 2 — Recommendation Synthesis                    │
│  _run_recommendation_sync(input_data, policy_results)      │
│                                                             │
│  Input:  {transfer_data, policy_results}                   │
│  LLM:    1 call with system prompt + business rules        │
│  Output: {recommendation, reasoning, confidence,           │
│           downgrade_to}                                    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Persist to SQLite           │
│  audit_runs + policy_results │
│  + trail + api_calls         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Human Auditor Review        │
│  (React frontend)            │
│  → accept / reject /         │
│    downgrade / ask_just      │
└─────────────────────────────┘
```

---

## Key Files Reference

| File | Role |
|------|------|
| `auditor/src/auditor/core.py` | Algorithm 1 engine — policy loading, batching, LLM calling, logprob confidence |
| `auditor/src/auditor/models.py` | Data models: `Policy`, `PolicyResult`, `PolicyStatus`, `AuditMetrics`, `TraceInfo` |
| `auditor/src/auditor/backends.py` | LLM backend protocol + default httpx client |
| `auditor/policies.json` | 7 policy definitions with prompt templates |
| `backend/src/services/auditor_service.py` | Orchestrator — wires Algorithm 1 + Algorithm 2 + persistence |
| `backend/src/services/audit_input_builder.py` | Transforms extraction output into audit_input shape |
| `backend/schemas/audit_input.json` | Canonical audit_input JSON Schema (30+ fields) |
| `frontend/src/types/audit.ts` | TypeScript types for audit data (frontend contract) |