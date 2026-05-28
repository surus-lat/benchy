# Task Catalog

Complete reference for all task groups and subtasks available in Benchy. Use task names
with the `--tasks` CLI flag or in model config YAML files.

## Task group reference syntax

```bash
# Run all subtasks in a group
--tasks spanish

# Run one specific subtask
--tasks spanish.copa_es

# Run multiple groups
--tasks spanish portuguese translation

# Use a predefined task group alias
--tasks latam_board
```

---

## Task group aliases (from `configs/config.yaml`)

| Alias | Expands to |
|-------|-----------|
| `latam_board` | `spanish`, `portuguese`, `translation`, `structured_extraction` |
| `structured_only` | `structured_extraction` |
| `image_extraction_only` | `image_extraction` |

---

## `spanish` — Spanish language tasks

**Primary metric:** accuracy  
**Capability requirements:** none (text only)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `copa_es` | COPA-ES | Causal reasoning — choose the most plausible cause or effect |
| `escola` | ESCOLA | Reading comprehension and cloze completion |
| `mgsm_direct_es_spanish_bench` | MGSM | Multilingual grade-school math (direct answer format) |
| `openbookqa_es` | OpenBookQA-ES | Science fact recall with open-book knowledge |
| `paws_es_spanish_bench` | PAWS-ES | Paraphrase detection (same/different meaning) |
| `teleia_cervantes_ave` | Teleia Cervantes AVE | Analogical reasoning in Spanish |
| `teleia_pce` | Teleia PCE | Reading comprehension — Spanish university entrance exam |
| `teleia_siele` | Teleia SIELE | Spanish language proficiency (SIELE exam format) |
| `wnli_es` | WNLI-ES | Winograd NLI — coreference-dependent inference |
| `xnli_es_spanish_bench` | XNLI-ES | Cross-lingual natural language inference |

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks spanish --limit 10
benchy eval --provider openai --model-name gpt-4o-mini --tasks spanish.copa_es --limit 5
```

---

## `portuguese` — Portuguese language tasks

**Primary metric:** accuracy / Pearson correlation (task-dependent)  
**Capability requirements:** none (text only)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `assin2_rte` | ASSIN2 RTE | Recognizing Textual Entailment in Portuguese |
| `assin2_sts` | ASSIN2 STS | Semantic Textual Similarity in Portuguese (Pearson correlation) |
| `bluex` | BlueX | Reading comprehension — Brazilian college entrance exam |
| `enem_challenge` | ENEM | High-school knowledge across subjects (Brazilian national exam) |
| `oab_exams` | OAB Exams | Brazilian Bar Association exam — legal reasoning |

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks portuguese --limit 10
```

---

## `translation` — Translation tasks

**Primary metric:** COMET score  
**Secondary metrics:** ChrF, BLEU  
**Capability requirements:** none (text only)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `flores` | FLORES-200 | Translation quality across many language pairs (FLORES evaluation set) |
| `opus` | OPUS | Translation quality using OPUS parallel corpora |

COMET scores range 0–1 (higher is better). ChrF and BLEU scores range 0–100.

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks translation --limit 10
```

---

## `structured_extraction` — JSON extraction tasks

**Primary metric:** `extraction_quality_score` / `document_extraction_score`  
**Capability requirements:** `supports_schema` (preferred), `supports_multimodal` (optional)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `chat_extract` | Chat Extract | Structured information extraction from chat/conversation text |
| `email_extract` | Email Extract | Key field extraction from email messages |
| `paraloq` | Paraloq | Complex structured extraction from documents |

Scores reflect field-level F1 accuracy, schema validity, and hallucination rate. See
`docs/SCORING.md` for metric definitions.

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks structured_extraction --limit 10
```

---

## `image_extraction` — Vision-language extraction tasks

**Primary metric:** `extraction_quality_score` / `document_extraction_score`  
**Capability requirements:** `supports_multimodal` (required), `supports_schema` (preferred)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `facturas` | Argentine invoices (images) | Structured extraction from invoice images |

Tasks in this group require multimodal support. The task is automatically skipped
on text-only models with a clear log message.

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks image_extraction --limit 5
benchy eval --provider google --model-name gemini-2.5-flash --tasks image_extraction --limit 5
```

---

## `document_extraction` — Document extraction tasks

**Primary metric:** `document_extraction_score`  
**Capability requirements:** `supports_multimodal` (required), `supports_schema` (preferred)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `facturas_argentinas` | Argentine invoices (PDFs) | Structured extraction from scanned/PDF invoices |

PDF documents are automatically rendered to PNG before being sent to multimodal models.
Use `--dataset` to select a different dataset or `--render-dpi` to control rendering quality.

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks document_extraction --limit 5
```

---

## `classify` — Classification tasks

**Primary metric:** accuracy  
**Capability requirements:** none (text only)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `diag_test` | Diagnostic test | Internal smoke/diagnostic test |
| `environmental_claims` | ClimateBERT Environmental Claims | Binary classification: is this an environmental claim? |
| `spanish_spam` | Spanish spam dataset | Binary classification: spam vs. ham (Spanish) |

**Usage:**
```bash
benchy eval --provider openai --model-name gpt-4o-mini --tasks classify --limit 10
benchy eval --provider openai --model-name gpt-4o-mini --tasks classify.environmental_claims --limit 10
```

---

## `image_manipulation` — Image manipulation tasks

**Primary metric:** image similarity score  
**Capability requirements:** `supports_multimodal` (required)

| Subtask | Dataset | What it measures |
|---------|---------|-----------------|
| `remove_background` | ICM57 | Background removal quality measured by image similarity metrics |

This task evaluates models that produce image outputs (not text). Currently supported
via Google Gemini image models.

**Usage:**
```bash
benchy eval --provider google --model-name gemini-2.5-flash-image \
  --tasks image_manipulation.remove_background --dataset ICM57 --limit 5

# Use a custom dataset from .data/
benchy eval --provider google --model-name gemini-2.5-flash-image \
  --tasks image_manipulation.remove_background --dataset my-dataset --limit 5
```

---

## Zero-code task types (CLI datasets)

In addition to the built-in task groups above, you can create tasks directly from CLI
without writing any Python:

| Task type | Flag | What it does |
|-----------|------|-------------|
| `classification` | `--task-type classification` | MCQ/binary classification |
| `structured` | `--task-type structured` | JSON structured extraction |
| `freeform` | `--task-type freeform` | Open-ended text generation |

**Usage:**
```bash
# Classification from HuggingFace
benchy eval --provider openai --model-name gpt-4o-mini \
  --task-type classification \
  --dataset-name climatebert/environmental_claims \
  --dataset-labels '{"0": "No", "1": "Yes"}' --limit 10

# Structured extraction from .data/
benchy eval --provider openai --model-name gpt-4o-mini \
  --task-type structured --dataset-name my-invoices --limit 5
```

See `docs/CLI_DATASET_USAGE.md` for full details.

---

## Custom datasets for existing tasks

Some tasks support multiple datasets. Use `--dataset` to select:

```bash
# Use a custom dataset for background removal
benchy eval --config surus-remove-background --dataset my-custom-dataset --limit 5

# Use the default ICM57 dataset
benchy eval --config surus-remove-background --dataset ICM57 --limit 5

# Override the dataset for any existing task
benchy eval --config my-model.yaml --tasks classify.environmental_claims \
  --dataset-name my-org/my-climate-dataset --dataset-split validation --limit 10
```

---

## Adding new tasks

To add a new task group to Benchy:

1. Create `src/tasks/<group_name>/` with `metadata.yaml` and one `.py` file per subtask
2. Each `.py` file defines a class extending the appropriate handler
3. The task is auto-discovered by convention — no registration needed

See `docs/contribute_tasks.md` for the full step-by-step guide.

---

## Related docs

- [Tutorial: Getting Started](tutorial-getting-started.md) — Run your first evaluation
- [CLI Reference](reference-cli.md) — All task-related flags
- [Handler System Guide](HANDLER_SYSTEM_GUIDE.md) — How to add tasks
- [Contributing Tasks](contribute_tasks.md) — Step-by-step task contribution
- [SCORING.md](SCORING.md) — Metric definitions and formulas
