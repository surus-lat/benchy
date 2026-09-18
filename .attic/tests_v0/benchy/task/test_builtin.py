"""The four built-in Task shapes: transcription, structured_extraction,
classification, freeform.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from benchy.core import (
    AudioPart,
    Capabilities,
    CapabilityError,
    Response,
    Sample,
    Task as TaskProto,
)
from benchy.task import builtin


class TestFreeform:
    def test_is_a_core_task(self):
        assert isinstance(builtin.freeform(ontology="qa/general/en"), TaskProto)

    def test_text_in_text_out_no_object_wrapper(self):
        task = builtin.freeform(ontology="qa/general/en")
        assert task.output_schema == {"type": "string"}

    def test_round_trip(self):
        task = builtin.freeform(ontology="qa/general/en")
        sample = Sample(id="1", input={"text": "what is 6*7?"}, expected="42")
        req = task.render(sample, Capabilities())
        pred = task.parse(Response(text="42"), Capabilities())
        assert pred.parse_ok
        assert pred.value == "42"
        assert req.meta["sample_id"] == "1"

    def test_no_json_schema_ever_requested(self):
        task = builtin.freeform(ontology="qa/general/en")
        sample = Sample(id="1", input={"text": "hi"})
        req = task.render(sample, Capabilities(structured_output=True))
        assert req.output_schema is None


class TestTranscription:
    def test_is_a_core_task(self):
        assert isinstance(builtin.transcription(ontology="transcription/fleurs/pt-BR"), TaskProto)

    def test_output_is_an_object_with_a_text_field(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        assert task.output_schema == {
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
        }

    def test_renders_an_audio_part(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        sample = Sample(id="1", input={"audio_path": "/data/fleurs/1.wav"})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert len(parts) == 1 and parts[0].path == "/data/fleurs/1.wav"

    def test_parses_plain_text_no_json_involved(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        pred = task.parse(Response(text="ola mundo"), Capabilities(audio_in=True))
        assert pred.parse_ok
        assert pred.value == {"text": "ola mundo"}

    def test_raises_capability_error_without_audio_in(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        sample = Sample(id="1", input={"audio_path": "/data/fleurs/1.wav"})
        with pytest.raises(CapabilityError):
            task.render(sample, Capabilities(audio_in=False))

    def test_accepts_bare_path_string_shape(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        sample = Sample(id="1", input={"audio": "/data/fleurs/1.wav"})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert parts[0].path == "/data/fleurs/1.wav"

    def test_accepts_hf_audio_feature_dict_shape(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        array = np.zeros(1600, dtype="float32")
        sample = Sample(id="1", input={"audio": {"path": "1.wav", "array": array, "sampling_rate": 16000}})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert parts[0].data is not None and parts[0].sample_rate == 16000

    def test_language_flows_into_default_instructions(self):
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR", language="Portuguese")
        assert "Portuguese" in task.instructions

    def test_fleurs_manifest_shape_from_the_data_worktree(self):
        """Matches `.data/transcription/{pt_br,es_419}/manifest.jsonl` rows."""
        task = builtin.transcription(ontology="transcription/fleurs/pt-BR")
        row = {"id": "1", "audio_path": "/data/1.wav", "expected": "ola", "gender": "female", "language": "pt-BR"}
        sample = Sample(id=row["id"], input={"audio_path": row["audio_path"]}, expected=row["expected"],
                         meta={"gender": row["gender"], "language": row["language"]})
        req = task.render(sample, Capabilities(audio_in=True))
        assert req.meta["sample_id"] == "1"
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert parts[0].path == "/data/1.wav"


class Negocio(BaseModel):
    descripcion_negocio: str
    meses_en_negocio: int
    cantidad_empleados: int


class LeadOutput(BaseModel):
    nombre: str
    tiene_negocio: bool
    negocio: Negocio | None = None


class TestStructuredExtraction:
    def test_is_a_core_task(self):
        task = builtin.structured_extraction(
            output={"type": "object", "properties": {"name": {"type": "string"}}},
            ontology="structured_extraction/leads/es-AR",
        )
        assert isinstance(task, TaskProto)

    def test_accepts_a_raw_dict_schema_for_output(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"], "additionalProperties": False}
        task = builtin.structured_extraction(
            output=schema, ontology="structured_extraction/leads/es-AR", instructions="Extract the name.",
        )
        assert task.output_schema == schema

    def test_default_input_is_permissive(self):
        task = builtin.structured_extraction(
            output={"type": "object", "properties": {"name": {"type": "string"}}},
            ontology="structured_extraction/leads/es-AR",
        )
        task.validate_sample(Sample(id="1", input={"text": "soy Ana"}))  # must not raise

    def test_accepts_a_pydantic_model_for_output(self):
        task = builtin.structured_extraction(output=LeadOutput, ontology="structured_extraction/leads/es-AR")
        assert task.output_schema["properties"]["nombre"]["type"] == "string"

    def test_nested_schema_round_trips_through_prosaic_repair(self):
        """The real reference dataset (chat_extract_data.jsonl) is nested."""
        task = builtin.structured_extraction(output=LeadOutput, ontology="structured_extraction/leads/es-AR")
        raw = (
            'Aqui esta la extraccion:\n```json\n'
            '{"nombre": "Ana", "tiene_negocio": true, '
            '"negocio": {"descripcion_negocio": "panaderia", '
            '"meses_en_negocio": 12, "cantidad_empleados": 3}}\n```'
        )
        pred = task.parse(Response(text=raw), Capabilities(structured_output=False))
        assert pred.parse_ok, pred.parse_error
        assert pred.value["negocio"]["cantidad_empleados"] == 3


class TestClassification:
    LABELS = ["urgente", "normal", "bajo"]

    def test_is_a_core_task(self):
        task = builtin.classification(self.LABELS, ontology="triage/tickets/es-AR")
        assert isinstance(task, TaskProto)

    def test_output_schema_is_an_enum(self):
        task = builtin.classification(self.LABELS, ontology="triage/tickets/es-AR")
        assert task.output_schema == {"type": "string", "enum": self.LABELS}

    def test_renders_lettered_choices_when_not_structured(self):
        task = builtin.classification(self.LABELS, ontology="triage/tickets/es-AR")
        sample = Sample(id="1", input={"text": "el servidor esta caido"})
        req = task.render(sample, Capabilities(structured_output=False))
        rendered = "\n".join(p.text for m in req.messages for p in m.parts if hasattr(p, "text"))
        assert "urgente" in rendered and "A." in rendered

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("A", "urgente"),
            ("Respuesta: normal", "normal"),
            ("2", "bajo"),
            ('{"label": "normal"}', "normal"),
            ("Creo que es urgente.", "urgente"),
        ],
    )
    def test_tolerant_parsing(self, raw, expected):
        task = builtin.classification(self.LABELS, ontology="triage/tickets/es-AR")
        pred = task.parse(Response(text=raw), Capabilities())
        assert pred.parse_ok and pred.value == expected

    def test_unresolvable_response_fails_without_raising(self):
        task = builtin.classification(self.LABELS, ontology="triage/tickets/es-AR")
        pred = task.parse(Response(text="I cannot determine this."), Capabilities())
        assert not pred.parse_ok

    def test_native_structured_output_reads_response_data(self):
        task = builtin.classification(self.LABELS, ontology="triage/tickets/es-AR")
        pred = task.parse(Response(data="normal"), Capabilities(structured_output=True))
        assert pred.parse_ok and pred.value == "normal"

    def test_requires_nonempty_labels(self):
        from benchy.task import Task

        with pytest.raises(ValueError):
            Task(name="x", ontology="x", mode="choice", labels=[])
