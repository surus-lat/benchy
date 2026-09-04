"""The generic `Task(...)` authoring surface.

Tests against hand-built `Capabilities`/`Response` objects from
`benchy.core` -- that is the entire cross-worktree contract, and it is
frozen, so these stay valid independent of the sibling worktrees.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from benchy.core import (
    AudioPart,
    Capabilities,
    CapabilityError,
    ImagePart,
    Prediction,
    Request,
    Sample,
    SchemaViolation,
    Task as TaskProto,
    TextPart,
)
from benchy.task import Task


class InvoiceInput(BaseModel):
    text: str


class InvoiceExtraction(BaseModel):
    vendor: str
    total: float


class TestConstructorAcceptsPydanticOrDict:
    def test_pydantic_models_compile_to_json_schema(self):
        task = Task(
            name="invoice_extraction",
            ontology="image_extraction/invoices/es-AR",
            input=InvoiceInput,
            output=InvoiceExtraction,
            instructions="Extract the fields from this Argentine invoice.",
        )
        assert task.input_schema["properties"]["text"]["type"] == "string"
        assert task.output_schema["properties"]["vendor"]["type"] == "string"
        assert task.output_schema["properties"]["total"]["type"] == "number"

    def test_raw_json_schema_dicts_work_identically(self):
        task = Task(
            name="leads",
            ontology="structured_extraction/leads/es-AR",
            input={"type": "object", "properties": {"text": {"type": "string"}}},
            output={"type": "object", "properties": {"name": {"type": "string"}},
                    "required": ["name"], "additionalProperties": False},
        )
        assert task.output_schema["required"] == ["name"]

    def test_none_input_output_resolve_to_a_permissive_schema(self):
        task = Task(name="x", ontology="x")
        assert task.input_schema == {"type": "object"}
        assert task.output_schema == {"type": "object"}

    def test_rejects_unsupported_spec_types(self):
        with pytest.raises(TypeError):
            Task(name="x", ontology="x", input=42)


class TestProtocolConformance:
    def test_isinstance_of_core_task_protocol(self):
        task = Task(name="x", ontology="x", output={"type": "string"})
        assert isinstance(task, TaskProto)

    def test_ontology_accepts_a_string(self):
        task = Task(name="x", ontology="a/b/c")
        assert str(task.ontology) == "a/b/c"


class TestRenderNegotiation:
    """The render()/parse() capability matrix -- the module's core promise."""

    SCHEMA = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
        "required": ["vendor", "total"],
    }

    def make_task(self) -> Task:
        return Task(
            name="invoice_extraction",
            ontology="image_extraction/invoices/es-AR",
            input={"type": "object", "properties": {"text": {"type": "string"}}},
            output=self.SCHEMA,
            instructions="Extract vendor and total.",
        )

    def test_structured_true_sets_request_output_schema(self):
        task = self.make_task()
        sample = Sample(id="s1", input={"text": "Factura de Acme por $100"})
        req = task.render(sample, Capabilities(structured_output=True))
        assert isinstance(req, Request)
        assert req.output_schema == self.SCHEMA

    def test_structured_false_leaves_output_schema_none_and_embeds_schema_in_prompt(self):
        task = self.make_task()
        sample = Sample(id="s1", input={"text": "Factura de Acme por $100"})
        req = task.render(sample, Capabilities(structured_output=False))
        assert req.output_schema is None
        rendered = "\n".join(p.text for m in req.messages for p in m.parts if isinstance(p, TextPart))
        assert "vendor" in rendered and "total" in rendered
        assert "JSON" in rendered

    def test_renders_differently_for_the_two_capability_states(self):
        task = self.make_task()
        sample = Sample(id="s1", input={"text": "Factura de Acme por $100"})
        req_native = task.render(sample, Capabilities(structured_output=True))
        req_prompted = task.render(sample, Capabilities(structured_output=False))
        native_text = "\n".join(p.text for m in req_native.messages for p in m.parts if isinstance(p, TextPart))
        prompted_text = "\n".join(p.text for m in req_prompted.messages for p in m.parts if isinstance(p, TextPart))
        assert native_text != prompted_text
        assert "vendor" not in native_text or "vendor" in prompted_text  # prompted always has it

    def test_both_capability_states_round_trip_a_well_formed_answer_to_the_same_value(self):
        from benchy.core import Response

        task = self.make_task()
        expected_value = {"vendor": "Acme", "total": 100.0}

        native_pred = task.parse(Response(data=expected_value), Capabilities(structured_output=True))
        prompted_pred = task.parse(
            Response(text='{"vendor": "Acme", "total": 100.0}'), Capabilities(structured_output=False)
        )
        assert native_pred.parse_ok and prompted_pred.parse_ok
        assert native_pred.value == prompted_pred.value == expected_value

    def test_sample_id_is_always_stamped_into_request_meta(self):
        task = self.make_task()
        sample = Sample(id="sample-42", input={"text": "x"})
        req = task.render(sample, Capabilities())
        assert req.meta["sample_id"] == "sample-42"

    def test_custom_render_fn_cannot_break_the_sample_id_contract(self):
        def bad_render(sample, caps, task):
            return Request(messages=(), meta={"sample_id": "WRONG"})

        task = Task(name="x", ontology="x", render_fn=bad_render)
        sample = Sample(id="real-id", input={})
        req = task.render(sample, Capabilities())
        assert req.meta["sample_id"] == "real-id"

    def test_custom_render_fn_is_used_verbatim_otherwise(self):
        def custom_render(sample, caps, task):
            return Request(messages=(TextPart("hi"),), meta={})  # deliberately odd shape

        task = Task(name="x", ontology="x", render_fn=custom_render)
        sample = Sample(id="1", input={})
        req = task.render(sample, Capabilities())
        assert req.messages == (TextPart("hi"),)

    def test_custom_parse_fn_overrides_default_parsing(self):
        from benchy.core import Response

        task = Task(name="x", ontology="x", parse_fn=lambda r, c, t: Prediction(value="always this"))
        pred = task.parse(Response(text="anything"), Capabilities())
        assert pred.value == "always this"


class TestAudioCapabilityNegotiation:
    def make_audio_task(self) -> Task:
        return Task(
            name="asr",
            ontology="transcription/fleurs/pt-BR",
            input={"type": "object", "properties": {"audio": {}}},
            output={"type": "object", "properties": {"text": {"type": "string"}}},
            mode="text",
        )

    def test_raises_capability_error_on_text_only_system(self):
        task = self.make_audio_task()
        sample = Sample(id="1", input={"audio": "/tmp/a.wav"})
        with pytest.raises(CapabilityError) as exc_info:
            task.render(sample, Capabilities(audio_in=False))
        message = str(exc_info.value)
        assert "audio_in" in message
        assert "asr" in message

    def test_succeeds_on_audio_capable_system_with_a_bare_path(self):
        task = self.make_audio_task()
        sample = Sample(id="1", input={"audio": "/tmp/a.wav"})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts]
        audio_parts = [p for p in parts if isinstance(p, AudioPart)]
        assert len(audio_parts) == 1
        assert audio_parts[0].path == "/tmp/a.wav"

    def test_succeeds_with_audio_path_key_instead_of_audio(self):
        task = self.make_audio_task()
        sample = Sample(id="1", input={"audio_path": "/tmp/b.wav"})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert parts[0].path == "/tmp/b.wav"

    def test_succeeds_with_the_hf_audio_feature_dict_shape(self):
        import numpy as np

        task = self.make_audio_task()
        array = np.zeros(1600, dtype="float32")
        sample = Sample(id="1", input={"audio": {"path": "orig.wav", "array": array, "sampling_rate": 16000}})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert len(parts) == 1
        assert parts[0].data is not None
        assert parts[0].sample_rate == 16000

    def test_hf_dict_without_array_falls_back_to_path(self):
        task = self.make_audio_task()
        sample = Sample(id="1", input={"audio": {"path": "/tmp/c.wav", "sampling_rate": 16000}})
        req = task.render(sample, Capabilities(audio_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, AudioPart)]
        assert parts[0].path == "/tmp/c.wav"

    def test_unusable_audio_value_raises_schema_violation_not_capability_error(self):
        task = self.make_audio_task()
        sample = Sample(id="1", input={"audio": 12345})
        with pytest.raises(SchemaViolation):
            task.render(sample, Capabilities(audio_in=True))


class TestImageCapabilityNegotiation:
    def make_image_task(self) -> Task:
        return Task(
            name="invoice_extraction",
            ontology="image_extraction/invoices/es-AR",
            input={"type": "object", "properties": {"image": {}}},
            output={"type": "object", "properties": {"vendor": {"type": "string"}}},
        )

    def test_raises_capability_error_on_a_text_only_system(self):
        task = self.make_image_task()
        sample = Sample(id="1", input={"image": "/tmp/invoice.png"})
        with pytest.raises(CapabilityError) as exc_info:
            task.render(sample, Capabilities(image_in=False))
        assert "image_in" in str(exc_info.value)

    def test_succeeds_on_image_capable_system(self):
        task = self.make_image_task()
        sample = Sample(id="1", input={"image": "/tmp/invoice.png"})
        req = task.render(sample, Capabilities(image_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, ImagePart)]
        assert parts[0].path == "/tmp/invoice.png"

    def test_succeeds_with_raw_bytes(self):
        task = self.make_image_task()
        sample = Sample(id="1", input={"image": b"\x89PNG\r\n"})
        req = task.render(sample, Capabilities(image_in=True))
        parts = [p for m in req.messages for p in m.parts if isinstance(p, ImagePart)]
        assert parts[0].data == b"\x89PNG\r\n"


class TestValidateSample:
    def test_passes_a_conforming_sample(self):
        task = Task(
            name="x", ontology="x",
            input={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        )
        task.validate_sample(Sample(id="1", input={"text": "hi"}))  # must not raise

    def test_raises_schema_violation_on_missing_required_field(self):
        task = Task(
            name="x", ontology="x",
            input={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        )
        with pytest.raises(SchemaViolation):
            task.validate_sample(Sample(id="1", input={}))

    def test_raises_schema_violation_on_wrong_type(self):
        task = Task(
            name="x", ontology="x",
            input={"type": "object", "properties": {"total": {"type": "number"}}},
        )
        with pytest.raises(SchemaViolation):
            task.validate_sample(Sample(id="1", input={"total": "not a number"}))

    def test_permissive_default_schema_accepts_anything(self):
        task = Task(name="x", ontology="x")
        task.validate_sample(Sample(id="1", input={"whatever": 1, "extra": [1, 2, 3]}))

    def test_media_fields_are_exempt_from_structural_validation(self):
        task = Task(
            name="x", ontology="x",
            input={"type": "object", "properties": {"audio": {"type": "string"}}, "required": ["audio"]},
        )
        import numpy as np

        # A numpy array is not a valid instance of {"type": "string"}, but
        # validate_sample must not reject it -- media fields are checked for
        # presence only, not JSON-Schema-validated.
        task.validate_sample(Sample(id="1", input={"audio": {"array": np.zeros(10), "sampling_rate": 16000}}))
