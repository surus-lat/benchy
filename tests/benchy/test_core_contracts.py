"""The spine is frozen. These tests are the lock.

Every parallel worktree builds against benchy.core. If one of these fails,
two modules have silently disagreed about the contract.
"""

from __future__ import annotations

import dataclasses

import pytest

from benchy.core import (
    AudioPart,
    Capabilities,
    ImagePart,
    Message,
    OntologyPath,
    Prediction,
    Record,
    Report,
    Request,
    Response,
    Sample,
    Score,
    TextPart,
)


class TestOntologyPath:
    def test_parses_all_three_levels(self):
        p = OntologyPath.parse("transcription/fleurs/pt-BR")
        assert (p.task, p.domain, p.language) == ("transcription", "fleurs", "pt-BR")

    @pytest.mark.parametrize(
        "raw", ["transcription", "transcription/fleurs", "transcription/fleurs/pt-BR"]
    )
    def test_round_trips_through_str(self, raw):
        assert str(OntologyPath.parse(raw)) == raw

    def test_tolerates_surrounding_slashes(self):
        assert str(OntologyPath.parse("/transcription/fleurs/")) == "transcription/fleurs"

    def test_rejects_more_than_three_segments(self):
        with pytest.raises(ValueError):
            OntologyPath.parse("a/b/c/d")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            OntologyPath.parse("")

    def test_rejects_language_without_domain(self):
        with pytest.raises(ValueError):
            OntologyPath(task="transcription", domain=None, language="pt-BR")

    def test_prefix_matching_walks_the_ontology(self):
        root = OntologyPath.parse("transcription")
        leaf = OntologyPath.parse("transcription/fleurs/pt-BR")
        assert root.is_prefix_of(leaf)
        assert not leaf.is_prefix_of(root)

    def test_prefix_matching_is_segment_wise_not_string_wise(self):
        assert not OntologyPath.parse("trans").is_prefix_of(OntologyPath.parse("transcription"))

    def test_is_hashable_so_it_can_key_a_registry(self):
        assert len({OntologyPath.parse("a/b"), OntologyPath.parse("a/b")}) == 1


class TestContentParts:
    def test_message_text_helper_builds_a_single_text_part(self):
        m = Message.text("user", "hola")
        assert m.role == "user"
        assert m.parts == (TextPart("hola"),)

    def test_parts_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            TextPart("x").text = "y"  # type: ignore[misc]

    def test_image_and_audio_parts_carry_a_mime_default(self):
        assert ImagePart(url="http://x/y.png").mime == "image/png"
        assert AudioPart(path="/tmp/a.wav").mime == "audio/wav"


class TestCapabilities:
    def test_text_only_system_rejects_audio(self):
        caps = Capabilities()
        assert caps.accepts(TextPart("x"))
        assert not caps.accepts(AudioPart(path="/tmp/a.wav"))

    def test_audio_system_accepts_audio(self):
        caps = Capabilities(audio_in=True)
        assert caps.accepts(AudioPart(path="/tmp/a.wav"))

    def test_the_field_set_is_cut_to_the_metal_and_locked(self):
        """The Cutting Theorem, enforced: a Capabilities field exists iff
        exam rendering or grading branches on it. Adding a sixth field is a
        deliberate act that must update this test and cite its branch."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(Capabilities)}
        assert fields == {
            "text_in", "image_in", "audio_in",
            "structured_output", "max_concurrency",
        }


class TestResponse:
    def test_ok_is_false_when_error_is_set(self):
        assert Response(text="hi").ok
        assert not Response(error="boom").ok

    def test_response_has_no_souvenir_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(Response)}
        assert fields == {"text", "data", "usage", "latency_ms", "error"}

    def test_aclose_is_not_part_of_the_system_protocol(self):
        from benchy.core import System

        class Minimal:
            url = "fake:1"
            capabilities = Capabilities()

            async def invoke(self, request): ...

        assert isinstance(Minimal(), System)


class TestValueObjectDefaults:
    def test_sample_defaults_are_not_shared(self):
        a, b = Sample(id="a", input={}), Sample(id="b", input={})
        assert a.meta == {} and b.meta == {}
        assert a.meta is not b.meta

    def test_request_requires_only_messages(self):
        r = Request(messages=(Message.text("user", "hi"),))
        assert r.output_schema is None and r.params == {}

    def test_prediction_defaults_to_parsed_ok(self):
        assert Prediction(value={"a": 1}).parse_ok

    def test_report_defaults_are_empty_not_none(self):
        rep = Report(benchmark="b", system="s", scorer="x", fitness=0.0, aggregate={})
        assert rep.records == () and rep.n_samples == 0

    def test_record_tolerates_a_total_failure(self):
        rec = Record(sample_id="1", prediction=None, score=None, error="timeout")
        assert rec.error == "timeout"

    def test_score_carries_a_breakdown(self):
        s = Score(value=0.5, breakdown={"vendor": 1.0, "total": 0.0}, scorer="field_wise")
        assert s.breakdown["vendor"] == 1.0


class TestProtocolsAreStructural:
    """A duck-typed object must satisfy the protocols — no inheritance."""

    def test_system_protocol_is_runtime_checkable(self):
        from benchy.core import System

        class Duck:
            url = "fake:1"
            capabilities = Capabilities()

            async def invoke(self, request): ...

        assert isinstance(Duck(), System)

        # The protocol does not mandate `aclose`: lifecycle belongs to the
        # system's owner (the caller), not to the exam. A duck with an
        # aclose is still a System; a duck without one is too.
        class DuckWithClose(Duck):
            async def aclose(self): ...

        assert isinstance(DuckWithClose(), System)

    def test_scorer_protocol_is_runtime_checkable(self):
        from benchy.core import Scorer

        class Duck:
            name = "x"

            def evaluate(self, prediction, expected, sample): ...
            def fitness(self, prediction, expected, sample): ...
            def aggregate(self, scores): ...

        assert isinstance(Duck(), Scorer)


class TestSpineIsExhaustivelyMinimal:
    """Minimality by exhaustion: every public name in the spine must be
    citable under the Cutting Theorem (branch / consumer / minimal wire).
    This census is the proof obligation; if a name loses all its citations,
    delete it here first, then in core.py."""

    def test_the_public_surface_census(self):
        import benchy.core as core

        assert set(core.__all__) == {
            # ontology
            "OntologyPath",
            # content
            "TextPart", "ImagePart", "AudioPart", "Part", "Message", "Role",
            # io
            "Sample", "Request", "Response", "Usage", "Prediction",
            # capability
            "Capabilities",
            # protocols
            "Task", "Scorer", "System", "Data",
            # results
            "Score", "Record", "Report",
            # types
            "LossFn", "SystemLoader",
            # errors
            "BenchyError", "LoadError", "SchemaViolation",
            "SystemFailure", "CapabilityError",
        }
