"""Spec-layer gate tests: one YAML file -> a live exam -> a score.

These run only where all five modules exist (the composition gate / the
merged tree). They pin three things:

1. Compilation is pure naming — the compiled exam behaves exactly like a
   hand-written one (same scorer repr, same fitness on the same system).
2. The fingerprint laws — identity is a property of the exam, not of the
   run: formatting/comments and the `system:`/`run:` sections never affect
   it; touching any exam component does.
3. `describe()` is a complete, JSON-able GUI contract — the frontend
   renders forms from it without parsing Python.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchy.core import SchemaViolation
from benchy.spec import compile_exam, compile_scoring, compile_system, describe, fingerprint, load_doc

pytestmark = pytest.mark.integration

EXAM_SPEC = """
spec_version: 1
exam:
  name: spec-gate
  task:
    freeform:
      ontology: qa/general/en
  data:
    spec: jsonl:samples.jsonl
    expected: expected
  scoring:
    exact_match: {}
system:
  url: echo:spec-gate
  text: "42"
run:
  max_concurrency: 4
"""


def _write_samples(d) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples.jsonl").write_text(
        json.dumps({"id": "1", "text": "q", "expected": "42"}) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("spec_gate")
    _write_samples(d)
    (d / "bench.yaml").write_text(EXAM_SPEC, encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def compiled(workdir):
    return compile_exam(workdir / "bench.yaml")


class TestCompile:
    def test_the_whole_yaml_file_compiles_to_a_live_benchmark(self, compiled):
        from benchy.benchmark import Benchmark

        assert isinstance(compiled, Benchmark)
        assert len(list(compiled.data)) == 1

    def test_data_rows_map_by_declared_field(self, compiled):
        samples = list(compiled.data)
        assert samples[0].expected == "42"

    def test_the_compiled_exam_runs_and_scores_end_to_end(self, compiled):
        system = compile_system(EXAM_SPEC)
        report = asyncio.run(compiled.run(system))
        assert report.fitness == 1.0
        assert report.n_errors == 0

    def test_compilation_is_pure_naming_same_repr_as_handwritten(self, workdir):
        from benchy.data import load as dload
        from benchy.scoring import exact_match
        from benchy.task import builtin

        handwritten_scoring = exact_match()
        bench = compile_exam(workdir / "bench.yaml")
        assert repr(bench.scoring) == repr(handwritten_scoring)


class TestScoringLanguage:
    def test_nested_scorer_trees_compile(self):
        sc = compile_scoring({"binary": {"inner": {"exact_match": {}}, "cutoff": 0.5}})
        assert repr(sc).startswith("binary(")

    def test_annotation_discipline_data_kwargs_stay_data(self):
        """`weights` maps onto a plain mapping parameter: even though
        `invert` is a real factory name, the value must NOT resolve."""
        sc = compile_scoring({
            "field_wise_weighted": {
                "fields": ["a", "b"],
                "per_field": {"exact_match": {}},
                "weights": {"a": 1.0, "b": 2.0},
            }
        })
        assert repr(sc).startswith("field_wise_weighted(")


class TestFingerprintLaws:
    def test_formatting_and_comments_do_not_change_it(self, workdir):
        noisy = (workdir / "bench.yaml").read_text(encoding="utf-8") + "\n# trailing comment\n"
        assert fingerprint(noisy) == fingerprint(workdir / "bench.yaml")

    def test_system_and_run_sections_do_not_change_it(self, workdir):
        doc = load_doc(workdir / "bench.yaml")
        doc["system"] = {"url": "echo:not-42"}
        doc["run"] = {"max_concurrency": 100}
        assert fingerprint(doc) == fingerprint(workdir / "bench.yaml")

    def test_touching_the_exam_changes_it(self, workdir):
        doc = load_doc(workdir / "bench.yaml")
        doc["exam"]["scoring"] = {"contains": {}}
        assert fingerprint(doc) != fingerprint(workdir / "bench.yaml")

    def test_editing_the_exam_name_does_not_change_it(self, workdir):
        """`exam.name` is a label, not semantics: renaming an exam keeps
        its identity — old scores stay comparable."""
        doc = load_doc(workdir / "bench.yaml")
        doc["exam"]["name"] = "renamed-v2"
        assert fingerprint(doc) == fingerprint(workdir / "bench.yaml")


class TestDescribe:
    def test_describe_is_jsonable_and_complete(self):
        d = describe()
        json.dumps(d)  # raises if anything non-JSONable
        assert d["spec_version"] == 1
        for sec in ("task", "scoring", "system", "data"):
            assert sec in d
        assert {"freeform", "structured_extraction", "classification", "transcription"} <= set(d["task"])
        assert "exact_match" in d["scoring"]
        assert "echo" in d["system"]["schemes"]
        assert "jsonl" in d["data"]["sources"]

    def test_describe_exposes_scorer_typed_params_for_the_gui(self):
        d = describe()
        fw = d["scoring"]["field_wise"]
        per_field = next(p for p in fw["params"] if p["name"] == "per_field")
        assert per_field["scorer_typed"] is True
        cut = next(p for p in d["scoring"]["binary"]["params"] if p["name"] == "cutoff")
        assert cut["scorer_typed"] is False


class TestErrors:
    def test_unknown_scorer_lists_known_names(self):
        with pytest.raises(SchemaViolation, match="known:"):
            compile_scoring({"no_such_scorer": {}})

    def test_unknown_task_lists_known_names(self):
        with pytest.raises(SchemaViolation, match="known:"):
            compile_exam({"spec_version": 1, "exam": {
                "task": {"no_such_task": {}},
                "data": {"spec": "jsonl:x.jsonl"},
                "scoring": {"exact_match": {}},
            }})

    def test_missing_exam_section(self):
        with pytest.raises(SchemaViolation, match="exam"):
            fingerprint({"spec_version": 1})

    def test_missing_data_spec(self):
        with pytest.raises(SchemaViolation, match="spec"):
            compile_exam({"spec_version": 1, "exam": {
                "task": {"freeform": {"ontology": "qa/general/en"}},
                "data": {"path": "nope.jsonl"},
                "scoring": {"exact_match": {}},
            }})

    def test_bad_version_is_rejected(self):
        with pytest.raises(SchemaViolation, match="spec_version"):
            load_doc({"spec_version": 2, "exam": {}})


# ---------------------------------------------------------------------------
# spec -> objects -> run: the whole point
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_yaml_to_score_in_one_breath(self, tmp_path):
        _write_samples(tmp_path)
        spec = f"""
spec_version: 1
exam:
  task: {{freeform: {{ontology: qa/general/en}}}}
  data: {{spec: jsonl:{tmp_path}/samples.jsonl, expected: expected}}
  scoring: {{exact_match: {{}}}}
system: {{url: echo:gate, text: "42"}}
"""
        bench = compile_exam(spec)
        system = compile_system(spec)
        report = asyncio.run(bench.run(system))
        assert report.fitness == 1.0
        assert report.n_errors == 0