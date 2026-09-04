"""Seam tests: the five modules must actually compose.

Written against the API contract handed to each worktree agent. If one of
these fails after the merge, two modules disagreed and the disagreement is
here, not inside either module's own green suite.
"""

from __future__ import annotations

import json

import pytest

from benchy.core import (
    Capabilities,
    Data as DataProto,
    Prediction,
    Request,
    Response,
    Sample,
    Scorer as ScorerProto,
    System as SystemProto,
    Task as TaskProto,
)

pytestmark = pytest.mark.seam


# --------------------------------------------------------------------------
# Seam 1: every module's product satisfies the frozen protocol
# --------------------------------------------------------------------------

class TestProtocolConformance:
    def test_scoring_produces_a_Scorer(self, mods):
        assert isinstance(mods["scoring"].exact_match(), ScorerProto)

    def test_system_produces_a_System(self, mods):
        assert isinstance(mods["system"].load("echo:"), SystemProto)

    def test_data_produces_a_Data(self, mods):
        d = mods["data"].Data.from_samples([Sample(id="1", input={"text": "x"}, expected="x")])
        assert isinstance(d, DataProto)

    def test_task_produces_a_Task(self, mods):
        assert isinstance(mods["task"].builtin.freeform(ontology="qa/general/en"), TaskProto)


# --------------------------------------------------------------------------
# Seam 2: Task <-> System. The render/parse bridge over an opaque system.
# --------------------------------------------------------------------------

class TestTaskSystemBridge:
    async def test_freeform_round_trip_through_echo(self, mods):
        task = mods["task"].builtin.freeform(ontology="qa/general/en")
        system = mods["system"].load("echo:", text="42")
        sample = Sample(id="1", input={"text": "what is 6*7?"}, expected="42")

        request = task.render(sample, system.capabilities)
        assert isinstance(request, Request)
        response = await system.invoke(request)
        assert isinstance(response, Response)
        prediction = task.parse(response, system.capabilities)
        assert isinstance(prediction, Prediction)
        assert prediction.parse_ok

    async def test_structured_task_uses_native_output_when_the_system_has_it(self, mods):
        """A capable system gets a schema-constrained request."""
        task = mods["task"].builtin.structured_extraction(
            output={"type": "object", "properties": {"name": {"type": "string"}},
                    "required": ["name"], "additionalProperties": False},
            ontology="structured_extraction/leads/es-AR",
            instructions="Extract the name.",
        )
        system = mods["system"].load(
            "echo:", data={"name": "Ana"},
            capabilities=Capabilities(structured_output=True),
        )
        sample = Sample(id="1", input={"text": "me llamo Ana"}, expected={"name": "Ana"})
        request = task.render(sample, system.capabilities)
        assert request.output_schema is not None, "capable system must get output_schema set"

        prediction = task.parse(await system.invoke(request), system.capabilities)
        assert prediction.parse_ok and prediction.value == {"name": "Ana"}

    async def test_structured_task_falls_back_to_prompt_encoding_and_repairs_prose(self, mods):
        """An incapable system gets the schema in the prompt and prose repaired."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"], "additionalProperties": False}
        task = mods["task"].builtin.structured_extraction(
            output=schema, ontology="structured_extraction/leads/es-AR",
            instructions="Extract the name.",
        )
        system = mods["system"].load(
            "echo:",
            text='Claro! Aqui tienes:\n```json\n{"name": "Ana"}\n```\nEspero que sirva.',
            capabilities=Capabilities(structured_output=False),
        )
        sample = Sample(id="1", input={"text": "me llamo Ana"}, expected={"name": "Ana"})
        request = task.render(sample, system.capabilities)
        assert request.output_schema is None or True  # either policy is fine
        rendered = "\n".join(
            p.text for m in request.messages for p in m.parts if hasattr(p, "text")
        )
        assert "name" in rendered, "schema must reach the prompt when output isn't native"

        prediction = task.parse(await system.invoke(request), system.capabilities)
        assert prediction.parse_ok, prediction.parse_error
        assert prediction.value == {"name": "Ana"}

    async def test_both_capability_paths_reach_the_same_value(self, mods):
        """The whole point of negotiation: the author sees one answer either way."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"], "additionalProperties": False}
        task = mods["task"].builtin.structured_extraction(
            output=schema, ontology="structured_extraction/leads/es-AR", instructions="Extract."
        )
        sample = Sample(id="1", input={"text": "soy Ana"}, expected={"name": "Ana"})

        native = mods["system"].load("echo:", data={"name": "Ana"},
                                     capabilities=Capabilities(structured_output=True))
        prosaic = mods["system"].load("echo:", text='{"name": "Ana"}',
                                      capabilities=Capabilities(structured_output=False))
        out = []
        for system in (native, prosaic):
            req = task.render(sample, system.capabilities)
            out.append(task.parse(await system.invoke(req), system.capabilities).value)
        assert out[0] == out[1]

    async def test_a_broken_response_yields_a_failed_prediction_not_an_exception(self, mods):
        task = mods["task"].builtin.structured_extraction(
            output={"type": "object", "properties": {"name": {"type": "string"}}},
            ontology="structured_extraction/leads/es-AR", instructions="Extract.",
        )
        system = mods["system"].load("echo:", text="I'm sorry, I can't help with that.")
        sample = Sample(id="1", input={"text": "soy Ana"}, expected={"name": "Ana"})
        pred = task.parse(await system.invoke(task.render(sample, system.capabilities)),
                          system.capabilities)
        assert isinstance(pred, Prediction) and not pred.parse_ok


# --------------------------------------------------------------------------
# Seam 3: Data -> Task schema validation
# --------------------------------------------------------------------------

class TestDataTaskSeam:
    def test_loaded_samples_validate_against_the_task_schema(self, mods, tmp_path):
        rows = [{"id": "1", "text": "soy Ana", "expected": {"name": "Ana"}},
                {"id": "2", "text": "me llamo Beto", "expected": {"name": "Beto"}}]
        p = tmp_path / "leads.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

        data = mods["data"].load(f"jsonl:{p}", input={"text": "text"}, expected="expected", id="id")
        task = mods["task"].builtin.structured_extraction(
            output={"type": "object", "properties": {"name": {"type": "string"}}},
            ontology="structured_extraction/leads/es-AR", instructions="Extract.",
        )
        samples = list(data)
        assert len(samples) == 2
        for s in samples:
            task.validate_sample(s)   # must not raise

    def test_data_is_reiterable_because_the_run_loop_walks_it_more_than_once(self, mods, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_text('{"id":"1","text":"x","expected":"x"}\n', encoding="utf-8")
        data = mods["data"].load(f"jsonl:{p}")
        assert [s.id for s in data] == [s.id for s in data]


# --------------------------------------------------------------------------
# Seam 4: Scoring <-> the engine's aggregate contract
# --------------------------------------------------------------------------

class TestScoringEngineSeam:
    def test_aggregate_always_exposes_a_fitness_key(self, mods):
        """Report.fitness is read straight out of this. It is a hard contract."""
        scoring = mods["scoring"].field_wise(
            fields=("name",), per_field=mods["scoring"].exact_match()
        )
        sample = Sample(id="1", input={}, expected={"name": "Ana"})
        scores = [scoring.evaluate({"name": "Ana"}, {"name": "Ana"}, sample),
                  scoring.evaluate({"name": "Bob"}, {"name": "Ana"}, sample)]
        agg = scoring.aggregate(scores)
        assert "fitness" in agg
        assert 0.0 <= float(agg["fitness"]) <= 1.0

    def test_scorer_repr_round_trips_so_benchmark_yaml_can_hold_a_rubric(self, mods):
        s = mods["scoring"]
        original = s.binary(s.field_wise(fields=("a", "b"), per_field=s.exact_match()))
        assert repr(s.parse_scorer(repr(original))) == repr(original)

    def test_error_metrics_are_inverted_so_higher_is_always_better(self, mods):
        wer = mods["scoring"].wer()
        sample = Sample(id="1", input={}, expected="hola mundo")
        perfect = wer.fitness("hola mundo", "hola mundo", sample)
        wrong = wer.fitness("chau planeta", "hola mundo", sample)
        assert perfect > wrong, "wer must invert: an optimizer maximizes fitness"


# --------------------------------------------------------------------------
# Seam 5: the whole thing — Benchmark composition, run, and loss export
# --------------------------------------------------------------------------

@pytest.fixture
def leads_bench(mods, tmp_path):
    rows = [{"id": "1", "text": "soy Ana", "expected": {"name": "Ana"}},
            {"id": "2", "text": "me llamo Beto", "expected": {"name": "Beto"}},
            {"id": "3", "text": "Carla aqui", "expected": {"name": "Carla"}}]
    p = tmp_path / "leads.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    task = mods["task"].builtin.structured_extraction(
        output={"type": "object", "properties": {"name": {"type": "string"}}},
        ontology="structured_extraction/leads/es-AR", instructions="Extract the name.",
    )
    data = mods["data"].load(f"jsonl:{p}", input={"text": "text"}, expected="expected", id="id")
    scoring = mods["scoring"].field_wise(fields=("name",), per_field=mods["scoring"].exact_match())
    return mods["benchmark"].Benchmark(
        task=task, data=data, scoring=scoring, ontology="structured_extraction/leads/es-AR"
    )


class TestEndToEnd:
    async def test_a_benchmark_grades_a_system(self, leads_bench, mods):
        system = mods["system"].load("echo:", data={"name": "Ana"},
                                     capabilities=Capabilities(structured_output=True))
        report = await leads_bench.run(system)
        assert report.n_samples == 3
        assert report.n_errors == 0
        assert 0.0 <= report.fitness <= 1.0
        assert [r.sample_id for r in report.records] == ["1", "2", "3"], "order must be stable"

    async def test_a_perfect_system_scores_one_and_a_wrong_one_scores_zero(self, leads_bench, mods):
        """The scale has to mean something end to end, not just per module."""
        perfect = mods["system"].load(
            "echo:", capabilities=Capabilities(structured_output=True),
            responses=[{"name": "Ana"}, {"name": "Beto"}, {"name": "Carla"}],
        )
        useless = mods["system"].load("echo:", data={"name": "ZZZ"},
                                      capabilities=Capabilities(structured_output=True))
        assert (await leads_bench.run(perfect)).fitness == pytest.approx(1.0)
        assert (await leads_bench.run(useless)).fitness == pytest.approx(0.0)

    async def test_one_bad_sample_does_not_abort_the_run(self, leads_bench, mods):
        system = mods["system"].load("echo:", error_on=["2"], data={"name": "Ana"},
                                     capabilities=Capabilities(structured_output=True))
        report = await leads_bench.run(system)
        assert report.n_samples == 3 and report.n_errors >= 1

    async def test_as_loss_is_exactly_the_report_fitness(self, leads_bench, mods):
        """The vision's headline feature. It must not drift from run()."""
        system = mods["system"].load("echo:", data={"name": "Ana"},
                                     capabilities=Capabilities(structured_output=True))
        loss = leads_bench.as_loss()
        assert await loss(system) == pytest.approx((await leads_bench.run(system)).fitness)

    async def test_one_benchmark_grades_many_systems_without_being_rebuilt(self, leads_bench, mods):
        """Why the system is the argument and not a constructor field."""
        systems = [
            mods["system"].load("echo:", data={"name": n},
                                capabilities=Capabilities(structured_output=True))
            for n in ("Ana", "ZZZ")
        ]
        reports = await leads_bench.compare(systems)
        assert len(reports) == 2
        assert reports[0].fitness > reports[1].fitness

    async def test_report_json_round_trips(self, leads_bench, mods):
        import benchy.report as report_mod

        system = mods["system"].load("echo:", data={"name": "Ana"},
                                     capabilities=Capabilities(structured_output=True))
        report = await leads_bench.run(system)
        assert report_mod.from_json(report_mod.to_json(report)).fitness == report.fitness

    async def test_limit_is_honoured(self, leads_bench, mods):
        system = mods["system"].load("echo:", data={"name": "Ana"},
                                     capabilities=Capabilities(structured_output=True))
        assert (await leads_bench.run(system, limit=2)).n_samples == 2


# --------------------------------------------------------------------------
# Seam 6: the authoring surface the vision actually sells
# --------------------------------------------------------------------------

class TestAuthoringSurface:
    def test_a_benchmark_round_trips_through_yaml(self, leads_bench, mods, tmp_path):
        p = tmp_path / "benchmark.yaml"
        leads_bench.to_yaml(p)
        reloaded = mods["benchmark"].Benchmark.from_yaml(p)
        assert str(reloaded.ontology) == "structured_extraction/leads/es-AR"

    def test_the_top_level_package_exposes_the_whole_flow(self):
        """`import benchy` is the product's front door."""
        import benchy

        assert benchy.Benchmark is not None
        for name in ("Task", "Sample", "Report", "Capabilities", "OntologyPath"):
            assert hasattr(benchy, name)
