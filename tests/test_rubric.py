"""Rubric-graded question answering: the `answer` task (ontology 1.1) and the
`rubric` field metric, whose grade comes from a grader AI-system.

The grader here is always a plain callable, so nothing in this file needs a model, a
network or a key. See docs/engine-v1/RUBRIC-QA.md for the design.
"""

from __future__ import annotations

import copy
import json
import textwrap

import pytest
from conftest import edit

from benchy import ontology, score
from benchy.cli import main
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run
from benchy.types import compile_schema

LEVELS = {1: "wrong", 2: "weak", 3: "partial", 4: "good", 5: "complete"}
#: Analytic: accuracy counts three times as much as clarity.
RUBRIC = {"criteria": {
    "accuracy": {"weight": 3, "description": "matches the answer key", "levels": LEVELS},
    "clarity": {"weight": 1, "levels": {0: "unreadable", 1: "clear"}},
}}
GRADER = {"type": "external", "id": "professor"}


def rubric(**criterion):
    """A one-criterion rubric whose criterion is `criterion` over sensible defaults."""
    return {"metric": "rubric", "params": {"criteria": {"only": {"weight": 1, "levels": LEVELS, **criterion}}}}


def exam(**scoring):
    """The rubric exam, with `scoring` keys replaced; a value of `None` removes one."""
    section = {
        "weights": {"answer": 1},
        "aggregator": "weighted_mean",
        "field_metrics": {"answer": {"metric": "rubric", "params": RUBRIC}},
        "grader": GRADER,
        **scoring,
    }
    return edit(
        ontology_version="1.1",
        benchmark={"task": "answer", "domain": "history", "language": "es"},
        program={"input": {"question": "string"}, "output": {"answer": "string"}},
        scoring={key: value for key, value in section.items() if value is not None},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "external", "id": "student"},
    )


IR = compile_benchmark(exam())
ROWS = [
    {"input": {"question": "q0"}, "expected": {"answer": "key0"}},
    {"input": {"question": "q1"}, "expected": {"answer": "key1"}},
]


def compile_fails(text) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        compile_benchmark(text)
    return exc.value


class Grader:
    """Grades by answer from a table of (accuracy, clarity), and records each call.

    By default "good" is accurate 4/5 and clear, the empty answer is the floor of both.
    """

    def __init__(self, grades=None):
        self.calls = []
        self.grades = grades or {"good": (4, 1), "": (1, 0)}

    def __call__(self, graded):
        self.calls.append(graded)
        accuracy, clarity = self.grades.get(graded["answer"], (2, 0))
        return {"accuracy": {"grade": accuracy, "justification": f"saw {graded['answer']!r}"},
                "clarity": {"grade": clarity, "justification": "ok"}}


async def execute(tmp_path, student, grader, rows=ROWS, ir=IR):
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return await run(ir, tmp_path, student, grader=grader)


# ---------------------------------------------------------------------------
# ontology 1.1: the `answer` task
# ---------------------------------------------------------------------------

def check(task, version, *, out=None, inp=None):
    benchmark = {"task": task, "domain": "history", "language": "es"}
    registry = ontology.load(version)
    ontology.check_classification(benchmark, registry)
    ontology.check_program(benchmark, compile_schema(inp or {"question": "string"}),
                           compile_schema(out or {"answer": "string"}), registry)


def test_ontology_1_1_is_1_0_plus_answer():
    old, new = ontology.load("1.0"), ontology.load("1.1")
    assert set(new["tasks"]) == set(old["tasks"]) | {"answer"}
    assert new["domains"] == old["domains"] and new["languages"] == old["languages"]


def test_answer_is_not_a_task_in_ontology_1_0():
    with pytest.raises(BenchyError) as exc:
        check("answer", "1.0")
    assert exc.value.code == "unknown_task"


def test_answer_accepts_any_input_and_one_string_output():
    check("answer", "1.1")
    check("answer", "1.1", inp={"exam_page": "image", "context": "document"})


@pytest.mark.parametrize("out", [{"answer": "int"}, {"answer": "string", "confidence": "float"}])
def test_answer_requires_exactly_one_string_output(out):
    with pytest.raises(BenchyError) as exc:
        check("answer", "1.1", out=out)
    assert exc.value.code == "task_program_mismatch"


def test_every_1_0_task_still_validates_under_1_1():
    check("extract", "1.1", out={"total": "float"})


# ---------------------------------------------------------------------------
# compiling a rubric
# ---------------------------------------------------------------------------

def test_rubric_compiles_to_ordered_criteria_with_ordered_levels_and_a_grader():
    (dimension,) = IR["scoring"]["dimensions"]
    assert dimension["metric"] == "rubric"
    assert dimension["params"]["criteria"] == [
        {"name": "accuracy", "weight": 3.0, "levels": [{"grade": g, "description": d} for g, d in LEVELS.items()],
         "description": "matches the answer key"},
        {"name": "clarity", "weight": 1.0, "levels": [{"grade": 0, "description": "unreadable"},
                                                      {"grade": 1, "description": "clear"}]},
    ]
    assert IR["scoring"]["grader"] == GRADER


def test_rubric_ir_is_byte_identical_across_compiles_and_json_round_trips():
    assert json.dumps(compile_benchmark(exam())) == json.dumps(IR)
    assert json.loads(json.dumps(IR)) == IR


def test_a_benchmark_without_a_rubric_has_no_grader_key():
    plain = compile_benchmark(edit(data={"path": "./exam.jsonl"}))
    assert "grader" not in plain["scoring"]


def test_a_holistic_rubric_is_the_one_criterion_case():
    ir = compile_benchmark(exam(field_metrics={"answer": rubric()}))
    assert [c["name"] for c in ir["scoring"]["dimensions"][0]["params"]["criteria"]] == ["only"]


@pytest.mark.parametrize("levels", [
    {1: "only one"},
    {1: "a", 3: "gap"},
    {1: "a", 2: ""},
    {1: "a", 2: 5},
    {"one": "a", "two": "b"},
    {True: "a", 2: "b"},
    {n: str(n) for n in range(12)},
    ["a", "b"],
])
def test_malformed_levels_are_rejected(levels):
    error = compile_fails(exam(field_metrics={"answer": rubric(levels=levels)}))
    assert (error.code, error.path) == ("invalid_rubric", ["scoring", "field_metrics", "answer"])


@pytest.mark.parametrize("criterion", [{"weight": -1}, {"weight": "3"}, {"weight": True}, {"weight": 0},
                                       {"description": 7}])
def test_malformed_criteria_are_rejected(criterion):
    assert compile_fails(exam(field_metrics={"answer": rubric(**criterion)})).code == "invalid_rubric"


@pytest.mark.parametrize("params", [{"criteria": {}}, {"criteria": ["accuracy"]}, {"criteria": {"": {"weight": 1, "levels": LEVELS}}}])
def test_a_rubric_needs_named_criteria(params):
    assert compile_fails(exam(field_metrics={"answer": {"metric": "rubric", "params": params}})).code == "invalid_rubric"


def test_a_rubric_and_its_criteria_are_closed():
    assert compile_fails(exam(field_metrics={"answer": {"metric": "rubric"}})).code == "missing_key"
    assert compile_fails(exam(field_metrics={"answer": {"metric": "rubric", "params": {**RUBRIC, "x": 1}}})).code == "unknown_key"
    assert compile_fails(exam(field_metrics={"answer": rubric(extra=1)})).code == "unknown_key"
    assert compile_fails(exam(field_metrics={"answer": {"metric": "rubric", "params": {"criteria": {"c": {"weight": 1}}}}})).code == "missing_key"


def test_a_rubric_grades_only_string_fields():
    text = edit(
        program={"input": {"text": "string"}, "output": {"label": {"enum": ["a", "b"]}, "total": "float"}},
        scoring={"weights": {"label": 1, "total": 1}, "aggregator": "weighted_mean", "grader": GRADER,
                 "field_metrics": {"total": {"metric": "rubric", "params": RUBRIC}}},
    )
    error = compile_fails(text)
    assert (error.code, error.path) == ("invalid_rubric", ["scoring", "field_metrics", "total"])


def test_a_rubric_scores_a_string_field_of_any_task():
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"summary": "string", "total": "float"}},
        scoring={"weights": {"summary": 1, "total": 1}, "aggregator": "weighted_mean", "grader": GRADER,
                 "field_metrics": {"summary": {"metric": "rubric", "params": RUBRIC}}},
    ))
    assert ir["benchmark"]["task"] == "extract"


def test_one_rubric_field_per_benchmark():
    error = compile_fails(edit(
        program={"input": {"text": "string"}, "output": {"a": "string", "b": "string"}},
        scoring={"weights": {"a": 1, "b": 1}, "aggregator": "weighted_mean", "grader": GRADER,
                 "field_metrics": {"a": {"metric": "rubric", "params": RUBRIC}, "b": {"metric": "rubric", "params": copy.deepcopy(RUBRIC)}}},
    ))
    assert (error.code, error.path) == ("invalid_rubric", ["scoring", "field_metrics"])


def test_a_rubric_without_a_grader_is_rejected():
    error = compile_fails(exam(grader=None))
    assert (error.code, error.path) == ("missing_key", ["scoring", "grader"])


def test_a_grader_without_a_rubric_is_rejected():
    error = compile_fails(edit(scoring={
        "weights": {"invoice_number": 1, "date": 1, "supplier": 1, "subtotal": 1, "total": 5},
        "aggregator": "weighted_mean", "grader": GRADER,
    }))
    assert (error.code, error.path) == ("invalid_value", ["scoring", "grader"])


def test_a_malformed_grader_is_reported_at_its_own_path():
    error = compile_fails(exam(grader={"type": "model", "provider": "openai"}))
    assert (error.code, error.path) == ("missing_key", ["scoring", "grader"])
    error = compile_fails(exam(grader={"type": "model", "provider": "", "model": "m"}))
    assert (error.code, error.path) == ("invalid_ai_system", ["scoring", "grader", "provider"])


# ---------------------------------------------------------------------------
# the grader contract
# ---------------------------------------------------------------------------

def test_the_grader_program_is_an_ordinary_ir_program_with_one_grade_per_criterion():
    program = score.grader_program(IR)
    assert program["input"]["fields"]["input"] == IR["program"]["input"]
    assert {name: node["type"] for name, node in program["input"]["fields"].items()} == {
        "input": "object", "reference": "string", "answer": "string", "rubric": "string",
    }
    assert program["output"] == {"type": "object", "fields": {"accuracy": score.GRADE, "clarity": score.GRADE}}


def test_each_criterion_maps_linearly_and_the_criteria_combine_by_weight():
    criteria = IR["scoring"]["dimensions"][0]["params"]["criteria"]

    def graded(accuracy, clarity):
        return {"accuracy": {"grade": accuracy}, "clarity": {"grade": clarity}}

    assert score.rubric_score(graded(5, 1), criteria) == 1.0
    assert score.rubric_score(graded(1, 0), criteria) == 0.0
    assert score.rubric_score(graded(4, 1), criteria) == (3 * 0.75 + 1 * 1.0) / 4
    assert score.rubric_score(graded(3, 0), criteria) == (3 * 0.5) / 4


# ---------------------------------------------------------------------------
# running a rubric
# ---------------------------------------------------------------------------

async def test_a_graded_answer_scores_its_weighted_criteria_and_keeps_every_grade_and_justification(tmp_path):
    result = await execute(tmp_path, lambda _: {"answer": "good"}, Grader())
    assert result["benchmark_score"] == 0.8125
    (field,) = result["results"][0]["field_scores"]
    assert field == {"path": ["answer"], "score": 0.8125, "weight": 1.0, "criteria": {
        "accuracy": {"grade": 4, "justification": "saw 'good'"},
        "clarity": {"grade": 1, "justification": "ok"},
    }}


async def test_the_grader_sees_the_question_the_reference_the_answer_and_the_rubric(tmp_path):
    grader = Grader()
    await execute(tmp_path, lambda _: {"answer": "good"}, grader)
    graded = [call for call in grader.calls if call["answer"] == "good"]
    assert graded[0] == {
        "input": {"question": "q0"},
        "reference": "key0",
        "answer": "good",
        "rubric": "accuracy: matches the answer key\n  1: wrong\n  2: weak\n  3: partial\n  4: good\n  5: complete"
                  "\n\nclarity\n  0: unreadable\n  1: clear",
    }


async def test_failed_examples_are_never_graded(tmp_path):
    def student(inputs):
        if inputs["question"] == "q0":
            raise RuntimeError("down")
        return {"answer": 7}

    grader = Grader()
    result = await execute(tmp_path, student, grader)
    assert [r["status"] for r in result["results"]] == ["execution_error", "invalid_output"]
    assert result["benchmark_score"] == 0.0
    assert all(call["answer"] == "" for call in grader.calls)  # only the baseline was graded


async def test_the_baseline_is_the_graded_empty_answer_once_per_question(tmp_path):
    grader = Grader({"good": (5, 1), "": (2, 0)})
    result = await execute(tmp_path, lambda _: {"answer": "good"}, grader)
    empties = [call for call in grader.calls if call["answer"] == ""]
    assert [call["reference"] for call in empties] == ["key0", "key1"]
    (field,) = result["fields"]
    assert field["metric"] == "rubric"
    assert field["baseline"] == 3 * 0.25 / 4
    assert field["score"] == 1.0
    assert field["counts_as_signal"] is True


async def test_the_result_records_the_grader_and_its_prompt_fingerprint(tmp_path):
    grader = Grader()
    result = await execute(tmp_path, lambda _: {"answer": "good"}, grader)
    assert result["grader"] == {"system": GRADER, "prompt_sha256": None}
    grader.prompt_sha256 = "abc"
    result = await execute(tmp_path, lambda _: {"answer": "good"}, grader)
    assert result["grader"]["prompt_sha256"] == "abc"


async def test_a_result_without_a_rubric_has_no_grader_block(tmp_path):
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
    ))
    rows = [{"input": {"text": "t"}, "expected": {"total": 1.0}}]
    result = await execute(tmp_path, lambda _: {"total": 1.0}, None, rows=rows, ir=ir)
    assert "grader" not in result


async def test_a_rubric_run_without_a_grader_aborts_before_any_example(tmp_path):
    calls = []
    with pytest.raises(BenchyError) as exc:
        await execute(tmp_path, lambda i: calls.append(i) or {"answer": "good"}, None)
    assert (exc.value.code, exc.value.path) == ("grader_not_bound", ["scoring", "grader"])
    assert calls == []


@pytest.mark.parametrize("reply", [
    {"accuracy": {"grade": 6, "justification": "off the scale"}, "clarity": {"grade": 1, "justification": ""}},
    {"accuracy": {"grade": "4", "justification": "a string"}, "clarity": {"grade": 1, "justification": ""}},
    {"accuracy": {"grade": 4}, "clarity": {"grade": 1, "justification": ""}},
    {"accuracy": {"grade": 4, "justification": "clarity forgotten"}},
    "four",
])
async def test_an_invalid_grade_aborts_the_run(tmp_path, reply):
    with pytest.raises(BenchyError) as exc:
        await execute(tmp_path, lambda _: {"answer": "good"}, lambda _: reply)
    assert (exc.value.phase, exc.value.code) == ("runtime", "grader_error")


async def test_a_grader_that_raises_aborts_the_run_rather_than_failing_the_student(tmp_path):
    def broken(_):
        raise TimeoutError("judge timed out")

    with pytest.raises(BenchyError) as exc:
        await execute(tmp_path, lambda _: {"answer": "good"}, broken)
    assert exc.value.code == "grader_error"
    assert "judge timed out" in exc.value.message


async def test_an_async_grader_works(tmp_path):
    async def grader(graded):
        return {"accuracy": {"grade": 5, "justification": "ok"}, "clarity": {"grade": 1, "justification": "ok"}}

    assert (await execute(tmp_path, lambda _: {"answer": "x"}, grader))["benchmark_score"] == 1.0


async def test_a_rubric_run_reruns_from_persisted_ir(tmp_path):
    persisted = json.loads(json.dumps(IR))
    first = await execute(tmp_path, lambda _: {"answer": "good"}, Grader())
    again = await execute(tmp_path, lambda _: {"answer": "good"}, Grader(), ir=persisted)
    assert json.dumps(first) == json.dumps(again)


# ---------------------------------------------------------------------------
# the CLI binds the grader the way it binds the AI-system
# ---------------------------------------------------------------------------

ADAPTERS = textwrap.dedent('''
    def student(inputs):
        return {"answer": "good"}

    def professor(graded):
        accuracy, clarity = (4, 1) if graded["answer"] else (1, 0)
        return {"accuracy": {"grade": accuracy, "justification": "by hand"},
                "clarity": {"grade": clarity, "justification": "by hand"}}
''')


@pytest.fixture
def bench(tmp_path):
    (tmp_path / "benchmark.yaml").write_text(exam())
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    (tmp_path / "people.py").write_text(ADAPTERS)
    return tmp_path


def test_cli_grader_flag_binds_any_adapter(bench, capsys):
    people = bench / "people.py"
    code = main(["run", str(bench / "benchmark.yaml"), "--adapter", f"{people}:student", "--grader", f"{people}:professor"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["benchmark_score"] == 0.8125


def test_cli_external_grader_without_the_flag_is_a_setup_error(bench, capsys):
    assert main(["run", str(bench / "benchmark.yaml"), "--adapter", f"{bench / 'people.py'}:student"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "grader_not_bound"


def test_cli_model_grader_comes_from_the_providers_with_the_grading_prompt(bench, monkeypatch):
    pytest.importorskip("llm_client")
    from benchy import providers

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ir = compile_benchmark(exam(grader={"type": "model", "provider": "openai", "model": "judge"}))
    grader = providers.for_grader(ir, bench)
    assert grader.prompt == providers.GRADING_PROMPT
    assert grader.prompt_sha256 == __import__("hashlib").sha256(providers.GRADING_PROMPT.encode()).hexdigest()
    assert set(grader.response_format["json_schema"]["schema"]["properties"]) == {"accuracy", "clarity"}


def test_a_declared_grader_prompt_replaces_the_default(bench, monkeypatch):
    pytest.importorskip("llm_client")
    from benchy import providers

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    (bench / "grader.md").write_text("Grade like a strict professor.")
    ir = compile_benchmark(exam(grader={"type": "model", "provider": "openai", "model": "judge", "prompt": "./grader.md"}))
    assert providers.for_grader(ir, bench).prompt == "Grade like a strict professor."


def test_grader_setup_errors_point_at_scoring_grader(bench, monkeypatch):
    pytest.importorskip("llm_client")
    from benchy import providers

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ir = compile_benchmark(exam(grader={"type": "model", "provider": "openai", "model": "judge", "prompt": "./missing.md"}))
    with pytest.raises(BenchyError) as exc:
        providers.for_grader(ir, bench)
    assert exc.value.path == ["scoring", "grader", "prompt"]
    ir = compile_benchmark(exam(grader={"type": "model", "provider": "nowhere", "model": "judge"}))
    with pytest.raises(BenchyError) as exc:
        providers.for_grader(ir, bench)
    assert exc.value.path == ["scoring", "grader", "provider"]
