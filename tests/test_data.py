"""P6 — streaming the exam, and the workspace boundary (spec §7, §10).

Covers conformance cases C16, C17, C18, C19, C32.
"""

from __future__ import annotations

import json

import pytest
from conftest import edit

from benchy import data
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError

TEXT_IR = compile_benchmark(edit(
    program={"input": {"text": "string"}, "output": {"total": "float"}},
    scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
))
IMAGE_IR = compile_benchmark(edit(
    program={"input": {"image": "image"}, "output": {"total": "float"}},
    scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
))


def write(tmp_path, *rows, name="exam.jsonl", raw=None):
    """Materialize a workspace containing `exam.jsonl`; return the workspace root."""
    body = raw if raw is not None else "\n".join(json.dumps(r) for r in rows)
    (tmp_path / name).write_text(body)
    return tmp_path


def rows(ir, workspace):
    return list(data.examples(ir, workspace))


def fails(ir, workspace) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        rows(ir, workspace)
    return exc.value


def row(text="hola", total=1.0):
    return {"input": {"text": text}, "expected": {"total": total}}


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

def test_one_and_many_rows(tmp_path):
    ws = write(tmp_path, row("a"), row("b"), row("c"))
    assert [(i, inp["text"]) for i, inp, _ in rows(TEXT_IR, ws)] == [(0, "a"), (1, "b"), (2, "c")]


def test_indices_are_zero_based_and_sequential(tmp_path):
    ws = write(tmp_path, row(), row(), row())
    assert [i for i, _, _ in rows(TEXT_IR, ws)] == [0, 1, 2]


def test_blank_lines_are_ignored_and_do_not_consume_an_index(tmp_path):
    raw = json.dumps(row("a")) + "\n\n   \n" + json.dumps(row("b")) + "\n"
    ws = write(tmp_path, raw=raw)
    assert [(i, inp["text"]) for i, inp, _ in rows(TEXT_IR, ws)] == [(0, "a"), (1, "b")]


def test_expected_values_are_returned_alongside_inputs(tmp_path):
    ws = write(tmp_path, row("a", 12.5))
    _, inputs, expected = rows(TEXT_IR, ws)[0]
    assert inputs == {"text": "a"}
    assert expected == {"total": 12.5}


def test_rows_are_streamed_not_preloaded(tmp_path):
    """A bad third row must not prevent the first two from being yielded."""
    raw = "\n".join([json.dumps(row("a")), json.dumps(row("b")), "{not json"])
    ws = write(tmp_path, raw=raw)
    stream = data.examples(TEXT_IR, ws)
    assert next(stream)[1]["text"] == "a"
    assert next(stream)[1]["text"] == "b"
    with pytest.raises(BenchyError) as exc:
        next(stream)
    assert exc.value.code == "invalid_dataset_record"


# ---------------------------------------------------------------------------
# row structure
# ---------------------------------------------------------------------------

def test_malformed_json_is_rejected_with_its_line_number(tmp_path):
    ws = write(tmp_path, raw=json.dumps(row()) + "\n{not json\n")
    err = fails(TEXT_IR, ws)
    assert err.code == "invalid_dataset_record"
    assert "line 2" in err.message


@pytest.mark.parametrize(
    "bad_row",
    [
        {"input": {"text": "a"}},                                    # no expected
        {"expected": {"total": 1.0}},                                # no input
        {"input": {"text": "a"}, "expected": {"total": 1.0}, "x": 1},  # extra root key
        {"input": {"text": "a"}, "output": {"total": 1.0}},           # wrong key name
    ],
)
def test_row_must_contain_exactly_input_and_expected(tmp_path, bad_row):
    assert fails(TEXT_IR, write(tmp_path, bad_row)).code == "invalid_dataset_record"


@pytest.mark.parametrize("scalar", ["a string", 5, [1, 2], None])
def test_non_object_row_is_rejected(tmp_path, scalar):
    assert fails(TEXT_IR, write(tmp_path, scalar)).code == "invalid_dataset_record"


def test_c32_empty_dataset_is_a_run_data_error(tmp_path):
    assert fails(TEXT_IR, write(tmp_path, raw="")).code == "empty_dataset"
    assert fails(TEXT_IR, write(tmp_path, raw="\n\n  \n")).code == "empty_dataset"


# ---------------------------------------------------------------------------
# rows validated against the compiled schemas (spec §8)
# ---------------------------------------------------------------------------

def test_c16_extra_dataset_input_field_is_a_dataset_error(tmp_path):
    bad = {"input": {"text": "a", "extra": "b"}, "expected": {"total": 1.0}}
    err = fails(TEXT_IR, write(tmp_path, bad))
    assert (err.phase, err.code, err.path) == ("dataset", "extra_field", ["extra"])


def test_c17_missing_expected_field_is_a_dataset_error(tmp_path):
    bad = {"input": {"text": "a"}, "expected": {}}
    err = fails(TEXT_IR, write(tmp_path, bad))
    assert (err.phase, err.code, err.path) == ("dataset", "missing_field", ["total"])


def test_wrong_input_type_is_a_dataset_error(tmp_path):
    bad = {"input": {"text": 5}, "expected": {"total": 1.0}}
    assert fails(TEXT_IR, write(tmp_path, bad)).code == "wrong_type"


def test_wrong_expected_type_is_a_dataset_error(tmp_path):
    bad = {"input": {"text": "a"}, "expected": {"total": "lots"}}
    assert fails(TEXT_IR, write(tmp_path, bad)).code == "wrong_type"


# ---------------------------------------------------------------------------
# data.path resolution (spec §10, amendment §2)
# ---------------------------------------------------------------------------

def test_missing_dataset_file_is_data_not_found(tmp_path):
    assert fails(TEXT_IR, tmp_path).code == "data_not_found"


def test_data_path_resolves_from_the_workspace_root(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "exam.jsonl").write_text(json.dumps(row()))
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
        data={"path": "sub/exam.jsonl"},
    ))
    assert len(rows(ir, tmp_path)) == 1


@pytest.mark.parametrize("escape", ["../outside.jsonl", "/etc/passwd"])
def test_data_path_may_not_escape_the_workspace(tmp_path, escape):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "outside.jsonl").write_text(json.dumps(row()))
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
        data={"path": escape},
    ))
    assert fails(ir, ws).code == "path_escape"


# ---------------------------------------------------------------------------
# artifacts (spec §10) — resolved relative to the JSONL directory
# ---------------------------------------------------------------------------

def test_c18_relative_artifact_resolves_to_an_absolute_path(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "invoice-001.png").write_bytes(b"png")
    ws = write(tmp_path, {"input": {"image": "assets/invoice-001.png"}, "expected": {"total": 1.0}})
    _, inputs, _ = rows(IMAGE_IR, ws)[0]
    assert inputs["image"] == str(assets / "invoice-001.png")
    assert inputs["image"] != "assets/invoice-001.png"


def test_artifacts_resolve_from_the_jsonl_directory_not_the_workspace_root(tmp_path):
    sub = tmp_path / "sub"
    (sub / "assets").mkdir(parents=True)
    (sub / "assets" / "a.png").write_bytes(b"png")
    (sub / "exam.jsonl").write_text(json.dumps({"input": {"image": "assets/a.png"}, "expected": {"total": 1.0}}))
    ir = compile_benchmark(edit(
        program={"input": {"image": "image"}, "output": {"total": "float"}},
        scoring={"weights": {"total": 1}, "aggregator": "weighted_mean"},
        data={"path": "sub/exam.jsonl"},
    ))
    _, inputs, _ = rows(ir, tmp_path)[0]
    assert inputs["image"] == str(sub / "assets" / "a.png")


def test_c19_missing_artifact_is_a_dataset_error(tmp_path):
    ws = write(tmp_path, {"input": {"image": "assets/absent.png"}, "expected": {"total": 1.0}})
    err = fails(IMAGE_IR, ws)
    assert (err.phase, err.code) == ("dataset", "artifact_not_found")


def test_artifact_traversal_outside_the_workspace_is_rejected(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.png").write_bytes(b"png")
    (ws / "exam.jsonl").write_text(json.dumps({"input": {"image": "../secret.png"}, "expected": {"total": 1.0}}))
    err = fails(IMAGE_IR, ws)
    assert err.code == "path_escape"


def test_artifact_symlink_escape_is_rejected(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.png").write_bytes(b"png")
    (ws / "link.png").symlink_to(tmp_path / "secret.png")
    (ws / "exam.jsonl").write_text(json.dumps({"input": {"image": "link.png"}, "expected": {"total": 1.0}}))
    assert fails(IMAGE_IR, ws).code == "path_escape"


def test_artifact_inside_the_workspace_via_symlink_is_allowed(tmp_path):
    (tmp_path / "real.png").write_bytes(b"png")
    (tmp_path / "link.png").symlink_to(tmp_path / "real.png")
    ws = write(tmp_path, {"input": {"image": "link.png"}, "expected": {"total": 1.0}})
    assert rows(IMAGE_IR, ws)[0][1]["image"] == str(tmp_path / "real.png")


def test_expected_artifacts_are_resolved_too(tmp_path):
    (tmp_path / "out.png").write_bytes(b"png")
    ir = compile_benchmark(edit(
        program={"input": {"text": "string"}, "output": {"image": "image"}},
        scoring={"weights": {"image": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
    ))
    ws = write(tmp_path, {"input": {"text": "a"}, "expected": {"image": "out.png"}})
    _, _, expected = rows(ir, ws)[0]
    assert expected["image"] == str(tmp_path / "out.png")
