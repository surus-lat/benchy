from pathlib import Path

from src.engine.checkpoint import get_checkpoint_path, get_config_hash, load_checkpoint, save_checkpoint
from src.task_completion_checker import TASK_STATUS_FILENAME, TaskCompletionChecker, read_task_status, write_task_status


def test_write_and_read_task_status_roundtrip(tmp_path) -> None:
    payload = write_task_status(
        tmp_path,
        task_name="spanish",
        status="passed",
        reason=None,
        summary={"score": 0.9},
        subtasks={"xnli": {"status": "passed"}},
        details={"source": "test"},
    )

    loaded = read_task_status(tmp_path)

    assert loaded is not None
    assert loaded["task"] == "spanish"
    assert loaded["status"] == "passed"
    assert loaded["summary"]["score"] == 0.9
    assert loaded["details"]["source"] == "test"
    assert payload["status"] == "passed"


def test_task_completion_checker_maps_subtask_refs_to_root_task_dir(tmp_path) -> None:
    checker = TaskCompletionChecker(
        output_path=str(tmp_path),
        run_id="run1",
        model_name="org/model",
    )

    model_root = Path(checker.model_output_path)
    spanish_dir = model_root / "spanish"
    spanish_dir.mkdir(parents=True)
    write_task_status(spanish_dir, task_name="spanish", status="passed")

    records = checker.get_task_records(["spanish.xnli", "portuguese"])

    assert records["spanish.xnli"]["completed"] is True
    assert records["spanish.xnli"]["status"] == "passed"
    assert records["portuguese"]["completed"] is False
    assert records["portuguese"]["status"] == "pending"
    assert Path(records["spanish.xnli"]["task_output_dir"]).name == "spanish"


def test_checkpoint_save_and_load_happy_path(tmp_path) -> None:
    ckpt_path = tmp_path / ".checkpoints" / "m_task_checkpoint.json"
    config_hash = get_config_hash({"model": "m", "task": "t", "batch_size": 2})

    save_checkpoint(
        ckpt_path,
        completed_ids=["s1", "s2"],
        config_hash=config_hash,
        metrics_by_id={"s1": {"valid": True}},
    )

    completed, metrics_by_id = load_checkpoint(ckpt_path, expected_config_hash=config_hash)

    assert completed == {"s1", "s2"}
    assert metrics_by_id["s1"]["valid"] is True


def test_checkpoint_load_ignores_hash_mismatch(tmp_path) -> None:
    ckpt_path = tmp_path / "checkpoint.json"
    save_checkpoint(ckpt_path, completed_ids=["s1"], config_hash="hash_a")

    completed, metrics = load_checkpoint(ckpt_path, expected_config_hash="hash_b")

    assert completed == set()
    assert metrics == {}


def test_get_checkpoint_path_sanitizes_model_name() -> None:
    path = get_checkpoint_path("/tmp/results", "org/model", "spanish")

    assert ".checkpoints" in str(path)
    assert "org_model_spanish_checkpoint.json" in str(path)


def test_read_task_status_returns_none_when_missing(tmp_path) -> None:
    assert read_task_status(tmp_path / "missing") is None
    assert TASK_STATUS_FILENAME == "task_status.json"
