"""Tests for FLEURS transcription subtasks (mocked dataset + audio cache)."""

from pathlib import Path
from unittest.mock import patch

from src.tasks.transcription.fleurs_es_latam import FleursEsLatam
from src.tasks.transcription.fleurs_pt_br import FleursPtBr


def test_fleurs_es_latam_class_attributes():
    assert FleursEsLatam.name == "fleurs_es_latam"
    assert FleursEsLatam.language == "es"
    assert FleursEsLatam.locale == "es_419"
    assert FleursEsLatam.dataset_config == "es_419"
    assert FleursEsLatam.dataset_name == "google/fleurs"
    assert FleursEsLatam.requires_audio is True


def test_fleurs_pt_br_class_attributes():
    assert FleursPtBr.name == "fleurs_pt_br"
    assert FleursPtBr.language == "pt"
    assert FleursPtBr.locale == "pt_br"
    assert FleursPtBr.dataset_config == "pt_br"


def test_load_dataset_returns_expected_sample_shape(tmp_path: Path, monkeypatch):
    fake_items = [
        {
            "id": 1,
            "audio": {"array": [0.0, 0.1, 0.2], "sampling_rate": 16000},
            "transcription": "hola mundo",
        },
        {
            "id": 2,
            "audio": {"array": [0.3, 0.2, 0.1], "sampling_rate": 16000},
            "transcription": "buenos dias",
        },
    ]

    saved_paths: list[Path] = []

    def fake_save(array, sampling_rate, output_path):  # noqa: ARG001
        saved_paths.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")

    # Patch the import sites used inside the subtask module, not the origin module.
    with patch("src.tasks.transcription.fleurs_es_latam.load_dataset", return_value=fake_items), \
         patch("src.tasks.transcription.fleurs_es_latam.save_audio_array", side_effect=fake_save):
        task = FleursEsLatam()
        task.data_dir = tmp_path  # redirect cache into pytest tmp_path
        samples = task.load_dataset()

    assert len(samples) == 2
    assert samples[0]["id"] == "es_419_1"
    assert samples[0]["language"] == "es"
    assert samples[0]["locale"] == "es_419"
    assert samples[0]["expected"] == "hola mundo"
    assert samples[0]["audio_path"] == str(tmp_path / "1.wav")
    assert len(saved_paths) == 2


def test_fleurs_es_latam_data_dir_under_repo_root(tmp_path: Path):
    task = FleursEsLatam()
    assert task.data_dir.parts[-3:] == (".data", "transcription", "es_419")
