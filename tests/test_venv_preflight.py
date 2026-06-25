"""Tests for the per-model venv pre-flight check.

The check lives in `src.benchy_cli_eval._check_required_venv`. It reads
`config['venv']` (path relative to repo root), compares it to `sys.prefix`,
and `sys.exit(2)` with a hint if they don't match. No-op when `venv:` is
absent.
"""

import pytest

from src.benchy_cli_eval import _check_required_venv, _REPO_ROOT


def test_no_venv_field_is_no_op():
    """Configs without `venv:` should never trigger the check."""
    _check_required_venv({"model": {"name": "whisper-tiny"}})


def test_matching_venv_is_no_op(monkeypatch):
    """When sys.prefix matches the declared venv, the check passes."""
    venv_dir = _REPO_ROOT / ".venv-test-match"
    monkeypatch.setattr("sys.prefix", str(venv_dir))
    _check_required_venv(
        {"model": {"name": "voxtral"}, "venv": ".venv-test-match"}
    )


def test_mismatched_venv_exits_with_hint(monkeypatch, capsys):
    """When sys.prefix is wrong, the check exits 2 with the correct invocation."""
    monkeypatch.setattr("sys.prefix", str(_REPO_ROOT / ".venv"))
    monkeypatch.setattr(
        "sys.argv", ["benchy", "eval", "-c", "voxtral-mini-4b", "--limit", "1"]
    )
    with pytest.raises(SystemExit) as exc:
        _check_required_venv(
            {"model": {"name": "Voxtral"}, "venv": ".venv-vox"}
        )
    assert exc.value.code == 2

    err = capsys.readouterr().err
    assert "voxtral" in err.lower()
    assert ".venv-vox" in err
    expected_bin = str((_REPO_ROOT / ".venv-vox" / "bin" / "benchy").resolve())
    assert expected_bin in err
    assert "setup-venvs.sh" in err
