from pathlib import Path

import pytest

from onedrive_destination_recommender import msg_reader


def test_pywin32_import_is_delayed_until_msg_access(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_modules: list[str] = []

    def unavailable_import(name: str):
        requested_modules.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(msg_reader, "import_module", unavailable_import)

    with pytest.raises(msg_reader.OutlookUnavailableError):
        msg_reader._load_win32com_client()

    assert requested_modules == ["win32com.client"]


def test_probe_rejects_non_msg_before_loading_pywin32(tmp_path: Path) -> None:
    other_file = tmp_path / "sample.pdf"
    other_file.write_bytes(b"")

    with pytest.raises(ValueError, match="MSG"):
        msg_reader.probe_msg_access(other_file)


def test_probe_rejects_missing_msg_before_loading_pywin32(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        msg_reader.probe_msg_access(tmp_path / "missing.msg")
