import subprocess
import sys
from pathlib import Path

import pytest

from onedrive_destination_recommender import msg_reader


def test_importing_msg_reader_does_not_load_pywin32() -> None:
    code = (
        "import sys; "
        "import onedrive_destination_recommender.msg_reader; "
        "assert not any(name == 'win32com' or name.startswith('win32com.') "
        "for name in sys.modules)"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


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


def test_probe_closes_opened_outlook_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    msg_path = tmp_path / "sample.msg"
    msg_path.write_bytes(b"")

    class FakeAttachments:
        Count = 2

    class FakeItem:
        Subject = "synthetic subject"
        Body = "synthetic body"
        Attachments = FakeAttachments()

        def __init__(self) -> None:
            self.close_arguments: list[int] = []

        def Close(self, save_mode: int) -> None:
            self.close_arguments.append(save_mode)

    item = FakeItem()

    class FakeNamespace:
        def OpenSharedItem(self, _path: str) -> FakeItem:
            return item

    class FakeOutlook:
        def GetNamespace(self, _name: str) -> FakeNamespace:
            return FakeNamespace()

    class FakeClient:
        @staticmethod
        def Dispatch(_name: str) -> FakeOutlook:
            return FakeOutlook()

    monkeypatch.setattr(msg_reader, "_load_win32com_client", lambda: FakeClient())

    result = msg_reader.probe_msg_access(msg_path)

    assert result.attachment_count == 2
    assert item.close_arguments == [1]
