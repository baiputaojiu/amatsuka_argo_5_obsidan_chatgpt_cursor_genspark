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


def test_msg_input_is_classified_by_extension_case_insensitively() -> None:
    assert msg_reader.is_msg_file("sample.MSG")
    assert not msg_reader.is_msg_file("sample.pdf")


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


def test_read_msg_content_reads_fields_and_attachment_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg_path = tmp_path / "sample.msg"
    msg_path.write_bytes(b"")

    class FakeAttachment:
        def __init__(self, file_name: str) -> None:
            self.FileName = file_name

    class FakeAttachments:
        Count = 2

        @staticmethod
        def Item(index: int) -> FakeAttachment:
            return FakeAttachment(("image1.png", "図面.pdf")[index - 1])

    class FakeItem:
        Subject = "合成件名"
        Body = "合成本文"
        Attachments = FakeAttachments()

        def __init__(self) -> None:
            self.close_arguments: list[int] = []

        def Close(self, save_mode: int) -> None:
            self.close_arguments.append(save_mode)

    item = FakeItem()

    class FakeNamespace:
        @staticmethod
        def OpenSharedItem(_path: str) -> FakeItem:
            return item

    class FakeOutlook:
        @staticmethod
        def GetNamespace(_name: str) -> FakeNamespace:
            return FakeNamespace()

    class FakeClient:
        @staticmethod
        def Dispatch(_name: str) -> FakeOutlook:
            return FakeOutlook()

    monkeypatch.setattr(msg_reader, "_load_win32com_client", lambda: FakeClient())

    content = msg_reader.read_msg_content(msg_path)

    assert content.subject == "合成件名"
    assert content.body == "合成本文"
    assert content.attachment_file_names == ("image1.png", "図面.pdf")
    assert content.attachment_count == 2
    assert content.subject_available
    assert content.body_available
    assert content.attachments_available
    assert item.close_arguments == [1]


def test_build_msg_search_terms_uses_subject_attachments_and_cleaned_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg_path = tmp_path / "mail.msg"
    msg_path.write_bytes(b"")
    content = msg_reader.MsgContent(
        subject="設備 カメラ",
        body="秋田 2025 03\n-----Original Message-----\n過去本文",
        attachment_file_names=("image1.png", "Layout.pdf"),
        attachment_count=2,
        subject_available=True,
        body_available=True,
        attachments_available=True,
    )
    monkeypatch.setattr(msg_reader, "read_msg_content", lambda _path: content)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    result = msg_reader.build_msg_search_terms(msg_path)
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert result.primary_terms == ("mail", "設備", "カメラ", "layout")
    assert result.auxiliary_terms == ("秋田", "2025")
    assert result.fully_parsed
    assert result.body_available
    assert result.warning is None
    assert after == before


def test_build_msg_search_terms_falls_back_to_file_name_without_writing_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg_path = tmp_path / "案件資料.msg"
    msg_path.write_bytes(b"original")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def unavailable_read(_path: Path) -> msg_reader.MsgContent:
        raise msg_reader.OutlookUnavailableError("synthetic failure")

    monkeypatch.setattr(msg_reader, "read_msg_content", unavailable_read)

    result = msg_reader.build_msg_search_terms(msg_path)
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert result.primary_terms == ("案件資料",)
    assert result.auxiliary_terms == ()
    assert not result.fully_parsed
    assert not result.body_available
    assert result.warning == "メール本文を利用できませんでした。"
    assert after == before


def test_build_msg_search_terms_keeps_available_fields_on_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg_path = tmp_path / "mail.msg"
    msg_path.write_bytes(b"")
    content = msg_reader.MsgContent(
        subject="利用可能な件名",
        body="",
        attachment_file_names=("図面.pdf",),
        attachment_count=1,
        subject_available=True,
        body_available=False,
        attachments_available=True,
    )
    monkeypatch.setattr(msg_reader, "read_msg_content", lambda _path: content)

    result = msg_reader.build_msg_search_terms(msg_path)

    assert result.primary_terms == ("mail", "利用可能な件名", "図面")
    assert result.auxiliary_terms == ()
    assert not result.fully_parsed
    assert result.warning == "メール本文を利用できませんでした。"
