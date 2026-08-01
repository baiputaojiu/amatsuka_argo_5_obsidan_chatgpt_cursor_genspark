import os
import sys
from pathlib import Path

import pytest

from onedrive_destination_recommender.msg_reader import (
    build_msg_search_terms,
    probe_msg_access,
)

pytestmark = pytest.mark.integration


def test_outlook_com_can_access_local_msg_fields() -> None:
    configured_path = os.environ.get("ODR_TEST_MSG_PATH")
    if not configured_path:
        pytest.skip("ODR_TEST_MSG_PATH is not configured")

    result = probe_msg_access(Path(configured_path))

    assert result.subject_accessible
    assert result.body_accessible
    assert isinstance(result.attachment_count, int)


@pytest.mark.skipif(sys.platform != "win32", reason="Outlook COM Windows test")
def test_outlook_com_reads_synthetic_msg_and_filters_inline_image(tmp_path: Path) -> None:
    import win32com.client

    inline_image = tmp_path / "image1.png"
    inline_image.write_bytes(b"synthetic image placeholder")
    report = tmp_path / "Layout.pdf"
    report.write_bytes(b"synthetic report placeholder")
    msg_path = tmp_path / "synthetic-mail.msg"

    outlook = win32com.client.Dispatch("Outlook.Application")
    item = outlook.CreateItem(0)
    try:
        item.To = "synthetic@example.invalid"
        item.Subject = "設備 カメラ"
        item.Body = "秋田 2025\nMicrosoft Teams 会議に参加\n-----Original Message-----\n過去本文"
        item.Attachments.Add(str(inline_image))
        item.Attachments.Add(str(report))
        item.SaveAs(str(msg_path), 3)
    finally:
        item.Close(1)

    result = build_msg_search_terms(msg_path)

    assert result.fully_parsed
    assert "layout" in result.primary_terms
    assert "image1" not in result.primary_terms
    assert "秋田" in result.auxiliary_terms
    assert "microsoft" not in result.auxiliary_terms
    assert "過去本文" not in result.auxiliary_terms
