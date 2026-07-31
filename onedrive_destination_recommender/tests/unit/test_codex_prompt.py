from pathlib import Path

from onedrive_destination_recommender.codex_prompt import (
    build_codex_consultation,
    copy_codex_consultation,
)
from onedrive_destination_recommender.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        current_year_root=tmp_path / "020_FY_CURRENT",
        previous_year_root=tmp_path / "010_FY_PREVIOUS",
        pending_root=tmp_path / "020_FY_CURRENT" / "（Pending）未分類",
        candidate_count=10,
        excluded_folder_names=("除外サンプル",),
    )


def test_manual_consultation_requires_no_attachment(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    candidate = settings.current_year_root / "設備"

    result = build_codex_consultation(
        input_files=[],
        settings=settings,
        search_text="設備 カメラ",
        candidate_paths=[candidate],
    )

    assert "添付ファイルはありません" in result.attachment_guidance
    assert "なし（手動検索のみ）" in result.prompt
    assert str(settings.current_year_root.resolve()) in result.prompt
    assert str(settings.previous_year_root.resolve()) in result.prompt
    assert "設備 カメラ" in result.prompt
    assert str(candidate.resolve()) in result.prompt
    assert "ファイルの移動やフォルダの作成は行わず" in result.prompt


def test_msg_consultation_guides_original_msg_attachment(tmp_path: Path) -> None:
    msg_path = tmp_path / "案件.MSG"
    msg_path.write_bytes(b"synthetic")

    result = build_codex_consultation(
        input_files=[msg_path],
        settings=_settings(tmp_path),
        search_text="案件",
        candidate_paths=[],
    )

    assert "元のMSGファイルを添付" in result.attachment_guidance
    assert str(msg_path.resolve()) in result.attachment_guidance
    assert str(msg_path.resolve()) in result.prompt
    assert "該当候補なし" in result.prompt


def test_multiple_file_consultation_lists_all_files(tmp_path: Path) -> None:
    first = tmp_path / "図面.pdf"
    second = tmp_path / "仕様書.docx"
    first.write_bytes(b"synthetic")
    second.write_bytes(b"synthetic")

    result = build_codex_consultation(
        input_files=[first, second],
        settings=_settings(tmp_path),
        search_text="図面 仕様書",
        candidate_paths=[],
    )

    assert "入力した全ファイルを添付" in result.attachment_guidance
    for path in (first, second):
        assert str(path.resolve()) in result.attachment_guidance
        assert str(path.resolve()) in result.prompt


def test_consultation_limits_candidate_paths_to_top_ten(tmp_path: Path) -> None:
    candidates = [tmp_path / f"candidate-{index:02d}" for index in range(11)]

    result = build_codex_consultation(
        input_files=[],
        settings=_settings(tmp_path),
        search_text="案件",
        candidate_paths=candidates,
    )

    assert str(candidates[9].resolve()) in result.prompt
    assert str(candidates[10].resolve()) not in result.prompt


def test_consultation_keeps_disappeared_input_path_for_attachment_guidance(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.msg"

    result = build_codex_consultation(
        input_files=[missing],
        settings=_settings(tmp_path),
        search_text="案件",
        candidate_paths=[],
    )

    assert str(missing.resolve()) in result.attachment_guidance
    assert str(missing.resolve()) in result.prompt


def test_copy_consultation_replaces_clipboard_on_explicit_call(tmp_path: Path) -> None:
    result = build_codex_consultation(
        input_files=[],
        settings=_settings(tmp_path),
        search_text="案件",
        candidate_paths=[],
    )

    class FakeClipboard:
        def __init__(self) -> None:
            self.text = "old"

        def clipboard_clear(self) -> None:
            self.text = ""

        def clipboard_append(self, text: str) -> None:
            self.text += text

    clipboard = FakeClipboard()
    copy_codex_consultation(result, clipboard)

    assert clipboard.text == result.prompt
    assert result.attachment_guidance not in clipboard.text
