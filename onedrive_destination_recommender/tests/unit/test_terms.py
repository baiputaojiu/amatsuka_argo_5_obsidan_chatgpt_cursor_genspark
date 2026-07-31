from onedrive_destination_recommender.terms import (
    clean_msg_body,
    initial_terms_from_file_names,
    is_inline_image_attachment,
    normalize_path_for_matching,
    normalize_terms,
    searchable_attachment_names,
)


def test_normalize_terms_applies_nfkc_casefold_boundaries_and_deduplication() -> None:
    terms = normalize_terms("ＡＢＣ分類 20XX0101_定例資料_サンプル.pptx A abc")

    assert terms == ("abc", "分類", "20260804", "週報", "秋田")


def test_normalize_terms_drops_short_terms_but_keeps_short_numbers_for_primary() -> None:
    assert normalize_terms("A 棟 12 AB 設備") == ("12", "ab", "設備")


def test_normalize_terms_drops_auxiliary_numbers_shorter_than_four_digits() -> None:
    assert normalize_terms("12 123 1234 設備", auxiliary=True) == ("1234", "設備")


def test_normalize_terms_does_not_treat_decimal_like_manual_terms_as_extensions() -> None:
    assert normalize_terms("3.5inch Rev1.2_図面 report.pdf") == (
        "5inch",
        "rev1",
        "図面",
        "report",
    )


def test_initial_terms_from_multiple_file_names_does_not_read_files() -> None:
    assert initial_terms_from_file_names(
        ["20XX0101_定例資料_サンプル.pptx", "Camera-Layout.PDF", "archive.tar.7z"]
    ) == (
        "20260804",
        "週報",
        "秋田",
        "camera",
        "layout",
        "archive",
    )


def test_normalize_path_for_matching_preserves_short_fragments_for_partial_match() -> None:
    assert normalize_path_for_matching("設備A/週報") == "設備 a 週報"


def test_clean_msg_body_removes_original_message_and_reply_header_history() -> None:
    original_history = "確認してください。\n-----Original Message-----\n古い本文"
    header_history = "現在の本文\nFrom: sender@example.com\nSent: today\nTo: user\n古い本文"

    assert clean_msg_body(original_history) == "確認してください。"
    assert clean_msg_body(header_history) == "現在の本文"


def test_clean_msg_body_removes_quotes_meeting_lines_signature_and_contacts() -> None:
    body = """確認してください。
連絡先 user@example.com https://example.com 03-1234-5678
> 引用文
Microsoft Teams 会議に参加
会議 ID: 123
パスコード: 456
Join the meeting now
---
署名
"""

    cleaned = clean_msg_body(body)

    assert "確認してください。" in cleaned
    assert "連絡先" in cleaned
    assert "example" not in cleaned
    assert "03-1234-5678" not in cleaned
    assert "引用文" not in cleaned
    assert "Teams" not in cleaned
    assert "署名" not in cleaned


def test_clean_msg_body_limits_result_after_cleaning() -> None:
    assert clean_msg_body("A" * 2100) == "A" * 2000


def test_inline_image_attachment_pattern_uses_name_and_supported_extension() -> None:
    assert is_inline_image_attachment("IMAGE12.PNG")
    assert is_inline_image_attachment("ｉｍａｇｅ１２.jpg")
    assert not is_inline_image_attachment("image12.pdf")
    assert not is_inline_image_attachment("camera12.png")
    assert searchable_attachment_names(
        ["image1.png", "図面.pdf", "camera12.png", "IMAGE2.EMZ"]
    ) == ("図面.pdf", "camera12.png")
