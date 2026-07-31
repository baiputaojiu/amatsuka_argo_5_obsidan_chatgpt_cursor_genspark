from onedrive_destination_recommender.terms import (
    initial_terms_from_file_names,
    normalize_path_for_matching,
    normalize_terms,
)


def test_normalize_terms_applies_nfkc_casefold_boundaries_and_deduplication() -> None:
    terms = normalize_terms("ＡＢＣ分類 20XX0101_定例資料_サンプル.pptx A abc")

    assert terms == ("abc", "分類", "20260804", "週報", "秋田")


def test_normalize_terms_drops_short_terms_but_keeps_short_numbers_for_primary() -> None:
    assert normalize_terms("A 棟 12 AB 設備") == ("12", "ab", "設備")


def test_normalize_terms_drops_auxiliary_numbers_shorter_than_four_digits() -> None:
    assert normalize_terms("12 123 1234 設備", auxiliary=True) == ("1234", "設備")


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
