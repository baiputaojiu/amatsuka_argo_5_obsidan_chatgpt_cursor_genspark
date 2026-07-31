import os
from pathlib import Path

import pytest

from onedrive_destination_recommender.ranking import (
    RankingError,
    YearScope,
    rank_candidates,
)
from onedrive_destination_recommender.settings import Settings


def _settings(tmp_path: Path, candidate_count: int = 10) -> Settings:
    return Settings(
        current_year_root=tmp_path / "020_FY_CURRENT",
        previous_year_root=tmp_path / "010_FY_PREVIOUS",
        pending_root=tmp_path / "020_FY_CURRENT" / "（Pending）未分類",
        candidate_count=candidate_count,
        excluded_folder_names=("除外サンプル",),
    )


def _folder(root: Path, relative: str) -> str:
    return str(root.joinpath(*relative.split("/")))


def test_ranking_uses_only_relative_path_and_excludes_primary_zero(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    folders = [
        _folder(settings.current_year_root, "設備"),
        _folder(settings.previous_year_root, "補助だけ"),
    ]

    assert rank_candidates(folders, settings, ["2026"], ["補助"]) == ()


def test_auxiliary_terms_break_only_primary_ties(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    folders = [
        _folder(settings.current_year_root, "案件_補助"),
        _folder(settings.current_year_root, "案件_設備"),
        _folder(settings.current_year_root, "案件_通常"),
    ]

    ranked = rank_candidates(folders, settings, ["案件", "設備"], ["補助"])

    assert [candidate.relative_path for candidate in ranked] == [
        os.path.join("案件_設備"),
        os.path.join("案件_補助"),
        os.path.join("案件_通常"),
    ]
    assert ranked[1].matched_auxiliary_terms == ("補助",)


def test_current_year_precedes_previous_even_with_fewer_primary_matches(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    folders = [
        _folder(settings.previous_year_root, "週報_秋田"),
        _folder(settings.current_year_root, "週報"),
    ]

    ranked = rank_candidates(folders, settings, ["週報", "秋田"])

    assert [candidate.year for candidate in ranked] == [
        YearScope.CURRENT,
        YearScope.PREVIOUS,
    ]


def test_ancestor_folding_is_limited_to_same_year(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    current_parent = _folder(settings.current_year_root, "設備")
    current_child = _folder(settings.current_year_root, "設備/設備資料")
    previous_child = _folder(settings.previous_year_root, "設備/設備資料")

    ranked = rank_candidates(
        [current_parent, current_child, previous_child],
        settings,
        ["設備"],
    )

    assert [(candidate.year, candidate.relative_path) for candidate in ranked] == [
        (YearScope.CURRENT, "設備"),
        (YearScope.PREVIOUS, os.path.join("設備", "設備資料")),
    ]


def test_descendant_remains_when_it_has_more_primary_matches(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    parent = _folder(settings.current_year_root, "設備")
    child = _folder(settings.current_year_root, "設備/秋田")

    ranked = rank_candidates([parent, child], settings, ["設備", "秋田"])

    assert [candidate.relative_path for candidate in ranked] == [
        os.path.join("設備", "秋田"),
        "設備",
    ]


def test_three_equal_siblings_fold_to_parent_and_recalculate_display_terms(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    parent = "（Output）定例成果物"
    folders = [
        _folder(settings.previous_year_root, parent),
        _folder(settings.previous_year_root, f"{parent}/20250401_週報_秋田_特別"),
        _folder(settings.previous_year_root, f"{parent}/20250408_週報_秋田"),
        _folder(settings.previous_year_root, f"{parent}/20250415_週報_秋田"),
    ]

    ranked = rank_candidates(folders, settings, ["週報", "秋田"], ["特別", "月報"])

    assert len(ranked) == 1
    folded = ranked[0]
    assert folded.relative_path == parent
    assert folded.primary_match_count == 2
    assert folded.matched_primary_terms == ("週報",)
    assert folded.auxiliary_match_count == 1
    assert folded.matched_auxiliary_terms == ("月報",)


def test_two_nested_siblings_are_not_folded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    folders = [
        _folder(settings.current_year_root, "親/案件_A"),
        _folder(settings.current_year_root, "親/案件_B"),
    ]

    ranked = rank_candidates(folders, settings, ["案件"])

    assert len(ranked) == 2


def test_siblings_do_not_fold_at_year_root_or_across_different_primary_counts(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    folders = [
        _folder(settings.current_year_root, "週報_A"),
        _folder(settings.current_year_root, "週報_B"),
        _folder(settings.current_year_root, "週報_C_秋田"),
    ]

    ranked = rank_candidates(folders, settings, ["週報", "秋田"])

    assert len(ranked) == 3


def test_same_relative_path_in_both_years_is_not_deduplicated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    relative = "（Output）定例成果物"

    ranked = rank_candidates(
        [
            _folder(settings.current_year_root, relative),
            _folder(settings.previous_year_root, relative),
        ],
        settings,
        ["週報"],
    )

    assert [(candidate.year, candidate.relative_path) for candidate in ranked] == [
        (YearScope.CURRENT, relative),
        (YearScope.PREVIOUS, relative),
    ]


def test_weekly_report_scenario_prefers_current_destination(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output = "（Output）定例成果物"
    folders = [_folder(settings.current_year_root, output)]
    folders.extend(
        _folder(settings.previous_year_root, f"{output}/{date}_週報_秋田")
        for date in ("20250401", "20250408", "20250415")
    )

    ranked = rank_candidates(folders, settings, ["20260804", "週報", "秋田"])

    assert ranked[0].year is YearScope.CURRENT
    assert ranked[0].relative_path == output
    assert ranked[1].year is YearScope.PREVIOUS
    assert ranked[1].relative_path == output
    assert ranked[1].primary_match_count == 2


def test_candidate_count_limits_results_after_ranking(tmp_path: Path) -> None:
    settings = _settings(tmp_path, candidate_count=2)
    folders = [
        _folder(settings.current_year_root, "案件_C"),
        _folder(settings.current_year_root, "案件_A"),
        _folder(settings.current_year_root, "案件_B"),
    ]

    ranked = rank_candidates(folders, settings, ["案件"])

    assert [candidate.relative_path for candidate in ranked] == ["案件_A", "案件_B"]


def test_auxiliary_display_is_limited_to_three_terms_without_changing_count(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    folder = _folder(settings.current_year_root, "案件_補助一_補助二_補助三_補助四")

    ranked = rank_candidates(
        [folder],
        settings,
        ["案件"],
        ["補助一", "補助二", "補助三", "補助四"],
    )

    assert ranked[0].auxiliary_match_count == 4
    assert ranked[0].matched_auxiliary_terms == ("補助一", "補助二", "補助三")


def test_ranking_rejects_catalog_path_outside_year_roots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(RankingError, match="年度ルート外"):
        rank_candidates([tmp_path / "outside"], settings, ["outside"])
