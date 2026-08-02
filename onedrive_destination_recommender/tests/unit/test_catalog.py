import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from onedrive_destination_recommender import catalog
from onedrive_destination_recommender.catalog import (
    Catalog,
    CatalogError,
    load_catalog,
    scan_catalog,
    update_catalog,
    write_catalog_atomic,
)
from onedrive_destination_recommender.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    current = tmp_path / "current"
    previous = tmp_path / "previous"
    current.mkdir()
    previous.mkdir()
    return Settings(
        current_year_root=current,
        previous_year_root=previous,
        pending_root=current / "pending",
        candidate_count=10,
        excluded_folder_names=("除外サンプル",),
    )


def test_scan_catalog_collects_all_depths_and_excludes_exact_subtrees(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    deepest = settings.current_year_root / "A" / "B" / "C" / "D"
    deepest.mkdir(parents=True)
    (settings.current_year_root / "除外サンプル" / "secret").mkdir(parents=True)
    included_similar_name = settings.current_year_root / "除外サンプル資料"
    included_similar_name.mkdir()
    previous_folder = settings.previous_year_root / "previous-folder"
    previous_folder.mkdir()
    file_path = settings.current_year_root / "A" / "do-not-read.txt"
    file_path.write_text("synthetic", encoding="utf-8")
    timestamp = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    before = file_path.read_bytes()
    result = scan_catalog(settings, scanned_at=timestamp)
    after = file_path.read_bytes()

    folder_paths = set(result.catalog.folders)
    assert str(deepest) in folder_paths
    assert str(included_similar_name) in folder_paths
    assert str(previous_folder) in folder_paths
    assert not any("secret" in path for path in folder_paths)
    assert not any(path.endswith("do-not-read.txt") for path in folder_paths)
    assert result.catalog.folder_count == 6
    assert result.catalog.scanned_at == "2026-07-31T12:00:00+00:00"
    assert result.skipped_count == 0
    assert before == after


def test_scan_catalog_skips_inaccessible_subfolder_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    blocked = settings.current_year_root / "blocked"
    hidden_child = blocked / "hidden"
    hidden_child.mkdir(parents=True)
    visible = settings.previous_year_root / "visible"
    visible.mkdir()
    real_scandir = os.scandir

    def controlled_scandir(path: str | os.PathLike[str]):
        if Path(path) == blocked:
            raise PermissionError("synthetic access denial")
        return real_scandir(path)

    monkeypatch.setattr(catalog.os, "scandir", controlled_scandir)

    result = scan_catalog(settings)

    assert str(blocked) in result.catalog.folders
    assert str(hidden_child) not in result.catalog.folders
    assert str(visible) in result.catalog.folders
    assert result.skipped_count == 1


def test_scan_catalog_does_not_follow_junctions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    junction = settings.current_year_root / "junction"
    (junction / "outside-like-child").mkdir(parents=True)
    included = settings.previous_year_root / "included"
    included.mkdir()
    real_isjunction = os.path.isjunction

    def controlled_isjunction(path: str | os.PathLike[str]) -> bool:
        return Path(path) == junction or real_isjunction(path)

    monkeypatch.setattr(catalog.os.path, "isjunction", controlled_isjunction)

    result = scan_catalog(settings)

    assert str(junction) not in result.catalog.folders
    assert str(junction / "outside-like-child") not in result.catalog.folders
    assert str(included) in result.catalog.folders
    assert result.skipped_count == 0


def test_scan_catalog_stops_when_root_listing_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    real_scandir = os.scandir

    def controlled_scandir(path: str | os.PathLike[str]):
        if Path(path) == settings.current_year_root:
            raise PermissionError("synthetic root denial")
        return real_scandir(path)

    monkeypatch.setattr(catalog.os, "scandir", controlled_scandir)

    with pytest.raises(CatalogError, match="直下"):
        scan_catalog(settings)


def test_scan_catalog_stops_when_result_is_empty(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(CatalogError, match="0件"):
        scan_catalog(settings)


def test_scan_catalog_stops_when_configured_root_is_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.current_year_root.rmdir()

    with pytest.raises(CatalogError, match="存在しません"):
        scan_catalog(settings)


def test_catalog_round_trip_uses_only_three_persisted_fields(tmp_path: Path) -> None:
    catalog_path = tmp_path / "result.json"
    expected = Catalog(
        scanned_at="2026-07-31T12:00:00+00:00",
        folders=(str(tmp_path / "current" / "A"),),
    )

    write_catalog_atomic(expected, catalog_path)
    actual = load_catalog(catalog_path)

    assert actual == expected
    assert set(expected.to_dict()) == {"scanned_at", "folder_count", "folders"}


def test_load_catalog_rejects_mismatched_folder_count(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog-output.json"
    catalog_path.write_text(
        json.dumps(
            {
                "scanned_at": "2026-07-31T12:00:00+00:00",
                "folder_count": 2,
                "folders": [str(tmp_path / "A")],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="一致しません"):
        load_catalog(catalog_path)


def test_atomic_write_failure_preserves_existing_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path = tmp_path / "catalog-output.json"
    original = "previous catalog\n"
    catalog_path.write_text(original, encoding="utf-8")
    replacement = Catalog(scanned_at="2026-07-31T12:00:00+00:00", folders=("C:\\A",))

    def failing_replace(_source: os.PathLike[str], _target: os.PathLike[str]) -> None:
        raise PermissionError("synthetic replace failure")

    monkeypatch.setattr(catalog.os, "replace", failing_replace)

    with pytest.raises(CatalogError, match="維持"):
        write_catalog_atomic(replacement, catalog_path)

    assert catalog_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".catalog-output.json.*.tmp"))


def test_update_catalog_writes_only_after_successful_scan(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (settings.current_year_root / "folder").mkdir()
    catalog_path = tmp_path / "catalog-output.json"

    result = update_catalog(settings, catalog_path=catalog_path)

    assert result.catalog.folder_count == 1
    assert load_catalog(catalog_path) == result.catalog
