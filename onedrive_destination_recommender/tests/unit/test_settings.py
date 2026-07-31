import json
from pathlib import Path

import pytest

from onedrive_destination_recommender.settings import (
    RUNTIME_DIRECTORY_NAME,
    SettingsError,
    default_runtime_dir,
    load_settings,
)


def _valid_data(tmp_path: Path) -> dict[str, object]:
    return {
        "current_year_root": str(tmp_path / "current"),
        "previous_year_root": str(tmp_path / "previous"),
        "pending_root": str(tmp_path / "current" / "pending"),
        "candidate_count": 10,
        "excluded_folder_names": ["除外サンプル", "除外サンプル"],
    }


def _write_settings(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_settings_validates_and_deduplicates_exclusions(tmp_path: Path) -> None:
    settings_path = tmp_path / "config.json"
    _write_settings(settings_path, _valid_data(tmp_path))

    settings = load_settings(settings_path)

    assert settings.current_year_root == tmp_path / "current"
    assert settings.previous_year_root == tmp_path / "previous"
    assert settings.pending_root == tmp_path / "current" / "pending"
    assert settings.candidate_count == 10
    assert settings.excluded_folder_names == ("除外サンプル",)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("candidate_count", 0),
        ("candidate_count", True),
        ("excluded_folder_names", [""]),
        ("current_year_root", "relative/path"),
    ],
)
def test_load_settings_rejects_invalid_values(tmp_path: Path, key: str, value: object) -> None:
    data = _valid_data(tmp_path)
    data[key] = value
    settings_path = tmp_path / "config.json"
    _write_settings(settings_path, data)

    with pytest.raises(SettingsError):
        load_settings(settings_path)


def test_load_settings_rejects_missing_and_unknown_keys(tmp_path: Path) -> None:
    data = _valid_data(tmp_path)
    del data["pending_root"]
    data["unknown"] = "value"
    settings_path = tmp_path / "config.json"
    _write_settings(settings_path, data)

    with pytest.raises(SettingsError):
        load_settings(settings_path)


def test_load_settings_rejects_invalid_json(tmp_path: Path) -> None:
    settings_path = tmp_path / "config.json"
    settings_path.write_text("{", encoding="utf-8")

    with pytest.raises(SettingsError, match="JSON"):
        load_settings(settings_path)


def test_load_settings_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="見つかりません"):
        load_settings(tmp_path / "missing.json")


def test_default_runtime_dir_does_not_create_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    runtime_dir = default_runtime_dir()

    assert runtime_dir == tmp_path / RUNTIME_DIRECTORY_NAME
    assert not runtime_dir.exists()
