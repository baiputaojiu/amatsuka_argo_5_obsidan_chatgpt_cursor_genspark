import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_DIRECTORY_NAME = "OneDriveDestinationRecommender"
SETTINGS_FILE_NAME = "settings.json"
REQUIRED_KEYS = frozenset(
    {
        "current_year_root",
        "previous_year_root",
        "pending_root",
        "candidate_count",
        "excluded_folder_names",
    }
)


class SettingsError(ValueError):
    """Raised when the runtime location or settings file is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings."""

    current_year_root: Path
    previous_year_root: Path
    pending_root: Path
    candidate_count: int
    excluded_folder_names: tuple[str, ...]


def default_runtime_dir() -> Path:
    """Return the per-user runtime directory without creating it."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise SettingsError("環境変数LOCALAPPDATAを取得できませんでした。")
    return Path(local_app_data) / RUNTIME_DIRECTORY_NAME


def default_settings_path() -> Path:
    """Return the default settings path without creating it."""
    return default_runtime_dir() / SETTINGS_FILE_NAME


def _require_absolute_path(data: dict[str, Any], key: str) -> Path:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"{key}には空でない文字列を指定してください。")

    path = Path(value)
    if not path.is_absolute():
        raise SettingsError(f"{key}には絶対パスを指定してください。")
    return path


def _validate_data(data: Any) -> Settings:
    if not isinstance(data, dict):
        raise SettingsError("settings.jsonのルートはJSONオブジェクトにしてください。")

    actual_keys = frozenset(data)
    missing_keys = REQUIRED_KEYS - actual_keys
    unknown_keys = actual_keys - REQUIRED_KEYS
    if missing_keys:
        raise SettingsError(f"settings.jsonの必須項目が不足しています: {sorted(missing_keys)}")
    if unknown_keys:
        raise SettingsError(f"settings.jsonに未対応の項目があります: {sorted(unknown_keys)}")

    current_year_root = _require_absolute_path(data, "current_year_root")
    previous_year_root = _require_absolute_path(data, "previous_year_root")
    pending_root = _require_absolute_path(data, "pending_root")
    if current_year_root == previous_year_root:
        raise SettingsError("今年度フォルダと昨年度フォルダには異なるパスを指定してください。")

    candidate_count = data["candidate_count"]
    if type(candidate_count) is not int or candidate_count <= 0:
        raise SettingsError("candidate_countには1以上の整数を指定してください。")

    excluded_folder_names = data["excluded_folder_names"]
    if not isinstance(excluded_folder_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in excluded_folder_names
    ):
        raise SettingsError("excluded_folder_namesには空でない文字列の一覧を指定してください。")

    return Settings(
        current_year_root=current_year_root,
        previous_year_root=previous_year_root,
        pending_root=pending_root,
        candidate_count=candidate_count,
        excluded_folder_names=tuple(dict.fromkeys(excluded_folder_names)),
    )


def load_settings(settings_path: str | Path | None = None) -> Settings:
    """Load and validate settings without creating or changing any files."""
    path = Path(settings_path) if settings_path is not None else default_settings_path()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise SettingsError(f"settings.jsonが見つかりません: {path}") from exc
    except OSError as exc:
        raise SettingsError(f"settings.jsonを読み取れません: {path}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SettingsError("settings.jsonのJSON形式が不正です。") from exc

    return _validate_data(data)
