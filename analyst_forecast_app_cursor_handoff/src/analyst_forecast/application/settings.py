from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_WORKSPACE_RELATIVE = "30_Permanent/★アナリスト調査"


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vault_root: Path
    obsidian_vault_path: Path | None = None
    workspace_relative_path: str = DEFAULT_WORKSPACE_RELATIVE
    database_path: Path | None = None
    cursor_model: str | None = None
    chatgpt_model: str | None = None
    confidence_review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    default_period_months: int = Field(default=6, ge=1, le=120)
    market_provider_order: list[str] = Field(default_factory=lambda: ["yfinance", "fred", "csv"])

    @model_validator(mode="before")
    @classmethod
    def coerce_workspace_root(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        vault_root = data.get("vault_root")
        obsidian = data.get("obsidian_vault_path")
        relative = data.get("workspace_relative_path", DEFAULT_WORKSPACE_RELATIVE)
        if vault_root is None and obsidian is not None:
            data = dict(data)
            data["vault_root"] = Path(os.path.expandvars(os.path.expanduser(str(obsidian)))) / str(
                relative
            )
        return data

    @field_validator("vault_root", "obsidian_vault_path", "database_path", mode="before")
    @classmethod
    def expand_path(cls, value: Any) -> Any:
        if value is None:
            return None
        return Path(os.path.expandvars(os.path.expanduser(str(value))))

    @field_validator("workspace_relative_path")
    @classmethod
    def validate_relative_workspace(cls, value: str) -> str:
        normalized = value.replace("\\", "/").strip()
        if not normalized:
            raise ValueError("workspace_relative_pathは空にできません")
        path = Path(normalized)
        if path.is_absolute() or (len(normalized) >= 2 and normalized[1] == ":"):
            raise ValueError("workspace_relative_pathに絶対パスは使えません")
        if ".." in Path(normalized).parts:
            raise ValueError("workspace_relative_pathに '..' は使えません")
        return normalized

    @property
    def workspace_root(self) -> Path:
        return self.vault_root

    @property
    def database_file(self) -> Path:
        return self.database_path or self.workspace_root / "_system" / "database.sqlite"


def default_config_path() -> Path:
    explicit = os.environ.get("ANALYST_FORECAST_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "AnalystForecast" / "config.yaml"
    return Path.home() / ".config" / "analyst-forecast" / "config.yaml"


def save_settings(settings: AppSettings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json")
    _atomic_write(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def load_settings(path: Path | None = None) -> AppSettings:
    selected = path or default_config_path()
    if not selected.is_file():
        raise FileNotFoundError(
            f"設定ファイルがありません: {selected}\n"
            "次の操作: analyst-forecast init --vault-root <保存先> を実行してください。"
        )
    raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"設定ファイルの内容がYAMLオブジェクトではありません: {selected}")
    return AppSettings.model_validate(raw)


def _atomic_write(path: Path, content: str) -> None:
    from analyst_forecast.application.io_utils import atomic_write_text

    atomic_write_text(path, content)
