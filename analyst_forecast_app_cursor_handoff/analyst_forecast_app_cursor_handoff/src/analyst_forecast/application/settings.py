from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vault_root: Path
    database_path: Path | None = None
    cursor_model: str | None = None
    chatgpt_model: str | None = None
    confidence_review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    default_period_months: int = Field(default=6, ge=1, le=120)
    market_provider_order: list[str] = Field(default_factory=lambda: ["yfinance", "fred", "csv"])

    @field_validator("vault_root", "database_path", mode="before")
    @classmethod
    def expand_path(cls, value: Any) -> Any:
        if value is None:
            return None
        return Path(os.path.expandvars(os.path.expanduser(str(value))))

    @property
    def database_file(self) -> Path:
        return self.database_path or self.vault_root / "_system" / "database.sqlite"


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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
