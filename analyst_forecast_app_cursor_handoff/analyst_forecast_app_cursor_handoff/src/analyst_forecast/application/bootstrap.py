from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from analyst_forecast import __version__
from analyst_forecast.application.settings import AppSettings, save_settings
from analyst_forecast.infrastructure.db.migration import upgrade_database


def initialize_workspace(settings: AppSettings, *, config_path: Path) -> None:
    root = settings.vault_root
    system = root / "_system"

    for path in (
        system / "market_cache",
        system / "backups" / "database",
        system / "backups" / "configuration",
        system / "backups" / "mappings",
        root / "docs",
        root / "prompts",
        root / "analysts",
    ):
        path.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = system / "backups" / "configuration" / f"config__{stamp}.yaml"
        shutil.copy2(config_path, backup)
    save_settings(settings, config_path)

    portable_config = {
        "schema_version": "1.0.0",
        "app_version": __version__,
        "vault_root": "..",
        "database_path": "database.sqlite",
        "confidence_review_threshold": settings.confidence_review_threshold,
        "default_period_months": settings.default_period_months,
        "market_provider_order": settings.market_provider_order,
        "note": "実パスとモデル名はローカル設定を正本とし、このsnapshotへ保存しません。",
    }
    _atomic_yaml(system / "config.yaml", portable_config)

    initial_yaml: dict[str, object] = {"schema_version": "1.0.0", "items": []}
    for filename in (
        "analysts.yaml",
        "forecast_targets.yaml",
        "target_mappings.yaml",
    ):
        _write_yaml_if_missing(system / filename, initial_yaml)

    _write_yaml_if_missing(
        system / "prompt_versions.yaml",
        {
            "schema_version": "1.0.0",
            "prompts": {
                prompt_id: {"version": "1.0.0"} for prompt_id in ("P05", "P08", "P11", "P12")
            },
        },
    )
    _write_yaml_if_missing(
        system / "task_catalog.yaml",
        {
            "schema_version": "1.0.0",
            "tasks": [
                "import_raw",
                "extract_forecasts",
                "validate_ai_output",
                "evaluate_direction",
                "review_results",
            ],
        },
    )
    _write_text_if_missing(
        root / "README.md",
        "# アナリスト予想検証\n\n"
        "この領域はCLIが生成する個人用データ領域です。"
        "`_system/database.sqlite` が機械処理上の正本です。\n",
    )

    upgrade_database(
        settings.database_file,
        backup_dir=system / "backups" / "database",
    )


def _atomic_yaml(path: Path, data: object) -> None:
    content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_yaml_if_missing(path: Path, data: object) -> None:
    if not path.exists():
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )


def _write_text_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8", newline="\n")
