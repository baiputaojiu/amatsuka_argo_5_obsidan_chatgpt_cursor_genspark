from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from analyst_forecast import __version__
from analyst_forecast.application.settings import AppSettings, save_settings
from analyst_forecast.infrastructure.db.migration import upgrade_database
from analyst_forecast.resources import iter_resource_files, read_text_resource


def initialize_workspace(
    settings: AppSettings,
    *,
    config_path: Path,
    update_docs: bool = False,
) -> None:
    root = settings.workspace_root
    system = root / "_system"

    for path in (
        system / "market_cache",
        system / "backups" / "database",
        system / "backups" / "configuration",
        system / "backups" / "mappings",
        system / "raw_artifacts",
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
        "schema_version": "1.1.0",
        "app_version": __version__,
        "workspace_relative_path": settings.workspace_relative_path,
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

    seed_workspace_docs(root, update=update_docs)
    seed_workspace_prompts(root, update=update_docs)

    upgrade_database(
        settings.database_file,
        backup_dir=system / "backups" / "database",
    )


def seed_workspace_docs(root: Path, *, update: bool = False) -> None:
    for relative, source in iter_resource_files("docs"):
        if relative == "README.md":
            destination = root / "README.md"
        elif relative == "AI_WORK_GUIDE.md":
            destination = root / "AI_WORK_GUIDE.md"
        else:
            destination = root / "docs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not update:
            continue
        if destination.exists() and update:
            backup = (
                root
                / "_system"
                / "backups"
                / "configuration"
                / f"docs__{destination.name}__{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")


def seed_workspace_prompts(root: Path, *, update: bool = False) -> None:
    prompts_root = root / "prompts"
    prompts_root.mkdir(parents=True, exist_ok=True)
    for relative, source in iter_resource_files("prompts"):
        destination = prompts_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not update:
            continue
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    catalog = json.loads(read_text_resource("prompts", "catalog.json"))
    (prompts_root / "PROMPT_MANIFEST.json").write_text(
        json.dumps(
            {
                "app_version": __version__,
                "template_version": catalog.get("template_version", "1.0.0"),
                "seeded_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
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
