"""Unit tests: config.json migration (Ch34.2 mandatory)."""

import json
import pytest
from pathlib import Path
from outlook_google_sync.config.migration import migrate_config
from outlook_google_sync.models.config_schema import SCHEMA_VERSION


def test_migration_from_old_version(tmp_path):
    """Backward compatible: old version gets defaults filled in."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"schema_version": 0}), encoding="utf-8")
    result = migrate_config(cfg)
    assert result["schema_version"] == SCHEMA_VERSION
    assert "runtime_state" in result
    assert "sync_metadata" in result


def test_migration_current_version(tmp_path):
    """Current version passes through unchanged."""
    cfg = tmp_path / "config.json"
    data = {"schema_version": SCHEMA_VERSION, "runtime_state": {"x": 1}, "sync_metadata": {}}
    cfg.write_text(json.dumps(data), encoding="utf-8")
    result = migrate_config(cfg)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["runtime_state"]["x"] == 1


def test_migration_future_version_raises(tmp_path):
    """Incompatible future version raises and creates backup."""
    cfg = tmp_path / "config.json"
    future = SCHEMA_VERSION + 10
    cfg.write_text(json.dumps({"schema_version": future}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        migrate_config(cfg)
    backup = tmp_path / f"config.backup.v{future}.json"
    assert backup.exists()


def test_migration_no_file(tmp_path):
    """Missing file returns defaults."""
    cfg = tmp_path / "config.json"
    result = migrate_config(cfg)
    assert result["schema_version"] == SCHEMA_VERSION
