"""Round6 migration evidence: PK/legacy hash + DDL-after failure restore."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from alembic import command as alembic_command
from test_round5_migration import _seed_data_bearing_0007, _sha256

from analyst_forecast.infrastructure.db.migration import MigrationError, upgrade_database

# 0007-era tables that must keep PK sets and legacy projections.
LEGACY_TABLES = (
    "analysts",
    "runs",
    "sources",
    "run_sources",
    "ai_artifacts",
    "forecast_groups",
    "forecast_issuances",
    "forecast_components",
    "forecast_evidence",
    "targets",
    "target_mappings",
    "evaluations",
    "evaluation_snapshots",
)


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row[1]) for row in rows]


def _pk_set(conn: sqlite3.Connection, table: str) -> set[tuple[Any, ...]]:
    pk_cols = [
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        if int(row[5]) > 0
    ]
    if not pk_cols:
        return set()
    order = ", ".join(pk_cols)
    rows = conn.execute(f"SELECT {order} FROM {table} ORDER BY {order}").fetchall()
    return {tuple(row) for row in rows}


def _legacy_projection_hash(conn: sqlite3.Connection, table: str, columns: list[str]) -> str:
    order_cols = ", ".join(columns)
    rows = conn.execute(f"SELECT {order_cols} FROM {table} ORDER BY {order_cols}").fetchall()
    canonical = []
    for row in rows:
        item = {}
        for name, value in zip(columns, row, strict=True):
            if value is None:
                item[name] = None
            elif isinstance(value, bytes):
                item[name] = value.hex()
            else:
                item[name] = value
        canonical.append(item)
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snapshot(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        snap: dict[str, Any] = {
            "revision": version[0] if version else None,
            "tables": {},
        }
        for table in LEGACY_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                continue
            cols = _table_columns(conn, table)
            snap["tables"][table] = {
                "pk": _pk_set(conn, table),
                "columns": cols,
                "hash": _legacy_projection_hash(conn, table, cols),
                "count": conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
            }
        return snap


def test_r6_038_039_data_bearing_pk_and_legacy_hash(tmp_path: Path) -> None:
    database = tmp_path / "r6_data.sqlite"
    counts = _seed_data_bearing_0007(database)
    before = _snapshot(database)
    assert before["revision"] == "0007"
    assert before["tables"]["forecast_issuances"]["count"] >= 1
    assert before["tables"]["forecast_components"]["count"] >= 2

    upgrade_database(database)
    after = _snapshot(database)
    assert after["revision"] not in {None, "0007"}

    for table, before_meta in before["tables"].items():
        after_meta = after["tables"][table]
        assert after_meta["pk"] == before_meta["pk"], table
        # Legacy columns only — ignore newly added columns in after hash comparison.
        legacy_cols = before_meta["columns"]
        with sqlite3.connect(database) as conn:
            after_legacy_hash = _legacy_projection_hash(conn, table, legacy_cols)
        assert after_legacy_hash == before_meta["hash"], table
        assert (
            after_meta["count"] == before_meta["count"] == counts.get(table, before_meta["count"])
            or after_meta["count"] == before_meta["count"]
        )

    with sqlite3.connect(database) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_r6_041_042_ddl_after_commit_then_restore(tmp_path: Path) -> None:
    database = tmp_path / "r6_partial.sqlite"
    backup_dir = tmp_path / "backups"
    _seed_data_bearing_0007(database)
    before_hash = _sha256(database)
    before = _snapshot(database)

    def _fake_upgrade(config: object, revision: str) -> None:
        # Use the live DB path (wrapper already made backup).
        with sqlite3.connect(database) as conn:
            conn.execute("ALTER TABLE forecast_issuances ADD COLUMN forced_partial_marker TEXT")
            conn.execute(
                "UPDATE forecast_issuances SET human_readable_summary='CORRUPTED' "
                "WHERE forecast_issuance_id='FCI-000001'"
            )
            conn.commit()
            cols = [
                row[1] for row in conn.execute("PRAGMA table_info(forecast_issuances)").fetchall()
            ]
            assert "forced_partial_marker" in cols
        raise RuntimeError("forced failure after DDL+data commit")

    with (
        mock.patch.object(alembic_command, "upgrade", side_effect=_fake_upgrade),
        pytest.raises(MigrationError),
    ):
        upgrade_database(database, backup_dir=backup_dir)

    assert _sha256(database) == before_hash
    after = _snapshot(database)
    assert after["revision"] == before["revision"]
    with sqlite3.connect(database) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(forecast_issuances)").fetchall()]
        assert "forced_partial_marker" not in cols
        summary = conn.execute(
            "SELECT human_readable_summary FROM forecast_issuances "
            "WHERE forecast_issuance_id='FCI-000001'"
        ).fetchone()[0]
        assert summary != "CORRUPTED"
        assert (
            after["tables"]["forecast_issuances"]["hash"]
            == before["tables"]["forecast_issuances"]["hash"]
        )
