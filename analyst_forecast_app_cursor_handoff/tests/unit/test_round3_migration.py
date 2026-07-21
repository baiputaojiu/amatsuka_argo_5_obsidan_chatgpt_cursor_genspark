"""Round3 migration path coverage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from analyst_forecast.infrastructure.db.migration import upgrade_database


def _assert_round3_columns(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        issuance_cols = {
            row[1] for row in connection.execute("PRAGMA table_info(forecast_issuances)")
        }
        evidence_cols = {
            row[1] for row in connection.execute("PRAGMA table_info(forecast_evidence)")
        }
        series_cols = {row[1] for row in connection.execute("PRAGMA table_info(market_series)")}
        analyst_cols = {row[1] for row in connection.execute("PRAGMA table_info(analysts)")}
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert "verified_attribution_status" in issuance_cols
    assert "segment_id" in evidence_cols
    assert "series_kind" in series_cols
    assert "aliases_updated_at" in analyst_cols
    assert version == ("0010",)


def test_empty_database_upgrades_to_head(tmp_path: Path) -> None:
    database = tmp_path / "empty.sqlite"
    upgrade_database(database)
    _assert_round3_columns(database)


def test_upgrade_from_0005_to_head(tmp_path: Path) -> None:
    database = tmp_path / "from0005.sqlite"
    upgrade_database(database, revision="0005")
    upgrade_database(database)
    _assert_round3_columns(database)


def test_upgrade_from_0006_to_head(tmp_path: Path) -> None:
    database = tmp_path / "from0006.sqlite"
    upgrade_database(database, revision="0006")
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO analysts (
                analyst_id, canonical_name, normalized_name, aliases, specialties,
                created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "A0001",
                "匿名アナリストA",
                "匿名アナリストa",
                "[]",
                "[]",
                "2026-07-20T00:00:00+00:00",
                "2026-07-20T00:00:00+00:00",
                1,
            ),
        )
        connection.commit()
    upgrade_database(database)
    _assert_round3_columns(database)
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT analyst_id, aliases_updated_at FROM analysts WHERE analyst_id='A0001'"
        ).fetchone()
    assert row == ("A0001", None)


def test_upgrade_from_0007_to_head(tmp_path: Path) -> None:
    """R4-043: upgrade from 0007 reaches head (0009)."""
    database = tmp_path / "from0007.sqlite"
    upgrade_database(database, revision="0007")
    with sqlite3.connect(database) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert version == ("0007",)
    upgrade_database(database)
    _assert_round3_columns(database)
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "artifact_applicability" in tables
