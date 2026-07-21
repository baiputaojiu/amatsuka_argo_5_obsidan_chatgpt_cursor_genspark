"""R4-044: Alembic head upgrade integrity (metadata, FK, row counts)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from analyst_forecast.infrastructure.db.migration import upgrade_database
from analyst_forecast.infrastructure.db.models import Base
from analyst_forecast.infrastructure.db.session import create_sqlite_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = (
    PROJECT_ROOT
    / "src"
    / "analyst_forecast"
    / "infrastructure"
    / "db"
    / "migrations"
)


def _seed_pre_round4_fixture(database: Path) -> dict[str, int]:
    upgrade_database(database, revision="0007")
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
                "監査アナリスト",
                "監査アナリスト",
                "[]",
                "[]",
                "2026-07-20T00:00:00+00:00",
                "2026-07-20T00:00:00+00:00",
                1,
            ),
        )
        connection.execute(
            """
            INSERT INTO runs (
                run_id, analyst_id, period_start, period_end, evaluation_as_of,
                selected_media, focus_targets, ai_environment, model_configuration,
                status, run_path, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RUN-AUDIT-001",
                "A0001",
                "2026-01-01",
                "2026-06-30",
                "2026-07-01",
                '["youtube"]',
                "[]",
                "{}",
                "{}",
                "active",
                "runs/RUN-AUDIT-001",
                "2026-07-20T00:00:00+00:00",
                "2026-07-20T00:00:00+00:00",
                1,
            ),
        )
        connection.commit()
        return {
            "analysts": connection.execute("SELECT COUNT(*) FROM analysts").fetchone()[0],
            "runs": connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        }


def test_r4_044_upgrade_preserves_rows_fk_and_metadata(tmp_path: Path) -> None:
    """R4-044: after upgrade to head — FK check empty, row counts kept, alembic check ok."""
    database = tmp_path / "audit.sqlite"
    before = _seed_pre_round4_fixture(database)
    upgrade_database(database)

    with sqlite3.connect(database) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        assert version == ("0009",)
        after = {
            "analysts": connection.execute("SELECT COUNT(*) FROM analysts").fetchone()[0],
            "runs": connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        }
        assert after == before
        fk_violations = list(connection.execute("PRAGMA foreign_key_check"))
        assert fk_violations == []

        # R4-036 adjacent: made_at nullable after 0009 (notnull=0)
        made_at_col = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(forecast_issuances)")
        }["made_at"]
        assert made_at_col[3] == 0, "made_at must be nullable after 0009"

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    engine = create_sqlite_engine(database)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
    engine.dispose()

    inspector = inspect(create_engine(f"sqlite:///{database}"))
    tables = set(inspector.get_table_names())
    assert "forecast_issuances" in tables
    assert "artifact_applicability" in tables
    model_tables = {t.name for t in Base.metadata.sorted_tables}
    missing = model_tables - tables
    assert not missing, f"missing tables vs metadata: {missing}"
