import sqlite3
from pathlib import Path

from analyst_forecast.infrastructure.db.migration import upgrade_database


def test_existing_0001_database_upgrades_without_losing_run_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.sqlite"
    upgrade_database(database, revision="0001")
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
        connection.execute(
            """
            INSERT INTO runs (
                run_id, analyst_id, period_start, period_end, evaluation_as_of,
                selected_media, focus_targets, ai_environment, model_configuration,
                status, run_path, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "RUN-20260720-001",
                "A0001",
                "2026-01-01",
                "2026-06-30",
                "2026-07-20",
                '["youtube"]',
                "[]",
                '["cursor"]',
                "{}",
                "not_started",
                "analysts/fixture/RUN-20260720-001",
                "2026-07-20T00:00:00+00:00",
                "2026-07-20T00:00:00+00:00",
                1,
            ),
        )
        connection.execute(
            """
            INSERT INTO sources (
                source_id, analyst_id, medium, retrieved_at, raw_file_path, raw_hash,
                acquisition_status, source_relation, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SRC-000001",
                "A0001",
                "youtube",
                "2026-07-20T00:00:00+00:00",
                "analysts/fixture/raw.txt",
                "a" * 64,
                "acquired",
                "original",
                "2026-07-20T00:00:00+00:00",
                "2026-07-20T00:00:00+00:00",
                1,
            ),
        )
        connection.execute(
            """
            INSERT INTO run_sources (
                run_id, source_id, observed_medium, linked_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "RUN-20260720-001",
                "SRC-000001",
                "youtube",
                "2026-07-20T00:00:00+00:00",
            ),
        )
        connection.commit()

    upgrade_database(database)

    with sqlite3.connect(database) as connection:
        status = connection.execute(
            """
            SELECT processing_status
            FROM run_sources
            WHERE run_id = 'RUN-20260720-001' AND source_id = 'SRC-000001'
            """
        ).fetchone()
        artifact_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'ai_artifacts'
            """
        ).fetchone()
    assert status == ("raw_imported",)
    assert artifact_table == ("ai_artifacts",)
