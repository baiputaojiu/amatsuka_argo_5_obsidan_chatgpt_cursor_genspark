"""Round5 migration tests: data-bearing 0007→head, 0009→head, restore-on-failure."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from unittest import mock

import pytest
from alembic import command as alembic_command

from analyst_forecast.infrastructure.db.migration import MigrationError, upgrade_database


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_data_bearing_0007(database: Path) -> dict[str, int]:
    """Seed full child-row graph on 0007 schema (migration-test-only SQL)."""
    upgrade_database(database, revision="0007")
    ts = "2026-07-20T00:00:00+00:00"
    hx = "a" * 64
    # Keep JSON outside the f-string to avoid brace-escape bugs.
    empty_json = "{}"
    instruments_json = '[{"symbol":"N225","currency":"JPY","weight":1.0}]'
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            f"""
            INSERT INTO analysts (
                analyst_id, canonical_name, normalized_name, aliases, specialties,
                created_at, updated_at, version
            ) VALUES (
                'A0001', '監査アナリスト', '監査アナリスト', '[]', '[]',
                '{ts}', '{ts}', 1
            );
            INSERT INTO runs (
                run_id, analyst_id, period_start, period_end, evaluation_as_of,
                selected_media, focus_targets, ai_environment, model_configuration,
                status, run_path, created_at, updated_at, version
            ) VALUES (
                'RUN-20260101-001', 'A0001', '2026-01-01', '2026-06-30', '2026-07-01',
                '["youtube"]', '[]', '{empty_json}', '{empty_json}', 'active',
                'runs/RUN-20260101-001', '{ts}', '{ts}', 1
            );
            INSERT INTO sources (
                source_id, analyst_id, medium, url, title, recorded_at, published_at,
                retrieved_at, evidence_level, raw_file_path, raw_hash, acquisition_status,
                source_relation, created_at, updated_at, version
            ) VALUES (
                'SRC-000001', 'A0001', 'youtube', 'https://example.invalid/v1', 'fixture',
                '{ts}', '{ts}', '{ts}', 'A', 'raw/a.txt', '{hx}', 'acquired',
                'original', '{ts}', '{ts}', 1
            );
            INSERT INTO run_sources (
                run_id, source_id, observed_medium, linked_at, processing_status
            ) VALUES (
                'RUN-20260101-001', 'SRC-000001', 'youtube', '{ts}', 'accepted'
            );
            INSERT INTO prompt_executions (
                prompt_execution_id, run_id, prompt_id, prompt_version, environment,
                model, input_files, output_file, executed_at, validation_status,
                created_at, updated_at, version
            ) VALUES (
                'PEX-000001', 'RUN-20260101-001', 'P08', '2.0.0', 'cursor',
                'fixture', '[]', 'out.json', '{ts}', 'accepted', '{ts}', '{ts}', 1
            );
            INSERT INTO ai_artifacts (
                ai_artifact_id, run_id, source_id, prompt_execution_id, prompt_id,
                schema_version, input_hash, output_hash, classified_file_path,
                classification, resolution_status, importance,
                payload, created_at, updated_at, version
            ) VALUES (
                'AIF-000001', 'RUN-20260101-001', 'SRC-000001', 'PEX-000001', 'P08',
                '2.1.0', '{hx}', '{hx}', 'out.json',
                'accepted', 'accepted', 'normal',
                '{empty_json}', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_groups (
                forecast_group_id, analyst_id, central_thesis, first_issued_at,
                latest_issued_at, current_stance, reaffirmation_count, revision_count,
                withdrawal_status, created_at, updated_at, version
            ) VALUES (
                'FCG-000001', 'A0001', 'thesis', '{ts}', '{ts}', 'up', 0, 0,
                'active', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_issuances (
                forecast_issuance_id, analyst_id, forecast_group_id, ai_artifact_id,
                source_id, local_ref, made_at, publicly_available_at, forecast_type,
                commitment_strength, evidence_level, extraction_confidence,
                human_readable_summary, relation_to_previous, current_status,
                created_at, updated_at, version
            ) VALUES (
                'FCI-000001', 'A0001', 'FCG-000001', 'AIF-000001', 'SRC-000001',
                'forecast-a', '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9,
                'summary', 'initial', 'active', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_evidence (
                forecast_evidence_id, forecast_issuance_id, source_id, quote,
                start_offset, end_offset, role, created_at, updated_at, version
            ) VALUES (
                'FCE-000001', 'FCI-000001', 'SRC-000001', 'quote', 0, 5, 'prediction',
                '{ts}', '{ts}', 1
            );
            INSERT INTO targets (
                target_id, raw_label, canonical_name, target_type, ticker, currency,
                aliases, created_at, updated_at, version
            ) VALUES (
                'TGT-000001', '日経平均', '日経平均', 'index', 'N225', 'JPY',
                '[]', '{ts}', '{ts}', 1
            );
            INSERT INTO target_mappings (
                target_mapping_id, target_id, mapping_method, evaluation_instruments,
                knowledge_cutoff, source_evidence, mapping_status, mapping_hash,
                locked_at, created_at, updated_at, version
            ) VALUES (
                'TMP-000001', 'TGT-000001', 'manual',
                '{instruments_json}',
                '{ts}', 'fixture', 'verified', '{hx}',
                '{ts}', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_components (
                forecast_component_id, forecast_issuance_id, local_ref, sequence_number,
                prediction_form, direction, time_source, raw_target_label,
                target_resolution_status, target_id, target_mapping_id, importance,
                created_at, updated_at, version
            ) VALUES
            (
                'FCC-000001', 'FCI-000001', 'component-a', 1, 'period_direction',
                'up', 'explicit', '日経平均', 'locked', 'TGT-000001', 'TMP-000001',
                'normal', '{ts}', '{ts}', 1
            ),
            (
                'FCC-000002', 'FCI-000001', 'component-b', 2, 'period_direction',
                'up', 'explicit', 'TOPIX', 'pending', NULL, NULL,
                'normal', '{ts}', '{ts}', 1
            );
            INSERT INTO evaluations (
                evaluation_id, forecast_component_id, target_mapping_id,
                evaluation_method_version, evaluation_as_of, evaluation_status,
                created_at, updated_at, version
            ) VALUES (
                'EVAL-000001', 'FCC-000001', 'TMP-000001', 'direction-v2.0.0',
                '2026-07-01', 'unevaluable', '{ts}', '{ts}', 1
            );
            INSERT INTO evaluation_snapshots (
                evaluation_snapshot_id, evaluation_id, snapshot_at, status,
                created_at, updated_at, version
            ) VALUES (
                'EVS-000001', 'EVAL-000001', '2026-07-01', 'unevaluable',
                '{ts}', '{ts}', 1
            );
            """
        )
        conn.commit()
        return {
            "issuances": conn.execute("SELECT COUNT(*) FROM forecast_issuances").fetchone()[0],
            "components": conn.execute("SELECT COUNT(*) FROM forecast_components").fetchone()[0],
            "evidence": conn.execute("SELECT COUNT(*) FROM forecast_evidence").fetchone()[0],
            "evaluations": conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
            "snapshots": conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0],
        }


def test_r5_001_data_bearing_0007_upgrades_to_head(tmp_path: Path) -> None:
    database = tmp_path / "bearing.sqlite"
    backup_dir = tmp_path / "backups"
    before = _seed_data_bearing_0007(database)
    upgrade_database(database, backup_dir=backup_dir)
    with sqlite3.connect(database) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        assert version == ("0010",)
        after = {
            "issuances": conn.execute("SELECT COUNT(*) FROM forecast_issuances").fetchone()[0],
            "components": conn.execute("SELECT COUNT(*) FROM forecast_components").fetchone()[0],
            "evidence": conn.execute("SELECT COUNT(*) FROM forecast_evidence").fetchone()[0],
            "evaluations": conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
            "snapshots": conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0],
        }
        assert after == before
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert list(conn.execute("PRAGMA foreign_key_check")) == []
        fk_list = list(conn.execute("PRAGMA foreign_key_list(forecast_components)"))
        assert any(row[2] == "forecast_issuances" for row in fk_list)
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(forecast_issuances)")}
        assert "uq_forecast_issuances_active_lineage" in indexes
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='forecast_correction_operations'"
        ).fetchone()


def test_r5_005_0009_to_head_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "from0009.sqlite"
    upgrade_database(database, revision="0009")
    upgrade_database(database)
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("0010",)
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(forecast_issuances)")}
        assert "uq_forecast_issuances_active_lineage" in indexes
    upgrade_database(database)
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("0010",)


def test_r5_006_migration_failure_restores_backup(tmp_path: Path) -> None:
    database = tmp_path / "restore.sqlite"
    backup_dir = tmp_path / "backups"
    _seed_data_bearing_0007(database)
    pre_hash = _sha256(database)
    with sqlite3.connect(database) as conn:
        pre_version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        pre_cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_issuances)")}
    assert "lifecycle_status" not in pre_cols

    def _boom(config, revision):  # type: ignore[no-untyped-def]
        raise RuntimeError("forced migration failure")

    with (
        mock.patch.object(alembic_command, "upgrade", side_effect=_boom),
        pytest.raises(MigrationError),
    ):
        upgrade_database(database, backup_dir=backup_dir)

    assert _sha256(database) == pre_hash
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == pre_version
        cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_issuances)")}
        assert "lifecycle_status" not in cols


def test_r5_011_active_lineage_unique_rejects_duplicate(tmp_path: Path) -> None:
    database = tmp_path / "uniq.sqlite"
    upgrade_database(database)
    ts = "2026-07-20T00:00:00+00:00"
    hx = "e" * 64
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            f"""
            INSERT INTO analysts (
                analyst_id, canonical_name, normalized_name, aliases, specialties,
                created_at, updated_at, version
            ) VALUES ('A0001', 'A', 'a', '[]', '[]', '{ts}', '{ts}', 1);
            INSERT INTO forecast_groups (
                forecast_group_id, analyst_id, central_thesis, first_issued_at,
                latest_issued_at, current_stance, reaffirmation_count, revision_count,
                withdrawal_status, created_at, updated_at, version
            ) VALUES (
                'FCG-000001', 'A0001', 't', '{ts}', '{ts}', 'up', 0, 0,
                'active', '{ts}', '{ts}', 1
            );
            INSERT INTO sources (
                source_id, analyst_id, medium, url, title, recorded_at, published_at,
                retrieved_at, evidence_level, raw_file_path, raw_hash, acquisition_status,
                source_relation, created_at, updated_at, version
            ) VALUES (
                'SRC-000001', 'A0001', 'youtube', 'https://example.invalid/x', 't',
                '{ts}', '{ts}', '{ts}', 'A', 'raw/x.txt', '{hx}', 'acquired',
                'original', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_issuances (
                forecast_issuance_id, analyst_id, forecast_group_id, source_id,
                local_ref, made_at, publicly_available_at, forecast_type,
                commitment_strength, evidence_level, extraction_confidence,
                human_readable_summary, relation_to_previous, current_status,
                lifecycle_status, generation, lineage_root_id,
                created_at, updated_at, version
            ) VALUES (
                'FCI-000001', 'A0001', 'FCG-000001', 'SRC-000001', 'f1',
                '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's', 'initial',
                'active', 'active', 1, 'ROOT-1', '{ts}', '{ts}', 1
            );
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"""
                INSERT INTO forecast_issuances (
                    forecast_issuance_id, analyst_id, forecast_group_id, source_id,
                    local_ref, made_at, publicly_available_at, forecast_type,
                    commitment_strength, evidence_level, extraction_confidence,
                    human_readable_summary, relation_to_previous, current_status,
                    lifecycle_status, generation, lineage_root_id,
                    created_at, updated_at, version
                ) VALUES (
                    'FCI-000002', 'A0001', 'FCG-000001', 'SRC-000001', 'f2',
                    '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's', 'initial',
                    'active', 'active', 1, 'ROOT-1', '{ts}', '{ts}', 1
                )
                """
            )


def test_r5_007_legacy_conflict_not_auto_promoted(tmp_path: Path) -> None:
    """R5-007: multi-active same lineage → legacy_conflict; none stays active."""
    database = tmp_path / "legacy_conflict.sqlite"
    upgrade_database(database, revision="0009")
    ts = "2026-07-20T00:00:00+00:00"
    hx = "c" * 64
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            f"""
            INSERT INTO analysts (
                analyst_id, canonical_name, normalized_name, aliases, specialties,
                created_at, updated_at, version
            ) VALUES ('A0001', 'A', 'a', '[]', '[]', '{ts}', '{ts}', 1);
            INSERT INTO forecast_groups (
                forecast_group_id, analyst_id, central_thesis, first_issued_at,
                latest_issued_at, current_stance, reaffirmation_count, revision_count,
                withdrawal_status, created_at, updated_at, version
            ) VALUES (
                'FCG-000001', 'A0001', 't', '{ts}', '{ts}', 'up', 0, 0,
                'active', '{ts}', '{ts}', 1
            );
            INSERT INTO sources (
                source_id, analyst_id, medium, url, title, recorded_at, published_at,
                retrieved_at, evidence_level, raw_file_path, raw_hash, acquisition_status,
                source_relation, created_at, updated_at, version
            ) VALUES (
                'SRC-000001', 'A0001', 'youtube', 'https://example.invalid/x', 't',
                '{ts}', '{ts}', '{ts}', 'A', 'raw/x.txt', '{hx}', 'acquired',
                'original', '{ts}', '{ts}', 1
            );
            INSERT INTO forecast_issuances (
                forecast_issuance_id, analyst_id, forecast_group_id, source_id,
                local_ref, made_at, publicly_available_at, forecast_type,
                commitment_strength, evidence_level, extraction_confidence,
                human_readable_summary, relation_to_previous, current_status,
                lifecycle_status, generation, lineage_root_id,
                created_at, updated_at, version
            ) VALUES
            (
                'FCI-000001', 'A0001', 'FCG-000001', 'SRC-000001', 'f1',
                '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's', 'initial',
                'active', 'active', 1, 'ROOT-LEGACY', '{ts}', '{ts}', 1
            ),
            (
                'FCI-000002', 'A0001', 'FCG-000001', 'SRC-000001', 'f2',
                '{ts}', '{ts}', 'directional', 'explicit', 'A', 0.9, 's', 'initial',
                'active', 'active', 1, 'ROOT-LEGACY', '{ts}', '{ts}', 1
            );
            """
        )
        conn.commit()

    upgrade_database(database)
    with sqlite3.connect(database) as conn:
        rows = list(
            conn.execute(
                "SELECT forecast_issuance_id, lifecycle_status, lifecycle_reason "
                "FROM forecast_issuances ORDER BY forecast_issuance_id"
            )
        )
        assert len(rows) == 2
        assert {r[1] for r in rows} == {"legacy_conflict"}
        assert all(r[2] == "round5_multi_active_lineage" for r in rows)
        active = conn.execute(
            "SELECT COUNT(*) FROM forecast_issuances WHERE lifecycle_status='active'"
        ).fetchone()[0]
        assert active == 0
