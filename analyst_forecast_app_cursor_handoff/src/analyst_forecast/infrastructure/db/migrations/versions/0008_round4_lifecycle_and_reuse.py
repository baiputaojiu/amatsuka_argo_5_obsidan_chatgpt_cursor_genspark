"""Round4: forecast issuance lifecycle, source reuse, state machine, cutoff.

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-21

- forecast_issuances: lifecycle_status, supersedes_forecast_issuance_id,
  superseded_at, superseded_by_issuance_id, lifecycle_reason,
  review_artifact_id, generation, lineage_root_id
- run_sources: preprocess_status, p08_review_status (separate axes)
- ai_artifact_applicability: association table for reuse
- evaluations: common_date_count, selected_start_date, selected_end_date,
  evaluation_method_version_detail, coverage_audit (JSON)

Legacy issuances are set to lifecycle_status='active' only when single
per lineage; conflicts marked 'legacy_conflict'.

Downgrade loses lifecycle/reuse data. Backup required.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- ForecastIssuance lifecycle ---
    with op.batch_alter_table("forecast_issuances") as batch:
        batch.add_column(
            sa.Column("lifecycle_status", sa.String(length=40), nullable=True)
        )
        batch.add_column(
            sa.Column("supersedes_forecast_issuance_id", sa.String(length=20), nullable=True)
        )
        batch.add_column(sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column("superseded_by_issuance_id", sa.String(length=20), nullable=True)
        )
        batch.add_column(sa.Column("lifecycle_reason", sa.Text(), nullable=True))
        batch.add_column(sa.Column("review_artifact_id", sa.String(length=20), nullable=True))
        batch.add_column(sa.Column("generation", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("lineage_root_id", sa.String(length=20), nullable=True))
        batch.create_index("ix_forecast_issuances_lifecycle_status", ["lifecycle_status"])
        batch.create_index("ix_forecast_issuances_lineage_root_id", ["lineage_root_id"])

    # Set legacy rows to active (single per lineage assumption for simple cases)
    op.execute(
        sa.text(
            "UPDATE forecast_issuances "
            "SET lifecycle_status = 'active', "
            "generation = 1, "
            "lineage_root_id = forecast_issuance_id "
            "WHERE lifecycle_status IS NULL"
        )
    )

    # --- Run source separate state axes ---
    with op.batch_alter_table("run_sources") as batch:
        batch.add_column(
            sa.Column("preprocess_status", sa.String(length=40), nullable=True)
        )
        batch.add_column(
            sa.Column("p08_review_status", sa.String(length=40), nullable=True)
        )
        batch.add_column(sa.Column("p09_attempt_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("terminal_reason", sa.Text(), nullable=True))

    op.execute(
        sa.text(
            "UPDATE run_sources SET preprocess_status = "
            "CASE WHEN processing_status IN ('accepted', 'processed_no_forecast', "
            "'processed_no_formal_forecast', 'processed_with_forecasts') "
            "THEN 'preprocess_accepted' ELSE 'preprocess_pending' END"
        )
    )

    # --- Artifact applicability (reuse association) ---
    op.create_table(
        "artifact_applicability",
        sa.Column("applicability_id", sa.String(length=20), primary_key=True),
        sa.Column("ai_artifact_id", sa.String(length=20), sa.ForeignKey("ai_artifacts.ai_artifact_id"), nullable=False),
        sa.Column("target_run_id", sa.String(length=32), sa.ForeignKey("runs.run_id"), nullable=False),
        sa.Column("target_source_id", sa.String(length=20), sa.ForeignKey("sources.source_id"), nullable=False),
        sa.Column("reused_from_artifact_id", sa.String(length=20), sa.ForeignKey("ai_artifacts.ai_artifact_id"), nullable=True),
        sa.Column("raw_artifact_id", sa.String(length=20), nullable=True),
        sa.Column("raw_hash", sa.String(length=64), nullable=False),
        sa.Column("applicability_status", sa.String(length=40), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("ai_artifact_id", "target_source_id", name="uq_applicability_artifact_source"),
    )

    # --- Evaluation coverage audit ---
    with op.batch_alter_table("evaluations") as batch:
        batch.add_column(sa.Column("common_date_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("selected_start_date", sa.Date(), nullable=True))
        batch.add_column(sa.Column("selected_end_date", sa.Date(), nullable=True))
        batch.add_column(sa.Column("coverage_audit", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("evaluations") as batch:
        batch.drop_column("coverage_audit")
        batch.drop_column("selected_end_date")
        batch.drop_column("selected_start_date")
        batch.drop_column("common_date_count")

    op.drop_table("artifact_applicability")

    with op.batch_alter_table("run_sources") as batch:
        batch.drop_column("terminal_reason")
        batch.drop_column("p09_attempt_count")
        batch.drop_column("p08_review_status")
        batch.drop_column("preprocess_status")

    with op.batch_alter_table("forecast_issuances") as batch:
        batch.drop_index("ix_forecast_issuances_lineage_root_id")
        batch.drop_index("ix_forecast_issuances_lifecycle_status")
        batch.drop_column("lineage_root_id")
        batch.drop_column("generation")
        batch.drop_column("review_artifact_id")
        batch.drop_column("lifecycle_reason")
        batch.drop_column("superseded_by_issuance_id")
        batch.drop_column("superseded_at")
        batch.drop_column("supersedes_forecast_issuance_id")
        batch.drop_column("lifecycle_status")
