"""Round5: active lineage uniqueness, correction ops, coverage columns.

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-21

- Partial unique index: one active issuance per lineage_root_id
- forecast_correction_operations audit table
- Backfill lineage_root_id where null
- Mark multi-active lineages as legacy_conflict (do not auto-pick winner)

Requires upgrade_database() FK-safe wrapper from Round5.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE forecast_issuances "
            "SET lineage_root_id = forecast_issuance_id "
            "WHERE lineage_root_id IS NULL"
        )
    )
    # Mark lineages that already have multiple active rows as conflict.
    op.execute(
        sa.text(
            """
            UPDATE forecast_issuances
            SET lifecycle_status = 'legacy_conflict',
                lifecycle_reason = 'round5_multi_active_lineage'
            WHERE lifecycle_status = 'active'
              AND lineage_root_id IN (
                SELECT lineage_root_id
                FROM forecast_issuances
                WHERE lifecycle_status = 'active'
                  AND lineage_root_id IS NOT NULL
                GROUP BY lineage_root_id
                HAVING COUNT(*) > 1
              )
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_forecast_issuances_active_lineage
            ON forecast_issuances(lineage_root_id)
            WHERE lifecycle_status = 'active' AND lineage_root_id IS NOT NULL
            """
        )
    )
    op.create_table(
        "forecast_correction_operations",
        sa.Column("operation_id", sa.String(length=20), primary_key=True),
        sa.Column("review_artifact_id", sa.String(length=20), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("reviewed_forecast_ref", sa.String(length=100), nullable=True),
        sa.Column("corrected_forecast_ref", sa.String(length=100), nullable=True),
        sa.Column("old_issuance_id", sa.String(length=20), nullable=True),
        sa.Column("new_issuance_id", sa.String(length=20), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_forecast_correction_operations_review",
        "forecast_correction_operations",
        ["review_artifact_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forecast_correction_operations_review",
        table_name="forecast_correction_operations",
    )
    op.drop_table("forecast_correction_operations")
    op.execute(sa.text("DROP INDEX IF EXISTS uq_forecast_issuances_active_lineage"))
