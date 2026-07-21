"""Round3: speaker attribution, evidence lineage, basket series identity.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-21

- forecast_issuances: speaker / verified attribution / statement_kind / made_at_source
- forecast_evidence: optional segment_id FK
- analysts: aliases_updated_at
- market_series: series_kind / series_identity / basket audit columns
- legacy market_series rows are treated as series_kind=raw
- legacy forecast_issuances without segment link stay unverified
  (verified_attribution_status NULL / legacy_unknown) and must not enter formal aggregates

Downgrade drops added columns; attribution and basket audit data would be lost.
Backup before downgrade is required.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysts") as batch:
        batch.add_column(sa.Column("aliases_updated_at", sa.DateTime(timezone=True), nullable=True))

    with op.batch_alter_table("forecast_issuances") as batch:
        batch.add_column(sa.Column("made_at_source", sa.String(length=40), nullable=True))
        batch.add_column(sa.Column("speaker_candidate", sa.String(length=200), nullable=True))
        batch.add_column(
            sa.Column("speaker_attribution_status", sa.String(length=40), nullable=True)
        )
        batch.add_column(
            sa.Column("verified_attribution_status", sa.String(length=40), nullable=True)
        )
        batch.add_column(sa.Column("attribution_confidence", sa.Float(), nullable=True))
        batch.add_column(sa.Column("attribution_basis", sa.Text(), nullable=True))
        batch.add_column(sa.Column("statement_kind", sa.String(length=40), nullable=True))
        batch.add_column(sa.Column("attribution_verification_reason", sa.Text(), nullable=True))
        batch.create_index(
            "ix_forecast_issuances_verified_attribution_status",
            ["verified_attribution_status"],
        )

    # legacy rows: mark as legacy_unknown so they stay out of formal aggregates
    op.execute(
        sa.text(
            "UPDATE forecast_issuances "
            "SET verified_attribution_status = 'legacy_unknown', "
            "speaker_attribution_status = COALESCE(speaker_attribution_status, 'legacy_unknown'), "
            "statement_kind = COALESCE(statement_kind, 'legacy_unknown') "
            "WHERE verified_attribution_status IS NULL"
        )
    )

    with op.batch_alter_table("forecast_evidence") as batch:
        batch.add_column(sa.Column("segment_id", sa.String(length=20), nullable=True))
        batch.create_foreign_key(
            "fk_forecast_evidence_segment_id",
            "segments",
            ["segment_id"],
            ["segment_id"],
        )
        batch.create_index("ix_forecast_evidence_segment_id", ["segment_id"])

    with op.batch_alter_table("market_series") as batch:
        batch.add_column(
            sa.Column("series_kind", sa.String(length=20), nullable=False, server_default="raw")
        )
        batch.add_column(sa.Column("series_identity", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("mapping_hash", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("input_series_hashes", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("basket_weights", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("common_date_rule", sa.String(length=80), nullable=True))
        batch.create_index("ix_market_series_series_kind", ["series_kind"])
        batch.create_index("ix_market_series_series_identity", ["series_identity"])
        batch.create_index("ix_market_series_mapping_hash", ["mapping_hash"])
        batch.create_index(
            "ix_market_series_kind_lookup",
            [
                "series_kind",
                "provider",
                "symbol",
                "currency",
                "start_date",
                "end_date",
            ],
        )

    op.execute(
        sa.text(
            "UPDATE market_series SET series_kind = 'raw' "
            "WHERE series_kind IS NULL OR series_kind = ''"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("market_series") as batch:
        batch.drop_index("ix_market_series_kind_lookup")
        batch.drop_index("ix_market_series_mapping_hash")
        batch.drop_index("ix_market_series_series_identity")
        batch.drop_index("ix_market_series_series_kind")
        batch.drop_column("common_date_rule")
        batch.drop_column("basket_weights")
        batch.drop_column("input_series_hashes")
        batch.drop_column("mapping_hash")
        batch.drop_column("series_identity")
        batch.drop_column("series_kind")

    with op.batch_alter_table("forecast_evidence") as batch:
        batch.drop_index("ix_forecast_evidence_segment_id")
        batch.drop_constraint("fk_forecast_evidence_segment_id", type_="foreignkey")
        batch.drop_column("segment_id")

    with op.batch_alter_table("forecast_issuances") as batch:
        batch.drop_index("ix_forecast_issuances_verified_attribution_status")
        batch.drop_column("attribution_verification_reason")
        batch.drop_column("statement_kind")
        batch.drop_column("attribution_basis")
        batch.drop_column("attribution_confidence")
        batch.drop_column("verified_attribution_status")
        batch.drop_column("speaker_attribution_status")
        batch.drop_column("speaker_candidate")
        batch.drop_column("made_at_source")

    with op.batch_alter_table("analysts") as batch:
        batch.drop_column("aliases_updated_at")
