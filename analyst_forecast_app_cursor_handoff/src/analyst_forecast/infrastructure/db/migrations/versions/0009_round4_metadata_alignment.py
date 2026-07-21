"""Round4: align forecast_issuances nullable columns with SQLAlchemy metadata.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-21

- made_at / publicly_available_at: nullable for unknown-time issuances
- lifecycle_status / generation: NOT NULL with defaults after 0008 backfill
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE forecast_issuances "
            "SET lifecycle_status = 'active' "
            "WHERE lifecycle_status IS NULL"
        )
    )
    op.execute(
        sa.text("UPDATE forecast_issuances SET generation = 1 WHERE generation IS NULL")
    )

    with op.batch_alter_table("forecast_issuances") as batch:
        batch.alter_column("made_at", existing_type=sa.DateTime(timezone=True), nullable=True)
        batch.alter_column(
            "publicly_available_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=True,
        )
        batch.alter_column(
            "lifecycle_status",
            existing_type=sa.String(length=40),
            nullable=False,
            server_default="active",
        )
        batch.alter_column(
            "generation",
            existing_type=sa.Integer(),
            nullable=False,
            server_default="1",
        )


def downgrade() -> None:
    with op.batch_alter_table("forecast_issuances") as batch:
        batch.alter_column("generation", existing_type=sa.Integer(), nullable=True)
        batch.alter_column(
            "lifecycle_status",
            existing_type=sa.String(length=40),
            nullable=True,
            server_default=None,
        )
        batch.alter_column(
            "publicly_available_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )
        batch.alter_column("made_at", existing_type=sa.DateTime(timezone=True), nullable=False)
