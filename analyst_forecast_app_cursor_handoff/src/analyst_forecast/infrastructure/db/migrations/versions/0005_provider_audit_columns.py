"""provider audit columns for market evaluation

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-20 23:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("market_series", schema=None) as batch_op:
        batch_op.add_column(sa.Column("provider_error_code", sa.String(length=60), nullable=True))
        batch_op.add_column(sa.Column("provider_error_message", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("retryable", sa.String(length=10), nullable=True))
        batch_op.add_column(sa.Column("attempt_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("cache_hit", sa.String(length=10), nullable=True))
        batch_op.create_index(
            "ix_market_series_lookup",
            [
                "provider",
                "symbol",
                "currency",
                "adjustment_type",
                "start_date",
                "end_date",
            ],
            unique=False,
        )

    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("provider_error_code", sa.String(length=60), nullable=True))
        batch_op.add_column(sa.Column("provider_error_message", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("retryable", sa.String(length=10), nullable=True))
        batch_op.add_column(sa.Column("attempt_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("cache_hit", sa.String(length=10), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.drop_column("cache_hit")
        batch_op.drop_column("attempt_count")
        batch_op.drop_column("retryable")
        batch_op.drop_column("provider_error_message")
        batch_op.drop_column("provider_error_code")

    with op.batch_alter_table("market_series", schema=None) as batch_op:
        batch_op.drop_index("ix_market_series_lookup")
        batch_op.drop_column("cache_hit")
        batch_op.drop_column("attempt_count")
        batch_op.drop_column("retryable")
        batch_op.drop_column("provider_error_message")
        batch_op.drop_column("provider_error_code")
