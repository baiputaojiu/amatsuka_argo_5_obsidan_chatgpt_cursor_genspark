"""Round2: align EvaluationRecord index with Alembic metadata.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-21

- ix_evaluations_component_as_of_method を model と一致するよう維持
- sources.raw_artifact_id の DB FK は SQLite batch 再作成が
  トランザクション内 PRAGMA foreign_keys=OFF で効かないため追加しない。
  参照整合はアプリケーション層と raw_artifact_id 索引で担保する。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_evaluations_component_as_of_method "
            "ON evaluations (forecast_component_id, evaluation_as_of, evaluation_method_version)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_evaluations_component_as_of_method"))
