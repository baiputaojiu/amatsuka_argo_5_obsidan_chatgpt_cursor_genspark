"""raw artifact and source occurrence separation

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-20 23:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("PRAGMA foreign_keys=OFF"))

    op.create_table(
        "raw_artifacts",
        sa.Column("raw_artifact_id", sa.String(length=20), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("canonical_path", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("encoding", sa.String(length=40), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("raw_artifact_id"),
    )
    op.create_index("ix_raw_artifacts_content_hash", "raw_artifacts", ["content_hash"], unique=True)

    with op.batch_alter_table("sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("raw_artifact_id", sa.String(length=20), nullable=True))

    with op.batch_alter_table("run_sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("local_input_path", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("input_kind", sa.String(length=20), server_default="copy", nullable=False)
        )
        batch_op.add_column(sa.Column("artifact_manifest_path", sa.Text(), nullable=True))

    sources = connection.execute(
        sa.text(
            """
            SELECT source_id, raw_hash, raw_file_path, created_at, updated_at, version
            FROM sources
            ORDER BY created_at ASC, source_id ASC
            """
        )
    ).fetchall()
    hash_to_artifact: dict[str, str] = {}
    sequence = connection.execute(
        sa.text("SELECT current_value FROM id_sequences WHERE sequence_key = 'RAW_ARTIFACT'")
    ).fetchone()
    next_value = int(sequence[0]) + 1 if sequence else 1
    if sequence is None:
        connection.execute(
            sa.text(
                "INSERT INTO id_sequences (sequence_key, current_value) VALUES ('RAW_ARTIFACT', 0)"
            )
        )

    for row in sources:
        source_id, raw_hash, raw_file_path, created_at, updated_at, version = row
        artifact_id = hash_to_artifact.get(raw_hash)
        if artifact_id is None:
            artifact_id = f"ART-{next_value:06d}"
            next_value += 1
            hash_to_artifact[raw_hash] = artifact_id
            connection.execute(
                sa.text(
                    """
                    INSERT INTO raw_artifacts (
                        raw_artifact_id, content_hash, canonical_path, byte_size, encoding,
                        first_seen_at, created_at, updated_at, version
                    ) VALUES (
                        :artifact_id, :content_hash, :canonical_path, :byte_size, :encoding,
                        :first_seen_at, :created_at, :updated_at, :version
                    )
                    """
                ),
                {
                    "artifact_id": artifact_id,
                    "content_hash": raw_hash,
                    "canonical_path": raw_file_path,
                    "byte_size": 0,
                    "encoding": "utf-8",
                    "first_seen_at": created_at,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "version": version,
                },
            )
        connection.execute(
            sa.text(
                "UPDATE sources SET raw_artifact_id = :artifact_id WHERE source_id = :source_id"
            ),
            {"artifact_id": artifact_id, "source_id": source_id},
        )
        connection.execute(
            sa.text(
                """
                UPDATE run_sources
                SET local_input_path = (
                    SELECT raw_file_path FROM sources WHERE sources.source_id = run_sources.source_id
                ),
                input_kind = 'copy'
                WHERE source_id = :source_id AND local_input_path IS NULL
                """
            ),
            {"source_id": source_id},
        )

    connection.execute(
        sa.text(
            "UPDATE id_sequences SET current_value = :value WHERE sequence_key = 'RAW_ARTIFACT'"
        ),
        {"value": max(next_value - 1, 0)},
    )

    # unique解除は index 再作成で行い、sources テーブル再作成を避ける
    connection.execute(sa.text("DROP INDEX IF EXISTS ix_sources_raw_hash"))
    connection.execute(
        sa.text("CREATE INDEX IF NOT EXISTS ix_sources_raw_hash ON sources (raw_hash)")
    )
    connection.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_sources_raw_artifact_id ON sources (raw_artifact_id)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_sources_artifact_analyst_medium "
            "ON sources (raw_artifact_id, analyst_id, medium)"
        )
    )

    connection.execute(sa.text("PRAGMA foreign_keys=ON"))


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("PRAGMA foreign_keys=OFF"))
    connection.execute(sa.text("DROP INDEX IF EXISTS ix_sources_artifact_analyst_medium"))
    connection.execute(sa.text("DROP INDEX IF EXISTS ix_sources_raw_artifact_id"))
    with op.batch_alter_table("run_sources", schema=None) as batch_op:
        batch_op.drop_column("artifact_manifest_path")
        batch_op.drop_column("input_kind")
        batch_op.drop_column("local_input_path")
    with op.batch_alter_table("sources", schema=None) as batch_op:
        batch_op.drop_column("raw_artifact_id")
    op.drop_index("ix_raw_artifacts_content_hash", table_name="raw_artifacts")
    op.drop_table("raw_artifacts")
    connection.execute(sa.text("DROP INDEX IF EXISTS ix_sources_raw_hash"))
    connection.execute(
        sa.text("CREATE UNIQUE INDEX IF NOT EXISTS ix_sources_raw_hash ON sources (raw_hash)")
    )
    connection.execute(sa.text("PRAGMA foreign_keys=ON"))
