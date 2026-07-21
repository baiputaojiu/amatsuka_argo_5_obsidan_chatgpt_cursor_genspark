"""workflow tasks and evaluation lookup indexes

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-20 23:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_tasks",
        sa.Column("workflow_task_id", sa.String(length=20), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("task_key", sa.String(length=80), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("executor", sa.String(length=40), nullable=False),
        sa.Column("depends_on", sa.JSON(), nullable=False),
        sa.Column("related_artifact_id", sa.String(length=20), nullable=True),
        sa.Column("related_source_id", sa.String(length=20), nullable=True),
        sa.Column("related_component_id", sa.String(length=20), nullable=True),
        sa.Column("supersedes_task_id", sa.String(length=20), nullable=True),
        sa.Column("resolved_by_task_id", sa.String(length=20), nullable=True),
        sa.Column("retryable", sa.String(length=10), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("recommended_rank", sa.Integer(), nullable=True),
        sa.Column("command_or_prompt", sa.Text(), nullable=True),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["related_artifact_id"], ["ai_artifacts.ai_artifact_id"]),
        sa.ForeignKeyConstraint(
            ["related_component_id"], ["forecast_components.forecast_component_id"]
        ),
        sa.ForeignKeyConstraint(["related_source_id"], ["sources.source_id"]),
        sa.ForeignKeyConstraint(["resolved_by_task_id"], ["workflow_tasks.workflow_task_id"]),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
        sa.ForeignKeyConstraint(["supersedes_task_id"], ["workflow_tasks.workflow_task_id"]),
        sa.PrimaryKeyConstraint("workflow_task_id"),
        sa.UniqueConstraint("run_id", "task_key", name="uq_workflow_task_run_key"),
    )
    with op.batch_alter_table("workflow_tasks", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_workflow_tasks_run_id"), ["run_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_workflow_tasks_status"), ["status"], unique=False)
        batch_op.create_index("ix_workflow_tasks_run_status", ["run_id", "status"], unique=False)

    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.create_index(
            "ix_evaluations_component_as_of_method",
            ["forecast_component_id", "evaluation_as_of", "evaluation_method_version"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.drop_index("ix_evaluations_component_as_of_method")

    with op.batch_alter_table("workflow_tasks", schema=None) as batch_op:
        batch_op.drop_index("ix_workflow_tasks_run_status")
        batch_op.drop_index(batch_op.f("ix_workflow_tasks_status"))
        batch_op.drop_index(batch_op.f("ix_workflow_tasks_run_id"))
    op.drop_table("workflow_tasks")
