"""llm_daily_usage: atomic per-day counter for the global LLM call budget

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12

Generated with `alembic revision --autogenerate`, then reviewed by hand.
Manual addition: backfill counters from existing query_logs (rows with a model
are LLM calls), so deploying this mid-day doesn't reset today's budget to zero.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_daily_usage",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("calls", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("calls >= 0", name=op.f("ck_llm_daily_usage_calls_non_negative")),
        sa.PrimaryKeyConstraint("day", name=op.f("pk_llm_daily_usage")),
    )

    # Manual: carry over LLM calls already made (budget days are UTC days).
    op.execute(
        """
        INSERT INTO llm_daily_usage (day, calls)
        SELECT (created_at AT TIME ZONE 'UTC')::date, count(*)
        FROM query_logs
        WHERE model IS NOT NULL
        GROUP BY 1
        """
    )


def downgrade() -> None:
    op.drop_table("llm_daily_usage")
